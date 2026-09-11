"""Pluto-specific loading, feature construction and trajectory conversion."""
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from head.model.base import BaseAdapter, Trajectory
from head.model.common.scenario import normalize_scalar_states


class Adapter(BaseAdapter):
    # Preserve the published experiment's route inference for this refactor.
    # This is explicitly NOT a strict history-only planning protocol.
    input_scope = "legacy_logged_route"

    @classmethod
    def load_config(cls):
        return OmegaConf.merge(
            OmegaConf.load(Path(__file__).parents[1] / "common/config.yaml"),
            super().load_config())

    def build_model(self, config):
        from .model.pluto_model import PlanningModel
        return PlanningModel(config=config)

    def load_weights(self, checkpoint):
        import torch
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else None
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Invalid Pluto checkpoint: {checkpoint}")
        state = {(k[6:] if k.startswith("model.") else k): v for k, v in state.items()}
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Pluto checkpoint missing weights: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            import warnings
            warnings.warn(f"Pluto ignored checkpoint keys: {incompatible.unexpected_keys}")
        self.model.to(self.device).eval()

    def initialize(self, checkpoint):
        from .planner import PlutoInference
        super().initialize(checkpoint)
        self.engine = PlutoInference(self.config)
        self.engine.device = str(self.device)
        self.engine.model = self.model
        self.sae_experiment = None
        sae_cfg = self.config.get("sae", {})
        if sae_cfg.get("mode", "off") not in ("off", False):
            try:
                from head.research.pluto_sae import FeatureExperiment
            except ImportError as exc:
                raise ImportError("SAE requires the separate research checkout; not bundled in this release") from exc
            self.sae_experiment = FeatureExperiment(sae_cfg, self.model, self.device)

    @staticmethod
    def reference_offset(scenario, step):
        metadata = scenario.get("metadata", {})
        parameters = metadata.get("ego_vehicle_parameters") or {}
        if parameters.get("rear_axle_to_center") is not None:
            return float(parameters["rear_axle_to_center"])
        if {"front_length", "rear_length"} <= parameters.keys():
            return (float(parameters["front_length"]) - float(parameters["rear_length"])) / 2
        track = scenario.get("tracks", {}).get(str(metadata.get("sdc_id", "ego")), {})
        lengths = np.asarray(track.get("state", {}).get("length", [4.8])).reshape(-1)
        return 0.35 * float(lengths[min(step, lengths.size - 1)]) if lengths.size else 1.67

    def compute_trajectory(self, model_input):
        if getattr(self, "sae_experiment", None) is not None:
            self.sae_experiment.context = (str(model_input.scenario["id"]), model_input.current_step)
        scenario = normalize_scalar_states(model_input.scenario)
        output = self.engine.run_inference(scenario, model_input.current_step)
        trajectories = np.asarray(output[0], dtype=np.float32)
        trajectory = trajectories[0] if trajectories.ndim == 3 else trajectories
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
            raise ValueError(f"Unsupported Pluto trajectory shape: {trajectories.shape}")
        if trajectory.shape[1] >= 6:
            # Native layout: x,y,cos(yaw),sin(yaw),vx,vy, not x,y,vx,vy.
            samples = np.concatenate([trajectory[:, :2], trajectory[:, 4:6]], axis=-1)
        elif trajectory.shape[1] == 3:
            samples = np.concatenate([
                trajectory[:, :2], np.gradient(trajectory[:, :2], self.dt, axis=0)
            ], axis=-1)
        else:
            samples = trajectory
        return Trajectory(samples, self.dt, self.reference_offset(model_input.scenario, model_input.current_step))

    def reset(self):
        from .features.builder import PlutoTestDataset
        self.engine.dataset = PlutoTestDataset(self.config, is_validation=True)
