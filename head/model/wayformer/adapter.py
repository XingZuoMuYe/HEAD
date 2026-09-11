"""WayFormer forecasting modes adapted to ego-only HEAD control."""
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from head.model.base import BaseAdapter, Trajectory
from head.model.common.scenario import normalize_scalar_states


class Adapter(BaseAdapter):
    @classmethod
    def load_config(cls):
        return OmegaConf.merge(
            OmegaConf.load(Path(__file__).parents[1] / "common/config.yaml"),
            super().load_config())

    def build_model(self, config):
        from .model import Wayformer
        return Wayformer(config=config)

    def initialize(self, checkpoint):
        if self.config.get("sae", {}).get("mode", "off") not in ("off", False):
            raise ValueError("This WayFormer adapter does not support the Pluto SAE hook")
        super().initialize(checkpoint)
        self.reset()

    def reset(self):
        from .features import UnitrajTestDataset
        self.dataset = UnitrajTestDataset(self.config)

    def _to_device(self, value):
        import torch
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._to_device(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._to_device(v) for v in value)
        return value

    @staticmethod
    def select_trajectory(output):
        import torch
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, dict):
            raise TypeError("WayFormer must return a trajectory/probability dictionary")
        trajectories = output["predicted_trajectory"]
        probabilities = output["predicted_probability"]
        if trajectories.ndim != 4 or probabilities.shape != trajectories.shape[:2]:
            raise ValueError("WayFormer expects trajectories [B,M,T,D] and probabilities [B,M]")
        if trajectories.shape[0] != 1:
            raise ValueError("HEAD ego adapter expects exactly one prediction target")
        if not torch.isfinite(probabilities).all():
            raise ValueError("WayFormer mode probabilities must be finite")
        mode = int(probabilities[0].argmax().item())
        return trajectories[0, mode, :, :2].detach().cpu().numpy()

    def compute_trajectory(self, model_input):
        import torch
        scenario = normalize_scalar_states(model_input.scenario)
        batch, centres = self.dataset.process_scenario(scenario, model_input.current_step)
        with torch.inference_mode():
            local = self.select_trajectory(self.model(self._to_device(batch)))
        centre = np.asarray(centres)
        centre = centre[0] if centre.ndim > 1 else centre
        heading = float(centre[6])
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = local @ rotation.T + centre[:2]
        # Native Gaussian dispersion columns are NOT velocities.
        # Preserve the controller's geometric lookahead speed estimate.
        return Trajectory(world, self.dt)
