"""MetaDrive ego policy for UniTraj closed-loop deployment."""

import numpy as np
import torch
from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.parse_object_state import parse_object_state

from .closed_loop_inference import UniTrajClosedLoopInference
from .pluto_closed_loop_inference import PlutoClosedLoopInference
from .trajectory_controller import TrajectoryController
from head.manager.artifact_paths import resolve_imitation_checkpoint
from head.manager.imitation_selector import resolve_imitation_strategy


class ImitationPlanningPolicy(BasePolicy):
    _head_cfg = None

    @classmethod
    def configure(cls, cfg):
        cls._head_cfg = cfg

    def __init__(self, control_object, random_seed):
        super().__init__(control_object, random_seed)
        self.head_cfg = self.__class__._head_cfg
        if self.head_cfg is None:
            raise RuntimeError("ImitationPlanningPolicy requires the HEAD configuration")
        imitation_cfg = self.head_cfg.args.workflow.policies.imitation
        self.warmup_steps = int(imitation_cfg.get("warmup_steps", 10))
        self.replan_frequency = int(imitation_cfg.get("replan_frequency", 5))
        self._prediction = None
        self.controller = TrajectoryController(imitation_cfg.get("controller", {}))
        requested = self.head_cfg.args.runtime.device
        self.device = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
        if self.device == "auto":
            self.device = "cpu"
        self._initialize_model()

    def _initialize_model(self):
        self._model, unitraj_cfg = resolve_imitation_strategy(self.head_cfg)
        # UnitrajTestDataset slices a full past window ending at current_step.
        self.warmup_steps = max(self.warmup_steps, int(unitraj_cfg.get("past_len", 21)))
        self.max_closed_loop_steps = (
            int(unitraj_cfg.get("past_len", 21))
            + int(unitraj_cfg.get("future_len", 60))
            - 1
        )
        # Dataset window length is not the simulator horizon. Keep historical
        # default for existing pilots; explicit longer runs still cap by scene.
        configured_limit = self.head_cfg.args.workflow.policies.imitation.get("max_closed_loop_steps", None)
        if configured_limit is not None:
            self.max_closed_loop_steps = int(configured_limit)
        imitation_cfg = self.head_cfg.args.workflow.policies.imitation
        checkpoint = resolve_imitation_checkpoint(self.head_cfg.args)
        source = imitation_cfg.get("source", None)
        loaded = torch.load(checkpoint, map_location=self.device, weights_only=False)
        state_dict = loaded.get("state_dict") if isinstance(loaded, dict) else None
        if state_dict is None:
            raise ValueError(f"Invalid imitation checkpoint '{checkpoint}': expected 'state_dict'")
        model_name = str(imitation_cfg.get("model", "")).lower()
        if model_name == "pluto":
            state_dict = {
                (key[len("model."):] if key.startswith("model.") else key): value
                for key, value in state_dict.items()
            }
            incompatible = self._model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                raise ValueError(f"Pluto checkpoint is missing weights: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[Pluto] ignored {len(incompatible.unexpected_keys)} unexpected checkpoint keys")
        else:
            self._model.load_state_dict(state_dict)
        self._model.to(self.device).eval()
        self.sae_experiment = None
        sae_cfg = imitation_cfg.get("sae", {})
        if sae_cfg.get("mode", "off") not in ("off", False):
            if model_name != "pluto":
                raise ValueError("SAE experiment currently supports Pluto only")
            from head.research.pluto_sae import FeatureExperiment
            self.sae_experiment = FeatureExperiment(sae_cfg, self._model, self.device)
        inference_class = (
            PlutoClosedLoopInference if model_name == "pluto"
            else UniTrajClosedLoopInference
        )
        self._inference = inference_class(
            unitraj_cfg,
            self._model,
            source=source,
            device=self.device,
        )

    @property
    def model(self):
        return self._model

    def _warmup(self, time_index):
        scenario = self.engine.data_manager.current_scenario
        sdc_id = str(scenario["metadata"]["sdc_id"])
        state = parse_object_state(scenario["tracks"][sdc_id], time_index)
        if state and state.get("valid", False):
            self.control_object.set_position(state["position"])
            self.control_object.set_velocity(state["velocity"])
            self.control_object.set_heading_theta(state["heading"])

    def _update_ego_history(self, time_index):
        """Feed the realized ego state into the next closed-loop replan."""
        scenario = self.engine.data_manager.current_scenario
        sdc_id = str(scenario["metadata"]["sdc_id"])
        state = scenario["tracks"][sdc_id]["state"]
        if time_index >= len(state["position"]):
            return
        position = np.asarray(state["position"])
        vehicle_xy = np.asarray(self.control_object.position, dtype=position.dtype)[:2]
        position[time_index, :2] = vehicle_xy
        state["position"] = position
        velocity = np.asarray(state["velocity"])
        velocity[time_index, :2] = np.asarray(self.control_object.velocity, dtype=velocity.dtype)[:2]
        state["velocity"] = velocity
        heading = np.asarray(state["heading"])
        if heading.ndim == 1:
            heading[time_index] = self.control_object.heading_theta
        else:
            heading[time_index, 0] = self.control_object.heading_theta
        state["heading"] = heading
        valid = np.asarray(state["valid"])
        valid[time_index] = True
        state["valid"] = valid

    def act(self, agent_id):
        self.action_info.clear()
        current_step = int(self.engine.episode_step)
        time_index = max(current_step - 1, 0)
        if time_index < self.warmup_steps:
            self._warmup(time_index)
            self.action_info["closed_loop_stage"] = "warmup"
            return None
        self._update_ego_history(time_index)
        if self._prediction is None or time_index % self.replan_frequency == 0:
            if self.sae_experiment is not None:
                scenario = self.engine.data_manager.current_scenario
                self.sae_experiment.context = (str(scenario["id"]), time_index)
            self._prediction = self._inference.predict(self.engine.data_manager.current_scenario, time_index)
            self.controller.reset()
            self.control_object.plan_traj = self._prediction[:, :2]
        control_position = self.control_object.position
        if hasattr(self._inference, "control_position"):
            control_position = self._inference.control_position(
                control_position, self.control_object.heading_theta
            )
        action = self.controller.control(
            self._prediction,
            control_position,
            self.control_object.heading_theta,
            self.control_object.speed,
        )
        self.action_info.update({"action": action, "closed_loop_stage": "inference"})
        return action

    def before_reset(self):
        self._prediction = None
        self.controller.reset()
