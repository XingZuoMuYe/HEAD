"""Model-independent ego policy: history, planning, control and evaluation."""

import numpy as np
import torch
from metadrive.policy.base_policy import BasePolicy
from metadrive.scenario.parse_object_state import parse_object_state

from .closed_loop_inference import ClosedLoopInference
from .trajectory_controller import TrajectoryController
from head.manager.artifact_paths import resolve_imitation_checkpoint


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
        imitation_cfg = self.head_cfg.args.workflow.policies.imitation
        checkpoint = resolve_imitation_checkpoint(self.head_cfg.args)
        self._inference = ClosedLoopInference(
            imitation_cfg, checkpoint, device=self.device,
            controller_dt=self.controller.dt,
        )
        self._model = self._inference.adapter.model
        self.warmup_steps = max(self.warmup_steps, self._inference.adapter.history_steps)
        # Episode horizon is not the network prediction window.
        limit = imitation_cfg.get("max_closed_loop_steps", None)
        self.max_closed_loop_steps = int(
            limit if limit is not None else self.head_cfg.args.evaluation.max_steps
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
        self._inference.reset()
