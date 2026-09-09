"""Adapter between the vendored Pluto planner and HEAD's ego controller."""

import numpy as np
import torch

from .unitraj_loader import ensure_unitraj_path


class PlutoClosedLoopInference:
    """Run Pluto candidate generation and expose one world-frame ego trajectory."""

    def __init__(self, config, model, source=None, device="cpu"):
        ensure_unitraj_path(source)
        from unitraj.closeloop.pluto.pluto_inference import PlutoInference

        self.device = torch.device(device)
        self.engine = PlutoInference(config)
        self.engine.device = str(self.device)
        self.engine.model = model.to(self.device).eval()
        self.rear_axle_to_center = 1.67

    def predict(self, scenario, current_step):
        metadata = scenario.get("metadata", {})
        parameters = metadata.get("ego_vehicle_parameters") or {}
        configured_offset = parameters.get("rear_axle_to_center")
        if configured_offset is None and {'front_length', 'rear_length'} <= parameters.keys():
            configured_offset = (float(parameters['front_length']) - float(parameters['rear_length'])) / 2
        if configured_offset is not None:
            self.rear_axle_to_center = float(configured_offset)
        else:
            sdc_id = str(metadata.get("sdc_id", "ego"))
            track = scenario.get("tracks", {}).get(sdc_id, {})
            lengths = np.asarray(track.get("state", {}).get("length", [4.8]))
            if lengths.size:
                index = min(max(int(current_step), 0), lengths.size - 1)
                self.rear_axle_to_center = 0.35 * float(lengths.reshape(-1)[index])
        output = self.engine.run_inference(scenario, int(current_step))
        trajectories = np.asarray(output[0], dtype=np.float32)
        if trajectories.ndim == 3:
            trajectory = trajectories[0]
        elif trajectories.ndim == 2:
            trajectory = trajectories
        else:
            raise ValueError(f"Unsupported Pluto trajectory shape: {trajectories.shape}")
        if trajectory.shape[0] < 2 or trajectory.shape[1] < 2:
            raise ValueError(f"Pluto returned an empty trajectory: {trajectory.shape}")

        # Pluto's control reference is [x, y, cos(yaw), sin(yaw), vx, vy].
        # HEAD's controller interprets columns 2:4 as velocity, so normalize it.
        if trajectory.shape[1] >= 6:
            trajectory = np.concatenate(
                [trajectory[:, :2], trajectory[:, 4:6]], axis=-1
            )
        elif trajectory.shape[1] == 3:
            velocity = np.gradient(trajectory[:, :2], 0.1, axis=0)
            trajectory = np.concatenate([trajectory[:, :2], velocity], axis=-1)
        return trajectory

    def control_position(self, center_position, heading):
        """Return MetaDrive vehicle position expressed at Pluto's rear axle."""
        position = np.asarray(center_position, dtype=np.float32).copy()
        position[:2] -= self.rear_axle_to_center * np.array(
            [np.cos(float(heading)), np.sin(float(heading))], dtype=np.float32
        )
        return position
