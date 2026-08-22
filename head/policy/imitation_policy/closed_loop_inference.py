"""Benchmark-compatible UniTraj closed-loop inference adapter."""

import numpy as np
import torch

from .unitraj_loader import ensure_unitraj_path


class UniTrajClosedLoopInference:
    def __init__(self, config, model, source=None, device="cpu"):
        ensure_unitraj_path(source)
        try:
            from unitraj.datasets.unitraj_test_dataset import UnitrajTestDataset
        except ImportError as exc:
            raise ImportError(
                "UniTraj closed-loop inference requires its dataset dependencies "
                "(including lightning). Install the UniTraj environment first."
            ) from exc
        self.config = config
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.dataset = UnitrajTestDataset(config)

    @staticmethod
    def _prediction_tensor(output):
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            for key in ("predicted_trajectory", "pred_trajs", "trajectory"):
                if key in output:
                    output = output[key]
                    break
            else:
                raise ValueError("UniTraj output has no predicted trajectory field")
        if not torch.is_tensor(output):
            raise TypeError("UniTraj model output must contain a torch Tensor")
        return output

    @staticmethod
    def _local_to_world(local, center):
        local = np.asarray(local, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        if local.ndim == 2:
            local = local[None]
        xy = local[..., :2]
        origin = center[:2]
        heading = float(center[6]) if len(center) > 6 else 0.0
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
        world = xy @ rotation.T + origin
        return world[0]

    def predict(self, scenario, current_step):
        batch, centers = self.dataset.process_scenario(scenario, int(current_step))
        batch = self._to_device(batch)
        with torch.no_grad():
            output = self.model.forward(batch)
        trajectories = self._prediction_tensor(output).detach().cpu().numpy()
        if trajectories.ndim == 4:
            trajectories = trajectories[0]
        if trajectories.ndim != 3:
            raise ValueError(f"Unsupported UniTraj trajectory shape: {trajectories.shape}")
        selected = trajectories[0]
        center = centers[0] if np.asarray(centers).ndim > 1 else centers
        return self._local_to_world(selected, center)

    def _to_device(self, value):
        if torch.is_tensor(value):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value
