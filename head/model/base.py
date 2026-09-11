"""Model-independent HEAD adapter input/output contract."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf


@dataclass(frozen=True)
class ModelInput:
    """Simulator-owned ScenarioDescription and zero-based observed frame.

    Treat the scenario as read-only (private adapter caches are allowed).
    New history-only adapters must not use recorded future states. The legacy
    Pluto adapter declares input_scope="legacy_logged_route" because its route
    builder uses the full recorded ego path; see the root README. Positions:
    metres; headings: radians; velocities: m/s. Ego history is closed-loop.
    """
    scenario: Mapping[str, Any]
    current_step: int

    def __post_init__(self):
        if self.current_step < 0:
            raise ValueError("current_step must be non-negative")


@dataclass(frozen=True)
class Trajectory:
    """One selected WORLD-frame trajectory [T,2] = x,y (optional vx,vy columns), sampled every dt.

    reference_offset: distance from tracked point to vehicle centre along
    heading. Zero for centre; positive for rear axle. Explicit reference
    preserves Pluto's controller rather than silently shifting its path.
    """
    samples: np.ndarray
    dt: float = 0.1
    reference_offset: float = 0.0

    def __post_init__(self):
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] not in (2, 4):
            raise ValueError("Trajectory.samples must be [T >= 2, 2 or 4]: x,y[,vx,vy]")
        if not np.isfinite(samples).all():
            raise ValueError("Trajectory.samples must be finite")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Trajectory.dt must be positive and finite")
        if not np.isfinite(self.reference_offset) or self.reference_offset < 0:
            raise ValueError("Trajectory.reference_offset must be finite and non-negative")
        object.__setattr__(self, "samples", samples)

    def control_position(self, centre, heading):
        position = np.asarray(centre, dtype=np.float32).copy()
        position[:2] -= self.reference_offset * np.array(
            [np.cos(float(heading)), np.sin(float(heading))], dtype=np.float32)
        return position


class BaseAdapter(ABC):
    """Implement inside one algorithm directory; no central registry edits.

    Override initialize for non-PyTorch or differently packaged checkpoints.
    compute_trajectory selects one mode and normalizes units/coordinates.
    """
    input_scope = "history_only"

    def __init__(self, config, device="cpu"):
        self.config, self.device, self.model = config, device, None

    @classmethod
    def load_config(cls):
        import inspect
        return OmegaConf.load(Path(inspect.getfile(cls)).with_name("config.yaml"))

    @property
    def history_steps(self):
        return int(self.config.get("past_len", 21))

    @property
    def dt(self):
        return float(self.config.get("step_interval", 0.1))

    def initialize(self, checkpoint):
        self.model = self.build_model(self.config)
        self.load_weights(checkpoint)

    def build_model(self, config):
        raise NotImplementedError("Implement build_model or override initialize")

    def load_weights(self, checkpoint):
        import torch
        # Trusted checkpoints only: Lightning files may include pickle metadata.
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else None
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Invalid model checkpoint: {checkpoint}")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @abstractmethod
    def compute_trajectory(self, model_input: ModelInput) -> Trajectory:
        """Return a finite, world-frame executable trajectory."""

    def reset(self):
        """Clear per-episode state, without reloading weights."""
