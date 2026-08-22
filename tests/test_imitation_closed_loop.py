import numpy as np

from head.policy.imitation_policy.trajectory_controller import TrajectoryController
from head.policy.imitation_policy.unitraj_loader import ensure_unitraj_path


def test_trajectory_controller_returns_bounded_action():
    controller = TrajectoryController({"dt": 0.1})
    trajectory = np.stack([np.arange(20), np.zeros(20), np.ones(20) * 5, np.zeros(20)], axis=1)
    action = controller.control(trajectory, [0.0, 0.0], 0.0, 2.0)
    assert len(action) == 2
    assert all(-1.0 <= value <= 1.0 for value in action)


def test_unitraj_source_validation(tmp_path):
    missing = tmp_path / "missing"
    try:
        ensure_unitraj_path(missing)
    except FileNotFoundError as exc:
        assert "unitraj" in str(exc)
    else:
        raise AssertionError("invalid UniTraj source should be rejected")
