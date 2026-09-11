import numpy as np

from head.policy.imitation_policy.trajectory_controller import TrajectoryController
from head.model import ModelInput, Trajectory, get_adapter_class
from head.model.pluto.adapter import Adapter as PlutoAdapter


def test_trajectory_controller_returns_bounded_action():
    controller = TrajectoryController({"dt": 0.1})
    trajectory = np.stack([np.arange(20), np.zeros(20), np.ones(20) * 5, np.zeros(20)], axis=1)
    action = controller.control(trajectory, [0.0, 0.0], 0.0, 2.0)
    assert len(action) == 2
    assert all(-1.0 <= value <= 1.0 for value in action)


def test_trajectory_controller_steers_toward_lateral_path_error():
    controller = TrajectoryController({"dt": 0.1})
    left_path = np.stack(
        [np.arange(20), np.ones(20), np.ones(20) * 5, np.zeros(20)],
        axis=1,
    )
    right_path = left_path.copy()
    right_path[:, 1] = -1.0

    left_action = controller.control(left_path, [0.0, 0.0], 0.0, 5.0)
    controller.reset()
    right_action = controller.control(right_path, [0.0, 0.0], 0.0, 5.0)

    assert left_action[0] > 0.0
    assert right_action[0] < 0.0


def test_adapter_name_validation():
    try:
        get_adapter_class("missing_adapter")
    except ValueError as exc:
        assert "Unknown adapter" in str(exc)
    else:
        raise AssertionError("invalid adapter should be rejected")


def test_pluto_and_wayformer_are_discoverable():
    assert get_adapter_class("pluto").load_config().dim == 128
    assert get_adapter_class("wayformer").load_config().hidden_size == 256


def test_pluto_adapter_uses_velocity_columns():
    class FakeEngine:
        def run_inference(self, scenario, current_step):
            trajectory = np.zeros((16, 4, 6), dtype=np.float32)
            trajectory[0, :, 0] = np.arange(4)
            trajectory[0, :, 2:4] = [1.0, 0.0]
            trajectory[0, :, 4:6] = [3.0, 4.0]
            return trajectory, None, None, {}

    adapter = PlutoAdapter({})
    adapter.engine = FakeEngine()
    trajectory = adapter.compute_trajectory(ModelInput({}, 21)).samples
    assert trajectory.shape == (4, 4)
    np.testing.assert_allclose(
        trajectory[:, 2:4], np.tile([3.0, 4.0], (4, 1))
    )


def test_pluto_control_position_uses_rear_axle():
    trajectory = Trajectory(np.zeros((2, 2)), reference_offset=2.0)
    position = trajectory.control_position([10.0, 5.0], 0.0)
    np.testing.assert_allclose(position[:2], [8.0, 5.0])


def test_pluto_collision_scoring_converts_rear_axle_to_vehicle_center():
    from head.model.pluto.trajectory_evaluator import TrajectoryEvaluator

    trajectory = np.zeros((1, 4, 2), dtype=np.float32)
    yaw = np.zeros((1, 4), dtype=np.float32)
    agent_position = np.zeros((1, 2, 21, 2), dtype=np.float32)
    agent_position[0, 1, :, 0] = 5.0
    input_data = {
        "agent": {
            "position": agent_position,
            "valid_mask": np.ones((1, 2, 21), dtype=bool),
            "heading": np.zeros((1, 2, 21), dtype=np.float32),
        }
    }
    predictions = np.zeros((1, 4, 3), dtype=np.float32)
    predictions[0, :, 0] = 5.0

    rear_referenced = TrajectoryEvaluator({"rear_axle_to_center": 2.0})
    center_referenced = TrajectoryEvaluator({"rear_axle_to_center": 0.0})

    assert rear_referenced._has_collision(
        trajectory, yaw, input_data, predictions
    ).tolist() == [True]
    assert center_referenced._has_collision(
        trajectory, yaw, input_data, predictions
    ).tolist() == [False]

    rear_referenced.evaluate(
        np.concatenate([trajectory, yaw[..., None]], axis=-1),
        input_data,
        predictions,
    )
    assert rear_referenced.last_hard_fail_mask.tolist() == [True]
