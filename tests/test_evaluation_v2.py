import importlib.util
import unittest
from pathlib import Path

import numpy as np

from evaluation.closed_loop_metrics import PROTOCOL_VERSION, compute_closed_loop_metric_rows


def make_frames(x, vx, other_x=50.0, other_vx=0.0, types=(1, 1), stopped=None):
    x = np.asarray(x, dtype=np.float32)
    vx = np.broadcast_to(np.asarray(vx, dtype=np.float32), x.shape)
    stopped = np.zeros_like(x, dtype=np.int32) if stopped is None else np.asarray(stopped, dtype=np.int32)
    frames = []
    for index in range(len(x)):
        frames.append({
            "x": np.asarray([x[index], other_x], dtype=np.float32),
            "y": np.zeros(2, dtype=np.float32),
            "heading": np.zeros(2, dtype=np.float32),
            "id": np.asarray([1, 2], dtype=np.int32),
            "length": np.full(2, 4.0, dtype=np.float32),
            "width": np.full(2, 2.0, dtype=np.float32),
            "vx": np.asarray([vx[index], other_vx], dtype=np.float32),
            "vy": np.zeros(2, dtype=np.float32),
            "valid": np.ones(2, dtype=np.int32),
            "stopped": np.asarray([stopped[index], int(other_vx == 0.0)], dtype=np.int32),
            "respawn_count": np.zeros(2, dtype=np.int32),
            "type": np.asarray(types, dtype=np.int32),
        })
    return frames


class TestClosedLoopMetrics(unittest.TestCase):
    def test_protocol_version(self):
        self.assertEqual(PROTOCOL_VERSION, "closed_loop_metrics_v2")

    def test_safe_constant_speed_is_comfortable(self):
        times = np.arange(41, dtype=np.float32) * 0.1
        row = compute_closed_loop_metric_rows(
            make_frames(5.0 * times, 5.0), [0, 2], dt=0.1
        )[0]

        self.assertEqual(row["ttc_within_bound_rate"], 1.0)
        self.assertEqual(row["ttc_safe_frame_rate"], 1.0)
        self.assertEqual(row["minimum_ttc_seconds"], 3.0)
        self.assertEqual(row["comfortable_rate"], 1.0)
        self.assertEqual(row["comfort_frame_rate"], 1.0)
        self.assertEqual(row["comfort_metric_agents"], 2)

    def test_directed_ttc_excludes_same_direction_rear_for_lead_vehicle(self):
        row = compute_closed_loop_metric_rows(
            make_frames([0.0, 0.5, 1.0], 5.0, other_x=6.0),
            [0, 2],
            dt=0.1,
        )[0]

        self.assertEqual(row["ttc_within_bound_rate"], 0.5)
        self.assertEqual(row["ttc_safe_frame_rate"], 0.5)
        self.assertAlmostEqual(row["minimum_ttc_seconds"], 0.2, places=5)

    def test_hard_acceleration_reduces_comfort_rates(self):
        times = np.arange(41, dtype=np.float32) * 0.1
        row = compute_closed_loop_metric_rows(
            make_frames(5.0 * times * times, 10.0 * times),
            [0, 2],
            dt=0.1,
        )[0]

        self.assertEqual(row["comfortable_rate"], 0.5)
        self.assertEqual(row["comfort_frame_rate"], 0.5)
        self.assertEqual(row["comfort_longitudinal_acceleration_rate"], 0.5)
        self.assertEqual(row["comfort_longitudinal_acceleration_frame_rate"], 0.5)

    def test_stop_boundary_does_not_create_derivative_spike(self):
        moving = np.arange(21, dtype=np.float32) * 0.5
        x = np.concatenate((moving, np.full(21, moving[-1], dtype=np.float32)))
        vx = np.concatenate((np.full(21, 5.0), np.zeros(21))).astype(np.float32)
        stopped = np.concatenate((np.zeros(21), np.ones(21))).astype(np.int32)
        row = compute_closed_loop_metric_rows(
            make_frames(x, vx, stopped=stopped), [0, 2], dt=0.1
        )[0]

        self.assertEqual(row["comfortable_rate"], 1.0)
        self.assertEqual(row["comfort_frame_rate"], 1.0)

    def test_non_vehicle_is_obstacle_but_not_comfort_focal(self):
        times = np.arange(41, dtype=np.float32) * 0.1
        row = compute_closed_loop_metric_rows(
            make_frames(2.0 * times, 2.0, types=(1, 2)), [0, 2], dt=0.1
        )[0]

        self.assertEqual(row["ttc_metric_agents"], 1)
        self.assertEqual(row["comfort_metric_agents"], 1)
        self.assertEqual(row["comfortable_rate"], 1.0)

    def test_agents_from_different_maps_do_not_interact(self):
        times = np.arange(20, dtype=np.float32) * 0.1
        frames = make_frames(times, 1.0, other_x=0.0, other_vx=1.0)
        rows = compute_closed_loop_metric_rows(frames, [0, 1, 2], dt=0.1)

        self.assertEqual([row["ttc_within_bound_rate"] for row in rows], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
