"""Convert a predicted ego trajectory into MetaDrive actions."""

import math
import numpy as np


class PID:
    def __init__(self, kp, ki, kd, integral_limit=10.0):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit = float(integral_limit)
        self.integral = 0.0
        self.previous = None

    def reset(self):
        self.integral = 0.0
        self.previous = None

    def __call__(self, error, dt):
        self.integral = float(np.clip(self.integral + error * dt,
                                      -self.integral_limit, self.integral_limit))
        derivative = 0.0 if self.previous is None else (error - self.previous) / max(dt, 1e-6)
        self.previous = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


class TrajectoryController:
    """Simple, deterministic trajectory tracker for ego-only closed loop."""

    def __init__(self, config=None):
        config = config or {}
        lateral = config.get("lateral_pid", {})
        longitudinal = config.get("longitudinal_pid", {})
        lookahead = config.get("lookahead", {})
        self.dt = float(config.get("dt", 0.1))
        self.lateral_pid = PID(lateral.get("kp", 0.8), lateral.get("ki", 0.05), lateral.get("kd", 0.0))
        self.longitudinal_pid = PID(longitudinal.get("kp", 2.0), longitudinal.get("ki", 0.0), longitudinal.get("kd", 0.0))
        self.min_lookahead = float(lookahead.get("min_distance", 2.0))
        self.max_lookahead = float(lookahead.get("max_distance", 6.0))
        self.last_match_idx = 0

    def reset(self):
        self.last_match_idx = 0
        self.lateral_pid.reset()
        self.longitudinal_pid.reset()

    @staticmethod
    def _wrap(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def control(self, trajectory, position, heading, speed):
        traj = np.asarray(trajectory, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
            raise ValueError("predicted trajectory must have shape [T, >=2]")
        pos = np.asarray(position, dtype=np.float32)[:2]
        heading = float(heading)
        speed = max(float(speed), 0.0)
        points = traj[:, :2]
        start = max(0, min(self.last_match_idx, len(points) - 1))
        end = min(len(points), start + 20)
        distances = np.linalg.norm(points[start:end] - pos, axis=1)
        match = start + int(np.argmin(distances))
        self.last_match_idx = match

        target = min(match + max(1, int(np.clip(speed * 0.8, self.min_lookahead, self.max_lookahead))), len(points) - 1)
        target_vec = points[target] - pos
        heading_vec = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32)
        lateral = float(heading_vec[0] * (points[match, 1] - pos[1]) - heading_vec[1] * (points[match, 0] - pos[0]))
        tangent_idx = min(match + 1, len(points) - 1)
        tangent = points[tangent_idx] - points[match]
        traj_heading = math.atan2(float(tangent[1]), float(tangent[0])) if np.linalg.norm(tangent) > 1e-5 else heading
        heading_error = self._wrap(traj_heading - heading)
        steering = self.lateral_pid(heading_error - math.atan2(1.5 * lateral, max(speed, 1.0)), self.dt)

        if traj.shape[1] >= 4:
            target_speed = float(np.linalg.norm(traj[target, 2:4]))
        else:
            distance = float(np.linalg.norm(points[target] - points[match]))
            target_speed = distance / max((target - match) * self.dt, self.dt)
        throttle_brake = self.longitudinal_pid(target_speed - speed, self.dt) / 5.0
        return [float(np.clip(steering, -1.0, 1.0)), float(np.clip(throttle_brake, -1.0, 1.0))]
