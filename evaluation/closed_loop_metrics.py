"""Additive closed-loop TTC and comfort metrics for Drive evaluations.

These diagnostics do not participate in PufferDrive's native score, reward,
termination, or policy action selection. Version 2 uses native velocities,
vehicle-only focal agents, lifecycle-aware smoothing, and directed TTC.
"""

import math

import numpy as np


PROTOCOL_VERSION = "closed_loop_metrics_v2"
VEHICLE_TYPE = 1
TTC_HORIZON_SECONDS = 3.0
TTC_THRESHOLD_SECONDS = 0.95
TTC_PROJECTION_STEP_SECONDS = 0.1
MIN_MOVING_SPEED = 0.05

COMFORT_THRESHOLDS = {
    "min_longitudinal_acceleration": -4.05,
    "max_longitudinal_acceleration": 2.40,
    "max_absolute_lateral_acceleration": 4.89,
    "max_absolute_yaw_acceleration": 1.93,
    "max_absolute_yaw_rate": 0.95,
    "max_absolute_longitudinal_jerk": 4.13,
    "max_jerk_magnitude": 8.37,
}

COMFORT_COMPONENTS = (
    "longitudinal_acceleration",
    "lateral_acceleration",
    "yaw_acceleration",
    "yaw_rate",
    "longitudinal_jerk",
    "jerk_magnitude",
)

ADDITIONAL_METRIC_NAMES = (
    "ttc_within_bound_rate",
    "ttc_safe_frame_rate",
    "mean_min_ttc_seconds",
    "comfortable_rate",
    "comfort_frame_rate",
    *(f"comfort_{name}_rate" for name in COMFORT_COMPONENTS),
    *(f"comfort_{name}_frame_rate" for name in COMFORT_COMPONENTS),
)

METRIC_WEIGHT_KEYS = {
    "ttc_within_bound_rate": "ttc_metric_agents",
    "mean_min_ttc_seconds": "ttc_metric_agents",
    "ttc_safe_frame_rate": "ttc_metric_frames",
    "comfortable_rate": "comfort_metric_agents",
    "comfort_frame_rate": "comfort_metric_frames",
}
METRIC_WEIGHT_KEYS.update(
    {f"comfort_{name}_rate": "comfort_metric_agents" for name in COMFORT_COMPONENTS}
)
METRIC_WEIGHT_KEYS.update(
    {f"comfort_{name}_frame_rate": "comfort_metric_frames" for name in COMFORT_COMPONENTS}
)

STATE_FIELDS = (
    "x", "y", "heading", "id", "length", "width",
    "vx", "vy", "valid", "stopped", "respawn_count", "type",
)


def metric_metadata(dt):
    """Return serializable protocol metadata for evaluation summaries."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "additional_metrics_only": True,
        "state_sample_period_seconds": float(dt),
        "focal_agent_types": ["vehicle"],
        "ttc": {
            "threshold_seconds": TTC_THRESHOLD_SECONDS,
            "prediction_horizon_seconds": TTC_HORIZON_SECONDS,
            "projection_step_seconds": TTC_PROJECTION_STEP_SECONDS,
            "motion_model": "native_velocity_constant_heading",
            "geometry": "oriented_boxes",
            "interaction_scope": "vehicle_focal_with_all_active_agent_obstacles",
            "relevance": "ahead_or_heading_crossing; same_direction_rear_excluded",
            "outputs": {
                "ttc_within_bound_rate": "vehicle trajectories with minimum TTC >= threshold",
                "ttc_safe_frame_rate": "vehicle frames with TTC >= threshold",
            },
        },
        "comfort": {
            "kinematics_source": "native_velocity_and_executed_heading",
            "smoothing": {
                "acceleration_and_yaw_rate": "Savitzky-Golay window=7 polyorder=2",
                "jerk_and_yaw_acceleration": "Savitzky-Golay window=15 polyorder=3",
            },
            "lifecycle_boundaries": ["identity", "valid", "stopped", "respawn_count"],
            "strict_metric": "all valid frames pass all six components",
            "frame_metric": "fraction of valid frames passing all six components",
            "thresholds": dict(COMFORT_THRESHOLDS),
        },
    }


class ClosedLoopMetricRecorder:
    """Collect lightweight Drive states and compute one metric row per map."""

    def __init__(self, agent_offsets, dt=0.1):
        offsets = np.asarray(agent_offsets, dtype=np.int64)
        if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
            raise ValueError("agent_offsets must be a cumulative 1D array starting at zero")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("agent_offsets must be non-decreasing")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.agent_offsets = offsets.copy()
        self.dt = float(dt)
        self._frames = []

    def record(self, state):
        expected = int(self.agent_offsets[-1])
        frame = {}
        for key in STATE_FIELDS:
            if key not in state:
                raise KeyError(
                    f"Global agent state is missing {key!r}; rebuild the Drive binding "
                    "for closed_loop_metrics_v2"
                )
            values = np.asarray(state[key])
            if values.shape != (expected,):
                raise ValueError(
                    f"Global agent state {key!r} has shape {values.shape}; expected {(expected,)}"
                )
            frame[key] = values.copy()
        self._frames.append(frame)

    def finalize(self):
        return compute_closed_loop_metric_rows(self._frames, self.agent_offsets, self.dt)


def _savgol_kernel(window, polyorder, derivative, dt):
    if window % 2 != 1 or window <= polyorder:
        raise ValueError("Savitzky-Golay window must be odd and exceed polyorder")
    offsets = np.arange(-(window // 2), window // 2 + 1, dtype=np.float64)
    design = np.vander(offsets, polyorder + 1, increasing=True)
    return math.factorial(derivative) * np.linalg.pinv(design)[derivative] / (dt ** derivative)


def _segment_derivative(
    values, segment_mask, identities, respawns, stopped, dt,
    window, polyorder, derivative,
):
    """Differentiate stable lifecycle segments with local polynomial smoothing."""
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    kernel = _savgol_kernel(window, polyorder, derivative, dt)
    time_count, slot_count = values.shape
    for slot in range(slot_count):
        finite = segment_mask[:, slot] & np.isfinite(values[:, slot])
        start = 0
        while start < time_count:
            while start < time_count and not finite[start]:
                start += 1
            if start >= time_count:
                break
            end = start + 1
            while (
                end < time_count
                and finite[end]
                and identities[end, slot] == identities[start, slot]
                and respawns[end, slot] == respawns[start, slot]
                and stopped[end, slot] == stopped[start, slot]
            ):
                end += 1
            if end - start >= window:
                data = values[start:end, slot]
                estimates = np.convolve(data, kernel[::-1], mode="valid")
                half = window // 2
                output[start + half : end - half, slot] = estimates
            start = end
    return output


def _unwrap_heading_by_segment(heading, segment_mask, identities, respawns, stopped):
    heading = np.asarray(heading, dtype=np.float64)
    output = np.full(heading.shape, np.nan, dtype=np.float64)
    time_count, slot_count = heading.shape
    for slot in range(slot_count):
        finite = segment_mask[:, slot] & np.isfinite(heading[:, slot])
        start = 0
        while start < time_count:
            while start < time_count and not finite[start]:
                start += 1
            if start >= time_count:
                break
            end = start + 1
            while (
                end < time_count
                and finite[end]
                and identities[end, slot] == identities[start, slot]
                and respawns[end, slot] == respawns[start, slot]
                and stopped[end, slot] == stopped[start, slot]
            ):
                end += 1
            output[start:end, slot] = np.unwrap(heading[start:end, slot])
            start = end
    return output


def _comfort_results(vx, vy, heading, identities, respawns, valid, stopped, types, dt):
    vehicle_valid = valid & (types == VEHICLE_TYPE)
    heading = _unwrap_heading_by_segment(
        heading, vehicle_valid, identities, respawns, stopped
    )
    args = (vehicle_valid, identities, respawns, stopped, dt)
    acceleration_x = _segment_derivative(vx, *args, 7, 2, 1)
    acceleration_y = _segment_derivative(vy, *args, 7, 2, 1)
    jerk_x = _segment_derivative(vx, *args, 15, 3, 2)
    jerk_y = _segment_derivative(vy, *args, 15, 3, 2)
    yaw_rate = _segment_derivative(heading, *args, 7, 2, 1)
    yaw_acceleration = _segment_derivative(heading, *args, 15, 3, 2)

    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    values = {
        "longitudinal_acceleration": acceleration_x * cos_heading + acceleration_y * sin_heading,
        "lateral_acceleration": -acceleration_x * sin_heading + acceleration_y * cos_heading,
        "yaw_acceleration": yaw_acceleration,
        "yaw_rate": yaw_rate,
        "longitudinal_jerk": jerk_x * cos_heading + jerk_y * sin_heading,
        "jerk_magnitude": np.hypot(jerk_x, jerk_y),
    }
    checks = {
        "longitudinal_acceleration": (
            (values["longitudinal_acceleration"] >= COMFORT_THRESHOLDS["min_longitudinal_acceleration"])
            & (values["longitudinal_acceleration"] <= COMFORT_THRESHOLDS["max_longitudinal_acceleration"])
        ),
        "lateral_acceleration": np.abs(values["lateral_acceleration"])
        <= COMFORT_THRESHOLDS["max_absolute_lateral_acceleration"],
        "yaw_acceleration": np.abs(values["yaw_acceleration"])
        <= COMFORT_THRESHOLDS["max_absolute_yaw_acceleration"],
        "yaw_rate": np.abs(values["yaw_rate"])
        <= COMFORT_THRESHOLDS["max_absolute_yaw_rate"],
        "longitudinal_jerk": np.abs(values["longitudinal_jerk"])
        <= COMFORT_THRESHOLDS["max_absolute_longitudinal_jerk"],
        "jerk_magnitude": values["jerk_magnitude"]
        <= COMFORT_THRESHOLDS["max_jerk_magnitude"],
    }
    common_valid = vehicle_valid.copy()
    for value in values.values():
        common_valid &= np.isfinite(value)
    all_checks = np.logical_and.reduce(tuple(checks.values()))
    agent_has_samples = np.any(common_valid, axis=0)
    strict = agent_has_samples & np.all((~common_valid) | all_checks, axis=0)
    component_strict = {
        name: agent_has_samples & np.all((~common_valid) | check, axis=0)
        for name, check in checks.items()
    }
    return common_valid, all_checks, strict, checks, component_strict, agent_has_samples


def _directed_frame_ttc(positions, velocities, headings, lengths, widths):
    """Return directed TTC for every focal candidate in one vectorized frame."""
    count = len(positions)
    result = np.full(count, TTC_HORIZON_SECONDS, dtype=np.float64)
    if count < 2:
        return result

    first, second = np.triu_indices(count, 1)
    delta_position = positions[second] - positions[first]
    delta_velocity = velocities[second] - velocities[first]
    first_heading = headings[first]
    second_heading = headings[second]
    first_long = np.column_stack((np.cos(first_heading), np.sin(first_heading)))
    first_lat = np.column_stack((-np.sin(first_heading), np.cos(first_heading)))
    second_long = np.column_stack((np.cos(second_heading), np.sin(second_heading)))
    second_lat = np.column_stack((-np.sin(second_heading), np.cos(second_heading)))
    axes = (first_long, first_lat, second_long, second_lat)
    projections = np.stack([
        np.einsum("ij,ij->i", delta_position, axis) for axis in axes
    ])
    velocity_projections = np.stack([
        np.einsum("ij,ij->i", delta_velocity, axis) for axis in axes
    ])

    first_half_length = 0.5 * lengths[first]
    first_half_width = 0.5 * widths[first]
    second_half_length = 0.5 * lengths[second]
    second_half_width = 0.5 * widths[second]
    heading_delta = second_heading - first_heading
    abs_cos = np.abs(np.cos(heading_delta))
    abs_sin = np.abs(np.sin(heading_delta))
    radii = np.stack((
        first_half_length + second_half_length * abs_cos + second_half_width * abs_sin,
        first_half_width + second_half_length * abs_sin + second_half_width * abs_cos,
        second_half_length + first_half_length * abs_cos + first_half_width * abs_sin,
        second_half_width + first_half_length * abs_sin + first_half_width * abs_cos,
    ))
    moving_axis = np.abs(velocity_projections) > 1e-8
    safe_velocity = np.where(moving_axis, velocity_projections, 1.0)
    first_crossing = (-radii - projections) / safe_velocity
    second_crossing = (radii - projections) / safe_velocity
    lower = np.where(moving_axis, np.minimum(first_crossing, second_crossing), -np.inf)
    upper = np.where(moving_axis, np.maximum(first_crossing, second_crossing), np.inf)
    impossible = np.any((~moving_axis) & (np.abs(projections) > radii), axis=0)
    entry = np.maximum(np.max(lower, axis=0), 0.0)
    exit_time = np.minimum(np.min(upper, axis=0), TTC_HORIZON_SECONDS)
    collides = (~impossible) & (entry <= exit_time) & (exit_time >= 0.0)
    quantized_entry = np.minimum(
        TTC_HORIZON_SECONDS,
        np.ceil((entry - 1e-6) / TTC_PROJECTION_STEP_SECONDS)
        * TTC_PROJECTION_STEP_SECONDS,
    )

    crossing = np.abs(np.sin(heading_delta)) >= 0.5
    first_ahead = np.einsum("ij,ij->i", delta_position, first_long) > 0.0
    second_ahead = np.einsum("ij,ij->i", -delta_position, second_long) > 0.0
    speed = np.linalg.norm(velocities, axis=1)
    first_relevant = collides & (speed[first] >= MIN_MOVING_SPEED) & (first_ahead | crossing)
    second_relevant = collides & (speed[second] >= MIN_MOVING_SPEED) & (second_ahead | crossing)
    np.minimum.at(result, first[first_relevant], quantized_entry[first_relevant])
    np.minimum.at(result, second[second_relevant], quantized_entry[second_relevant])
    return result


def _ttc_results(x, y, vx, vy, heading, lengths, widths, valid, types):
    time_count, slot_count = x.shape
    frame_valid = np.zeros((time_count, slot_count), dtype=bool)
    frame_safe = np.zeros((time_count, slot_count), dtype=bool)
    frame_ttc = np.full((time_count, slot_count), np.nan, dtype=np.float64)
    for frame_index in range(time_count):
        obstacle_valid = (
            valid[frame_index]
            & np.isfinite(x[frame_index])
            & np.isfinite(y[frame_index])
            & np.isfinite(vx[frame_index])
            & np.isfinite(vy[frame_index])
            & np.isfinite(heading[frame_index])
            & (lengths[frame_index] > 0.0)
            & (widths[frame_index] > 0.0)
        )
        indices = np.flatnonzero(obstacle_valid)
        if len(indices) == 0:
            continue
        frame_values = _directed_frame_ttc(
            np.column_stack((x[frame_index, indices], y[frame_index, indices])),
            np.column_stack((vx[frame_index, indices], vy[frame_index, indices])),
            heading[frame_index, indices],
            lengths[frame_index, indices],
            widths[frame_index, indices],
        )
        focal_local = np.flatnonzero(types[frame_index, indices] == VEHICLE_TYPE)
        focal_slots = indices[focal_local]
        focal_ttc = frame_values[focal_local]
        frame_valid[frame_index, focal_slots] = True
        frame_ttc[frame_index, focal_slots] = focal_ttc
        frame_safe[frame_index, focal_slots] = focal_ttc >= TTC_THRESHOLD_SECONDS
    agent_has_samples = np.any(frame_valid, axis=0)
    agent_min_ttc = np.full(slot_count, TTC_HORIZON_SECONDS, dtype=np.float64)
    for slot in np.flatnonzero(agent_has_samples):
        agent_min_ttc[slot] = np.min(frame_ttc[frame_valid[:, slot], slot])
    agent_safe = agent_has_samples & (agent_min_ttc >= TTC_THRESHOLD_SECONDS)
    return frame_valid, frame_safe, agent_has_samples, agent_safe, agent_min_ttc

def _empty_row():
    row = {name: 0.0 for name in ADDITIONAL_METRIC_NAMES}
    row.update({
        "minimum_ttc_seconds": TTC_HORIZON_SECONDS,
        "closed_loop_metric_agents": 0,
        "ttc_metric_agents": 0,
        "ttc_metric_frames": 0,
        "comfort_metric_agents": 0,
        "comfort_metric_frames": 0,
    })
    return row


def compute_closed_loop_metric_rows(frames, agent_offsets, dt=0.1, focal_mask=None):
    """Compute additive metric dictionaries aligned with Drive per-map logs."""
    offsets = np.asarray(agent_offsets, dtype=np.int64)
    if not frames:
        raise ValueError("At least one global-state frame is required")
    if dt <= 0:
        raise ValueError("dt must be positive")
    stacked = {
        key: np.stack([np.asarray(frame[key]) for frame in frames], axis=0)
        for key in STATE_FIELDS
    }
    expected_agents = int(offsets[-1])
    # HEAD extension: restrict aggregation, without dropping TTC obstacles.
    focal_mask = np.ones(expected_agents, dtype=bool) if focal_mask is None else np.asarray(focal_mask, dtype=bool)
    if focal_mask.shape != (expected_agents,):
        raise ValueError("focal_mask must have one entry per agent")
    if any(values.shape[1] != expected_agents for values in stacked.values()):
        raise ValueError("State-frame size does not match agent_offsets")

    rows = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        start, end = int(start), int(end)
        if end <= start:
            rows.append(_empty_row())
            continue
        data = {key: values[:, start:end] for key, values in stacked.items()}
        valid = data["valid"].astype(bool)
        stopped = data["stopped"].astype(bool)
        comfort = _comfort_results(
            data["vx"], data["vy"], data["heading"], data["id"],
            data["respawn_count"], valid, stopped, data["type"], dt,
        )
        common_valid, all_checks, strict, checks, component_strict, comfort_agents = comfort
        ttc = _ttc_results(
            data["x"], data["y"], data["vx"], data["vy"], data["heading"],
            data["length"], data["width"], valid, data["type"],
        )
        ttc_valid, ttc_safe, ttc_agents, ttc_agent_safe, agent_min_ttc = ttc
        focal = focal_mask[start:end]
        common_valid = common_valid & focal[None, :]
        comfort_agents = comfort_agents & focal
        ttc_valid = ttc_valid & focal[None, :]
        ttc_agents = ttc_agents & focal

        comfort_agent_count = int(np.sum(comfort_agents))
        comfort_frame_count = int(np.sum(common_valid))
        ttc_agent_count = int(np.sum(ttc_agents))
        ttc_frame_count = int(np.sum(ttc_valid))
        row = _empty_row()
        row.update({
            "closed_loop_metric_agents": ttc_agent_count,
            "ttc_metric_agents": ttc_agent_count,
            "ttc_metric_frames": ttc_frame_count,
            "comfort_metric_agents": comfort_agent_count,
            "comfort_metric_frames": comfort_frame_count,
        })
        if ttc_agent_count:
            row["ttc_within_bound_rate"] = float(np.mean(ttc_agent_safe[ttc_agents]))
            row["mean_min_ttc_seconds"] = float(np.mean(agent_min_ttc[ttc_agents]))
            row["minimum_ttc_seconds"] = float(np.min(agent_min_ttc[ttc_agents]))
        if ttc_frame_count:
            row["ttc_safe_frame_rate"] = float(np.mean(ttc_safe[ttc_valid]))
        if comfort_agent_count:
            row["comfortable_rate"] = float(np.mean(strict[comfort_agents]))
            for name in COMFORT_COMPONENTS:
                row[f"comfort_{name}_rate"] = float(
                    np.mean(component_strict[name][comfort_agents])
                )
        if comfort_frame_count:
            row["comfort_frame_rate"] = float(np.mean(all_checks[common_valid]))
            for name in COMFORT_COMPONENTS:
                row[f"comfort_{name}_frame_rate"] = float(
                    np.mean(checks[name][common_valid])
                )
        rows.append(row)
    return rows
