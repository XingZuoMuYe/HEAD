"""Metric helpers matching PufferDrive's native checkpoint evaluator."""

import numpy as np


def candidate_histogram_for_envs(
    selected_indices,
    agent_offsets,
    env_indices,
    candidate_count,
):
    """Count candidate selections only for the requested map environments."""
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    offsets = np.asarray(agent_offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("agent_offsets must start at zero and be one-dimensional")
    if int(offsets[-1]) != len(selected):
        raise ValueError("selected candidate count does not match agent_offsets")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")

    histogram = np.zeros(int(candidate_count), dtype=np.int64)
    for env_index in env_indices:
        env_index = int(env_index)
        if env_index < 0 or env_index + 1 >= len(offsets):
            raise IndexError(f"environment index {env_index} is out of range")
        start = int(offsets[env_index])
        end = int(offsets[env_index + 1])
        histogram += np.bincount(
            selected[start:end],
            minlength=int(candidate_count),
        )[: int(candidate_count)]
    return histogram


NATIVE_METRIC_NAMES = (
    "score",
    "collision_rate",
    "offroad_rate",
    "completion_rate",
    "dnf_rate",
    "lane_alignment_rate",
    "episode_return",
    "episode_length",
    "collisions_per_agent",
    "offroad_per_agent",
    "speed_at_goal",
    "perc_controlled",
    "perc_other",
)

COMFORT_COMPONENTS = (
    "longitudinal_acceleration",
    "lateral_acceleration",
    "yaw_acceleration",
    "yaw_rate",
    "longitudinal_jerk",
    "jerk_magnitude",
)

ADDITIONAL_NATIVE_METRIC_NAMES = (
    "ttc_within_bound_rate",
    "ttc_safe_frame_rate",
    "mean_min_ttc_seconds",
    "comfortable_rate",
    "comfort_frame_rate",
    *(f"comfort_{name}_rate" for name in COMFORT_COMPONENTS),
    *(f"comfort_{name}_frame_rate" for name in COMFORT_COMPONENTS),
)

ADDITIONAL_NATIVE_METRIC_WEIGHT_KEYS = {
    "ttc_within_bound_rate": "ttc_metric_agents",
    "mean_min_ttc_seconds": "ttc_metric_agents",
    "ttc_safe_frame_rate": "ttc_metric_frames",
    "comfortable_rate": "comfort_metric_agents",
    "comfort_frame_rate": "comfort_metric_frames",
}
ADDITIONAL_NATIVE_METRIC_WEIGHT_KEYS.update(
    {f"comfort_{name}_rate": "comfort_metric_agents" for name in COMFORT_COMPONENTS}
)
ADDITIONAL_NATIVE_METRIC_WEIGHT_KEYS.update(
    {f"comfort_{name}_frame_rate": "comfort_metric_frames" for name in COMFORT_COMPONENTS}
)


def _to_python(value):
    return value.item() if isinstance(value, np.generic) else value


def normalize_native_row(map_id, scenario_id, raw):
    """Normalize one native per-map log and recompute completion."""
    row = {"map_id": int(map_id), "scenario_id": str(scenario_id)}
    row.update({key: _to_python(value) for key, value in raw.items()})
    reached = float(row.get("goals_reached_this_episode", 0.0))
    sampled = float(row.get("goals_sampled_this_episode", 0.0))
    row["completion_rate"] = reached / sampled if sampled > 0 else 0.0
    return row


def summarize_native_rows(rows):
    """Aggregate native per-map logs using controlled-agent weights."""
    total_agents = sum(float(row.get("n", 0.0)) for row in rows)
    if total_agents <= 0:
        raise RuntimeError("Evaluation produced no controlled-agent records")

    summary = {
        "unique_maps": len({row["map_id"] for row in rows}),
        "unique_scenarios": len({row["scenario_id"] for row in rows}),
        "controlled_agents": int(total_agents),
    }
    for metric in NATIVE_METRIC_NAMES:
        weighted = sum(
            float(row.get(metric, 0.0)) * float(row.get("n", 0.0))
            for row in rows
        )
        summary[metric] = weighted / total_agents

    for count_key in sorted(set(ADDITIONAL_NATIVE_METRIC_WEIGHT_KEYS.values())):
        summary[count_key] = int(
            sum(float(row.get(count_key, 0.0)) for row in rows)
        )
    for metric in ADDITIONAL_NATIVE_METRIC_NAMES:
        weight_key = ADDITIONAL_NATIVE_METRIC_WEIGHT_KEYS[metric]
        denominator = float(summary[weight_key])
        weighted = sum(
            float(row.get(metric, 0.0)) * float(row.get(weight_key, 0.0))
            for row in rows
        )
        summary[metric] = weighted / denominator if denominator > 0 else 0.0
    ttc_rows = [
        row for row in rows if float(row.get("ttc_metric_agents", 0.0)) > 0
    ]
    summary["minimum_ttc_seconds"] = min(
        (float(row.get("minimum_ttc_seconds", 3.0)) for row in ttc_rows),
        default=3.0,
    )

    reached = sum(
        float(row.get("goals_reached_this_episode", 0.0))
        * float(row.get("n", 0.0))
        for row in rows
    )
    sampled = sum(
        float(row.get("goals_sampled_this_episode", 0.0))
        * float(row.get("n", 0.0))
        for row in rows
    )
    summary["completion_rate"] = reached / sampled if sampled > 0 else 0.0
    summary["goals_reached"] = reached
    summary["goals_sampled"] = sampled
    return summary
