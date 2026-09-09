"""Closed-loop evaluation metrics shared by deployment strategies.

HEAD records MetaDrive episode events here and delegates every reported metric
to the ``evaluation`` package through :mod:`head.manager.evaluation_v2`.  The
older UniTraj ``EvaluateMetrics`` recorder has been removed; ``evaluation`` is
now the single source of closed-loop scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from head.manager.artifact_paths import artifact_path


def _json_value(value: Any) -> Any:
    """Convert numpy/scalar values returned by MetaDrive to JSON values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_value(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _metrics_output_path(cfg) -> Path:
    task = str(cfg.args.task).split("-", 1)[0]
    policy = str(cfg.args.workflow.policy)
    map_name = str(getattr(cfg.args.scenario, "map", "unknown"))
    return (
        artifact_path(cfg.args, cfg.args.artifacts.evaluation)
        / "closed_loop"
        / policy
        / task
        / map_name
        / "metrics.json"
    )


class ClosedLoopMetricsRecorder:
    """Record MetaDrive episode events and the ``evaluation`` metrics."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.output_path = _metrics_output_path(cfg)
        self.episodes = []
        # Retained so downstream readers keep a stable metrics.json schema.
        self.error: Optional[str] = None
        evaluation = getattr(cfg.args, "evaluation", {})
        self.v2_enabled = bool(evaluation.get("updated_metrics", True))
        self.v2 = None
        self._step_flags = {
            "collision": False,
            "out_of_road": False,
            "arrive_dest": False,
        }

    @property
    def available(self) -> bool:
        return self.v2_enabled

    def start_episode(self) -> None:
        """Reset per-episode state before a new rollout."""
        if self.v2_enabled:
            from head.manager.evaluation_v2 import EvaluationV2
            self.v2 = EvaluationV2()
        self._step_flags = {
            "collision": False,
            "out_of_road": False,
            "arrive_dest": False,
        }

    def step(self, info, observation, step_index: int, env) -> None:
        info = info or {}
        if self.v2 is not None:
            self.v2.step(info, env)
        collision_keys = (
            "crash_vehicle", "crash_object", "crash_human",
            "crash_building", "crash_sidewalk",
        )
        self._step_flags["collision"] |= any(bool(info.get(key, False)) for key in collision_keys)
        self._step_flags["out_of_road"] |= bool(info.get("out_of_road", False))
        self._step_flags["arrive_dest"] |= bool(info.get("arrive_dest", False))

    def finish_episode(self, *, episode_index: int, reward: float, length: int, env) -> None:
        self.episodes.append(
            {
                "episode": int(episode_index),
                "reward": float(reward),
                "length": int(length),
                "collision": bool(self._step_flags["collision"]),
                "out_of_road": bool(self._step_flags["out_of_road"]),
                "arrive_dest": bool(self._step_flags["arrive_dest"]),
            }
        )

        item = self.episodes[-1]
        if self.v2 is not None:
            item["evaluation_v2"] = self.v2.finish()
            result = item["evaluation_v2"]
            print("[HEAD evaluation]", result.get("scenario_id"),
                  "validity:", result.get("validity"),
                  "strict:", result.get("metrics", {}).get("head_nuplan_style_strict_score"),
                  "frame:", result.get("metrics", {}).get("head_nuplan_style_frame_score"))
        item["success"] = not item["collision"] and not item["out_of_road"]
        print(
            "[闭环] Episode:{episode} Reward:{reward:.3f} Length:{length} "
            "Collision:{collision} OutOfRoad:{out_of_road} ArriveDest:{arrive_dest} "
            "Success:{success}".format(
                episode=item["episode"],
                reward=item["reward"],
                length=item["length"],
                collision=item["collision"],
                out_of_road=item["out_of_road"],
                arrive_dest=item["arrive_dest"],
                success=item["success"],
            )
        )

    def save(self) -> Optional[Path]:
        """Write aggregate and per-episode metrics, returning the path."""
        if not self.episodes and self.error is None:
            return None

        summary = {
            "mean_reward": mean(item["reward"] for item in self.episodes)
            if self.episodes
            else 0.0,
            "mean_length": mean(item["length"] for item in self.episodes)
            if self.episodes
            else 0.0,
        }
        successes = [item["success"] for item in self.episodes if item["success"] is not None]
        if successes:
            summary["success_rate"] = mean(bool(value) for value in successes)
        if self.episodes:
            summary["collision_rate"] = mean(bool(item["collision"]) for item in self.episodes)
            summary["out_of_road_rate"] = mean(bool(item["out_of_road"]) for item in self.episodes)
            summary["arrive_dest_rate"] = mean(bool(item["arrive_dest"]) for item in self.episodes)

        payload = {
            "schema_version": 1,
            "source": None,
            # The removed UniTraj recorder never contributes again; the key stays
            # so existing readers of metrics.json do not need a special case.
            "legacy_source": None,
            "available": self.v2_enabled,
            "error": self.error,
            "task": str(self.cfg.args.task),
            "policy": str(self.cfg.args.workflow.policy),
            "evaluation_mode": "closed_loop",
            "summary": _json_value(summary),
            "episodes": _json_value(self.episodes),
        }
        if self.v2_enabled:
            results = [item.get("evaluation_v2", {}) for item in self.episodes]
            valid_results = [item for item in results if item.get("available")]
            metric_keys = valid_results[0]["metrics"] if valid_results else []
            v2_summary = {
                key: mean(float(item["metrics"][key]) for item in valid_results)
                for key in metric_keys
                if all(isinstance(item["metrics"].get(key), (int, float)) for item in valid_results)
            }
            payload["evaluation_v2"] = dict(
                available=bool(results) and len(valid_results) == len(results),
                valid_episodes=len(valid_results), total_episodes=len(results),
                aggregation="equal_weight_per_scenario; counts are means; per-episode records authoritative",
                summary=v2_summary,
            )
            # Do not discard early collisions/arrivals just because comfort lacks
            # its 15-frame window. Each metric has its own explicit denominator.
            event_results = [item for item in results if item.get("evaluated_frames", 0) > 0]
            per_metric = {}
            keys = set().union(*(item.get("metrics", {}) for item in event_results))
            for key in sorted(keys):
                selected = event_results
                if "score" in key:
                    selected = valid_results
                elif "comfort" in key and "metric_" not in key:
                    selected = [item for item in event_results if item["metrics"].get("comfort_metric_frames", 0) > 0]
                elif "ttc" in key and "metric_" not in key:
                    selected = [item for item in event_results if item["metrics"].get("ttc_metric_frames", 0) > 0]
                values = [float(item["metrics"][key]) for item in selected
                          if isinstance(item.get("metrics", {}).get(key), (int, float))]
                per_metric[key] = dict(mean=mean(values) if values else None, valid_episodes=len(values),
                                       total_episodes=len(results))
            payload["evaluation_v2"]["per_metric"] = per_metric
            payload["evaluation_v2"]["no_controlled_frames_episodes"] = len(results)-len(event_results)
            payload["evaluation_v2"]["summary_scope"] = "complete_metric_episodes_only; use per_metric for partial episodes"
            payload["source"] = "HEAD evaluation/closed_loop_metrics.py + evaluation/compute_puffer_nuplan_style_scores.py"
            payload["available"] = payload["evaluation_v2"]["available"]
            payload["summary_scope"] = "MetaDrive episode events; canonical metrics in evaluation_v2"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "[闭环汇总] Episodes:{count} MeanReward:{reward:.3f} "
            "SuccessRate:{success:.3f} CollisionRate:{collision:.3f} "
            "OutOfRoadRate:{road:.3f} ArriveDestRate:{arrive:.3f}".format(
                count=len(self.episodes),
                reward=float(summary["mean_reward"]),
                success=float(summary.get("success_rate", 0.0)),
                collision=float(summary.get("collision_rate", 0.0)),
                road=float(summary.get("out_of_road_rate", 0.0)),
                arrive=float(summary.get("arrive_dest_rate", 0.0)),
            )
        )
        return self.output_path
