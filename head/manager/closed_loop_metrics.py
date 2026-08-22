"""Closed-loop evaluation metrics shared by deployment strategies.

UniTraj's evaluator is an optional dependency for HEAD.  This module keeps
that dependency lazy and writes a machine-readable summary without requiring
the UniTraj command-line runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from head.manager.artifact_paths import artifact_path
from head.policy.imitation_policy.unitraj_loader import ensure_unitraj_path


def _install_metadrive_compat_aliases() -> None:
    """Expose the old ``metadrive.metadrive`` import path used by UniTraj."""
    import metadrive
    import metadrive.envs
    import metadrive.envs.base_env
    import metadrive.utils
    import metadrive.utils.math

    aliases = {
        "metadrive.metadrive": metadrive,
        "metadrive.metadrive.envs": metadrive.envs,
        "metadrive.metadrive.envs.base_env": metadrive.envs.base_env,
        "metadrive.metadrive.utils": metadrive.utils,
        "metadrive.metadrive.utils.math": metadrive.utils.math,
    }
    for name, module in aliases.items():
        sys.modules.setdefault(name, module)


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
    """Record UniTraj's per-step/per-episode closed-loop metrics."""

    def __init__(self, cfg, evaluator_cls=None):
        self.cfg = cfg
        self.output_path = _metrics_output_path(cfg)
        self.evaluator = None
        self.episodes = []
        self.error: Optional[str] = None
        self._warned = False
        self._step_flags = {
            "collision": False,
            "out_of_road": False,
            "arrive_dest": False,
        }
        if evaluator_cls is not None:
            self.evaluator = evaluator_cls()
            return

        # UniTraj's detailed evaluator expects recorded-scenario navigation.
        # Generated MetaDrive tasks still use this recorder for generic
        # closed-loop safety statistics, but do not attempt the UniTraj import.
        if str(cfg.args.task) != "real_scenario-v0":
            return

        source = None
        policies = getattr(cfg.args.workflow, "policies", None)
        imitation = getattr(policies, "imitation", None) if policies is not None else None
        if imitation is not None:
            source = imitation.get("source", None)
        try:
            ensure_unitraj_path(source)
            _install_metadrive_compat_aliases()
            from unitraj.utils.evaluate_utils import EvaluateMetrics

            self.evaluator = EvaluateMetrics()
        except Exception as exc:  # optional metrics must not block deployment
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        return self.evaluator is not None

    def start_episode(self) -> None:
        """Reset per-episode state when the evaluator exposes that hook."""
        self._step_flags = {
            "collision": False,
            "out_of_road": False,
            "arrive_dest": False,
        }

    def step(self, info, observation, step_index: int, env) -> None:
        info = info or {}
        collision_keys = (
            "crash_vehicle", "crash_object", "crash_human",
            "crash_building", "crash_sidewalk",
        )
        self._step_flags["collision"] |= any(bool(info.get(key, False)) for key in collision_keys)
        self._step_flags["out_of_road"] |= bool(info.get("out_of_road", False))
        self._step_flags["arrive_dest"] |= bool(info.get("arrive_dest", False))
        if self.evaluator is None:
            return
        try:
            self.evaluator.step(info, observation, int(step_index), env)
        except Exception as exc:
            self._disable(exc)

    def finish_episode(self, *, episode_index: int, reward: float, length: int, env) -> None:
        scores = None
        scene_score = None
        success = None
        if self.evaluator is not None and length > 0:
            try:
                scene_score, scores, success = self.evaluator.reset(int(length), env)
            except Exception as exc:
                self._disable(exc)
        self.episodes.append(
            {
                "episode": int(episode_index),
                "reward": float(reward),
                "length": int(length),
                "scene_score": _json_value(scene_score),
                "scores": _json_value(scores),
                "success": _json_value(success),
                "collision": bool(self._step_flags["collision"]),
                "out_of_road": bool(self._step_flags["out_of_road"]),
                "arrive_dest": bool(self._step_flags["arrive_dest"]),
            }
        )

        item = self.episodes[-1]
        if item["success"] is None:
            item["success"] = not item["collision"] and not item["out_of_road"]
        score_text = ""
        if isinstance(item.get("scores"), dict):
            score_parts = []
            for key in ("no_collision", "ttc", "progress", "comfort"):
                value = item["scores"].get(key)
                if isinstance(value, (int, float)):
                    score_parts.append(f"{key}:{float(value):.3f}")
            if score_parts:
                score_text = " " + " ".join(score_parts)
        print(
            "[闭环] Episode:{episode} Reward:{reward:.3f} Length:{length} "
            "Collision:{collision} OutOfRoad:{out_of_road} ArriveDest:{arrive_dest} "
            "Success:{success}{scores}".format(
                episode=item["episode"],
                reward=item["reward"],
                length=item["length"],
                collision=item["collision"],
                out_of_road=item["out_of_road"],
                arrive_dest=item["arrive_dest"],
                success=item["success"],
                scores=score_text,
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

        if self.evaluator is not None:
            for key, values in getattr(self.evaluator, "round_scores", {}).items():
                numeric = [float(value) for value in values]
                if numeric:
                    summary[key] = mean(numeric)
            total_scores = [float(value) for value in getattr(self.evaluator, "total_scores", [])]
            if total_scores:
                summary["total_score"] = mean(total_scores)

        payload = {
            "schema_version": 1,
            "source": "UniTraj EvaluateMetrics" if self.evaluator is not None else None,
            "available": self.available,
            "error": self.error,
            "task": str(self.cfg.args.task),
            "policy": str(self.cfg.args.workflow.policy),
            "evaluation_mode": "closed_loop",
            "summary": _json_value(summary),
            "episodes": _json_value(self.episodes),
        }
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
        detailed = {
            key: summary[key]
            for key in ("total_score", "no_collision", "ttc", "progress", "comfort")
            if key in summary
        }
        if detailed:
            print("[闭环指标] " + " ".join(
                f"{key}:{float(value):.3f}" for key, value in detailed.items()
            ))
        return self.output_path

    def _disable(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        self.evaluator = None
        if not self._warned:
            print(f"[警告] 闭环指标不可用，继续评测: {self.error}")
            self._warned = True
