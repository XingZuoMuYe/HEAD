"""Unitraj close-loop trajectory scoring.

This file intentionally stays NuPlan-independent.
The goal is to provide *Pluto-like* feasibility checks ("出路") given the
feature dict produced by UniTraj's Pluto dataset.
"""

from __future__ import annotations

import json


from dataclasses import dataclass, fields
import os
from typing import Dict, Optional, Tuple

import numpy as np


def _to_numpy(x):
    if x is None:
        return None
    return x.detach().cpu().numpy() if hasattr(x, "detach") else (x.cpu().numpy() if hasattr(x, "cpu") else x)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _box_corners(xy: np.ndarray, yaw: np.ndarray, length: float, width: float) -> np.ndarray:
    """Compute oriented bounding box corners.

    Args:
        xy: (..., 2)
        yaw: (...,)
    Returns:
        corners: (..., 4, 2) in order [FR, FL, RL, RR] (consistent but arbitrary)
    """
    hl, hw = 0.5 * float(length), 0.5 * float(width)
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)  # (4,2)

    xy = np.asarray(xy, dtype=np.float32)
    yaw = np.asarray(yaw, dtype=np.float32)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2).astype(np.float32)  # (...,2,2)

    # broadcast: (...,4,2) = (...,4,2) @ (...,2,2)
    local_b = np.broadcast_to(local, xy.shape[:-1] + local.shape)  # (...,4,2)
    corners = local_b @ np.swapaxes(rot, -1, -2)  # (...,4,2)
    return corners + xy[..., None, :]


def _intersect_obb(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating axis test for 2D OBBs.

    Args:
        a: (4,2) corners
        b: (4,2) corners
    """
    # axes are normals of edges (or edges themselves for projection)
    def axes(c):
        e0 = c[1] - c[0]
        e1 = c[3] - c[0]
        e0 = e0 / (np.linalg.norm(e0) + 1e-8)
        e1 = e1 / (np.linalg.norm(e1) + 1e-8)
        return [e0, e1]

    def proj(c, ax):
        p = c @ ax
        return p.min(), p.max()

    for ax in axes(a) + axes(b):
        a0, a1 = proj(a, ax)
        b0, b1 = proj(b, ax)
        if a1 < b0 or b1 < a0:
            return False
    return True


def _point_in_poly(point: np.ndarray, poly: np.ndarray) -> bool:
    """Ray casting point-in-polygon for one point.

    poly: (M,2) closed or open.
    """
    x, y = point
    xp, yp = poly[:, 0], poly[:, 1]
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = xp[i], yp[i]
        xj, yj = xp[j], yp[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


@dataclass
class EvaluatorConfig:
    dt: float = 0.1
    ego_length: float = 4.8
    ego_width: float = 2.0
    rear_axle_to_center: float = 1.67
    agent_length: float = 4.8
    agent_width: float = 2.0
    collision_penalty: float = 100.0
    out_of_drivable_penalty: float = 40.0
    wrong_way_penalty: float = 30.0
    # soft weights
    weight_progress: float = 8.0
    weight_comfort: float = 2.0
    weight_ref_tracking: float = 6.0
    # thresholds
    max_abs_accel: float = 2.0
    max_abs_yaw_diff: float = np.deg2rad(90.0)
    drivable_poly_type_whitelist: Tuple[int, ...] = (0, 1, 2, 3)  # best-effort, dataset-dependent
    num_frames_eval: int = 40  # only evaluate first 40 frames (~4s) for speed, set -1 to evaluate all
    out_of_drivable_grace_ratio: float = 0.1
    progress_displacement_scale: float = 30.0
    progress_path_length_scale: float = 40.0
    stall_distance_threshold: float = 8.0
    stall_penalty: float = 3.0
    ref_tracking_margin: float = 2.0
    ref_tracking_terminal_weight: float = 1.5
    ref_tracking_min_ref_path_length: float = 15.0
    ref_tracking_coverage_buffer: float = 2.0


class TrajectoryEvaluator:
    """Evaluate candidate ego-centric trajectories with stronger feasibility checks."""

    def __init__(self, config: Optional[Dict] = None):
        cfg = dict(config or {})
        valid_keys = {f.name for f in fields(EvaluatorConfig)}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        unknown = sorted(set(cfg.keys()) - valid_keys)
        if unknown and os.environ.get("UNITRAJ_EVAL_WARN_UNKNOWN_CONFIG", "0") not in ("0", "false", "False", ""):
            print(f"[TrajectoryEvaluator] Ignoring unknown config keys: {unknown}")
        self.config = EvaluatorConfig(**filtered)

    def _emit_debug_log(self, payload: Dict) -> None:
        """Keep diagnostics in memory; never contact external debug servers."""
        self.last_debug = payload

    def evaluate(self, candidate_trajectories, input_data, predictions=None, candidate_ref_idx=None, debug_context=None):
        candidate_trajectories = _to_numpy(candidate_trajectories)
        input_data = input_data if isinstance(input_data, dict) else getattr(input_data, "data", input_data)
        predictions = _to_numpy(predictions)

        num_candidates = 0 if candidate_trajectories is None else len(candidate_trajectories)
        if num_candidates == 0:
            return np.zeros(0, dtype=np.float32)

        if candidate_ref_idx is None:
            candidate_ref_idx = np.zeros((num_candidates,), dtype=np.int64)
        else:
            candidate_ref_idx = np.asarray(candidate_ref_idx, dtype=np.int64).reshape(-1)
            if len(candidate_ref_idx) != num_candidates:
                candidate_ref_idx = np.resize(candidate_ref_idx, num_candidates)

        traj_xy = candidate_trajectories[..., :2]
        traj_yaw = candidate_trajectories[..., 2] if candidate_trajectories.shape[-1] > 2 else None

        if traj_yaw is None:
            # estimate yaw from finite differences
            dxy = np.diff(traj_xy, axis=1, prepend=traj_xy[:, :1])
            traj_yaw = np.arctan2(dxy[..., 1], dxy[..., 0] + 1e-8)

        hard_fail = np.zeros((num_candidates,), dtype=bool)
        score = np.zeros((num_candidates,), dtype=np.float32)

        # ---------- Hard constraints: collision / drivable / wrong-way ----------
        # pred_0 = None if predictions is None or len(predictions) == 0 else predictions[1:, ]
        collision = self._has_collision(traj_xy, traj_yaw, input_data, predictions)
        out_of_drivable = self._out_of_drivable(traj_xy, traj_yaw, input_data)
        wrong_way = self._wrong_way(traj_xy, traj_yaw, input_data, candidate_ref_idx)

        hard_fail |= collision
        hard_fail |= wrong_way
        self.last_collision_mask = collision.copy()
        self.last_hard_fail_mask = hard_fail.copy()

        collision_term = -self.config.collision_penalty * collision.astype(np.float32)
        out_of_drivable_term = -self.config.out_of_drivable_penalty * out_of_drivable.astype(np.float32)
        wrong_way_term = -self.config.wrong_way_penalty * wrong_way.astype(np.float32)
        penalty_total = collision_term + out_of_drivable_term + wrong_way_term
        score += penalty_total

        # ---------- Soft metrics ----------
        progress_raw = self._progress(traj_xy)
        comfort_raw = self._comfort(traj_xy)
        ref_tracking_raw = self._ref_tracking(traj_xy, input_data, candidate_ref_idx)
        progress_term = self.config.weight_progress * progress_raw
        comfort_term = self.config.weight_comfort * comfort_raw
        ref_tracking_term = self.config.weight_ref_tracking * ref_tracking_raw
        soft_total = progress_term + comfort_term + ref_tracking_term
        score += soft_total

        debug_context = debug_context or {}
        sorted_idx = np.asarray(debug_context.get("sorted_idx", np.arange(num_candidates)), dtype=np.int64).reshape(-1)
        if len(sorted_idx) != num_candidates:
            sorted_idx = np.resize(sorted_idx, num_candidates)
        route_ids = debug_context.get("route_ids")
        route_ids = list(route_ids) if route_ids is not None else []
        num_debug = min(num_candidates, int(debug_context.get("max_debug_candidates", 12)))
        entries = []
        for i in range(num_debug):
            ref_idx = int(candidate_ref_idx[i]) if len(candidate_ref_idx) > i else 0
            route_id = route_ids[ref_idx] if 0 <= ref_idx < len(route_ids) else None
            entries.append(
                {
                    "rank": i,
                    "sorted_idx": int(sorted_idx[i]),
                    "ref_idx": ref_idx,
                    "route_id": route_id,
                    "collision": bool(collision[i]),
                    "wrong_way": bool(wrong_way[i]),
                    "out_of_drivable_ratio": float(out_of_drivable[i]),
                    "collision_term": float(collision_term[i]),
                    "out_of_drivable_term": float(out_of_drivable_term[i]),
                    "wrong_way_term": float(wrong_way_term[i]),
                    "penalty_total": float(penalty_total[i]),
                    "progress_raw": float(progress_raw[i]),
                    "progress_term": float(progress_term[i]),
                    "comfort_raw": float(comfort_raw[i]),
                    "comfort_term": float(comfort_term[i]),
                    "ref_tracking_raw": float(ref_tracking_raw[i]),
                    "ref_tracking_term": float(ref_tracking_term[i]),
                    "soft_total": float(soft_total[i]),
                    "hard_fail": bool(hard_fail[i]),
                    "rule_score_total": float(score[i]),
                }
            )
        self._emit_debug_log(
            {
                "timestep": debug_context.get("timestep"),
                "num_candidates": int(num_candidates),
                "entries": entries,
            }
        )

        return score

    def _reference_line_arrays(self, input_data: Dict):
        ref = input_data.get("reference_line", None)
        if not isinstance(ref, dict):
            return None, None, None
        ref_pos = _to_numpy(ref.get("position", None))
        ref_valid = _to_numpy(ref.get("valid_mask", None))
        ref_ori = _to_numpy(ref.get("orientation", None))
        if ref_pos is None or ref_valid is None:
            return None, None, None
        if ref_pos.ndim == 4:
            ref_pos = ref_pos[0]
            ref_valid = ref_valid[0]
            if ref_ori is not None and ref_ori.ndim == 3:
                ref_ori = ref_ori[0]
        return ref_pos, ref_valid, ref_ori

    def _select_reference(self, ref_pos, ref_valid, ref_ori, ref_idx: int):
        if ref_pos is None or ref_valid is None:
            return None, None
        idx = None
        if 0 <= int(ref_idx) < len(ref_pos) and ref_valid[int(ref_idx)].sum() > 5:
            idx = int(ref_idx)
        else:
            for r in range(len(ref_pos)):
                if ref_valid[r].sum() > 5:
                    idx = r
                    break
        if idx is None:
            return None, None
        pts = ref_pos[idx][ref_valid[idx] > 0]
        if len(pts) == 0:
            return None, None
        oris = ref_ori[idx][ref_valid[idx] > 0] if ref_ori is not None else None
        return pts, oris

    def _get_circles(self, xy: np.ndarray, yaw: np.ndarray, length, width):
        """Approximate bounding box with 3 overlapping circles."""
        length = np.asarray(length, dtype=np.float32)
        width = np.asarray(width, dtype=np.float32)
        radius = width * 0.6  # slightly larger than half width
        offset = np.maximum(0.0, length / 2.0 - radius)
        offsets = np.stack([-offset, np.zeros_like(offset), offset], axis=-1)
        c, s = np.cos(yaw), np.sin(yaw)
        dir_vec = np.stack([c, s], axis=-1)
        if offset.ndim == 0:
            centers = xy[..., None, :] + dir_vec[..., None, :] * offsets[..., None]
        else:
            # Agent sizes are shaped [A], while positions may be [A, 2]
            # or [A, T, 2]. Keep the agent axis and broadcast the rest.
            offset_shape = (len(offset),) + (1,) * (xy.ndim - 2) + (3, 1)
            centers = (
                xy[..., None, :]
                + dir_vec[..., None, :] * offsets.reshape(offset_shape)
            )
        return centers, radius

    def _progress(self, traj_xy: np.ndarray) -> np.ndarray:
        step_delta = np.diff(traj_xy, axis=1)
        path_length = np.linalg.norm(step_delta, axis=-1).sum(axis=-1)
        displacement = np.linalg.norm(traj_xy[:, -1, :] - traj_xy[:, 0, :], axis=-1)

        progress_from_disp = np.clip(
            displacement / max(self.config.progress_displacement_scale, 1e-3), 0.0, 1.0
        )
        progress_from_path = np.clip(
            path_length / max(self.config.progress_path_length_scale, 1e-3), 0.0, 1.0
        )
        progress = 0.7 * progress_from_disp + 0.3 * progress_from_path

        stall_ratio = np.clip(
            displacement / max(self.config.stall_distance_threshold, 1e-3), 0.0, 1.0
        )
        stall_penalty = -self.config.stall_penalty * (1.0 - stall_ratio)
        stall_penalty[displacement >= self.config.stall_distance_threshold] = 0.0

        return progress + stall_penalty

    def _comfort(self, traj_xy: np.ndarray) -> np.ndarray:
        # velocity in m/s assuming dt
        v = np.linalg.norm(np.diff(traj_xy, axis=1), axis=-1) / max(self.config.dt, 1e-3)
        a = np.diff(v, axis=1) / max(self.config.dt, 1e-3)
        max_abs_a = np.max(np.abs(a), axis=1, initial=0.0)
        # +1 for comfortable, -1 for uncomfortable
        return np.where(max_abs_a > self.config.max_abs_accel, -1.0, 1.0)

    def _ref_tracking(self, traj_xy: np.ndarray, input_data: Dict, candidate_ref_idx: np.ndarray) -> np.ndarray:
        ref_pos, ref_valid, _ = self._reference_line_arrays(input_data)
        if ref_pos is None or ref_valid is None:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        scores = np.zeros((traj_xy.shape[0],), dtype=np.float32)
        for i in range(traj_xy.shape[0]):
            pts, _ = self._select_reference(ref_pos, ref_valid, None, int(candidate_ref_idx[i]))
            if pts is None or len(pts) < 2:
                continue

            ref_step = np.diff(pts, axis=0)
            ref_path_length = float(np.linalg.norm(ref_step, axis=-1).sum())
            if ref_path_length < float(self.config.ref_tracking_min_ref_path_length):
                # Skip ref-tracking penalty when the reference itself does not cover enough horizon.
                continue

            traj_step = np.diff(traj_xy[i], axis=0)
            traj_cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(traj_step, axis=-1))])
            coverage_limit = ref_path_length + float(self.config.ref_tracking_coverage_buffer)
            valid_prefix = traj_cum <= coverage_limit
            if valid_prefix.sum() < 2:
                valid_prefix[: min(2, len(valid_prefix))] = True

            traj_prefix = traj_xy[i, valid_prefix]
            d = np.linalg.norm(traj_prefix[:, None, :] - pts[None, :, :], axis=-1)
            min_dist_per_step = d.min(axis=-1)
            mean_min = float(min_dist_per_step.mean())
            terminal_min = float(min_dist_per_step[-1])
            mean_excess = max(0.0, mean_min - self.config.ref_tracking_margin)
            terminal_excess = max(0.0, terminal_min - self.config.ref_tracking_margin)
            scores[i] = -(mean_excess + self.config.ref_tracking_terminal_weight * terminal_excess)
        return scores

    def _agents_current(self, input_data: Dict):
        agent = input_data.get("agent", {}) if isinstance(input_data.get("agent", {}), dict) else {}
        pos = _to_numpy(agent.get("position", None))
        valid = _to_numpy(agent.get("valid_mask", None))
        heading = _to_numpy(agent.get("heading", None))
        shape = _to_numpy(agent.get("shape", None))
        if pos is None or valid is None:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=bool),
                None,
                None,
            )
        # expected shape: (B, A, T, 2)
        if pos.ndim == 4:
            pos = pos[0]
            valid = valid[0]
            if heading is not None and heading.ndim >= 3:
                heading = heading[0]
            if shape is not None and shape.ndim >= 4:
                shape = shape[0]
        cur_pos = pos[:, -1]
        cur_valid = valid[:, -1].astype(bool)
        cur_heading = heading[:, -1] if heading is not None else None
        cur_shape = shape[:, -1] if shape is not None else None
        # drop ego idx 0 if present
        if len(cur_pos) > 0:
            cur_pos = cur_pos[1:]
            cur_valid = cur_valid[1:]
            if cur_heading is not None:
                cur_heading = cur_heading[1:]
            if cur_shape is not None:
                cur_shape = cur_shape[1:]
        return cur_pos, cur_valid, cur_heading, cur_shape

    def _has_collision(self, traj_xy: np.ndarray, traj_yaw: np.ndarray, input_data: Dict, predictions: Optional[np.ndarray] = None) -> np.ndarray:
        cur_pos, cur_valid, cur_heading, cur_shape = self._agents_current(input_data)
        k = traj_xy.shape[0]
        if len(cur_pos) == 0 or cur_valid.sum() == 0:
            return np.zeros((k,), dtype=bool)

        t = traj_xy.shape[1]
        t_eval = t if self.config.num_frames_eval <= 0 else min(t, self.config.num_frames_eval)

        eval_xy = traj_xy[:, :t_eval]
        eval_yaw = traj_yaw[:, :t_eval]

        # Pluto plans at the rear axle while ScenarioNet agent tracks and
        # MetaDrive collision boxes are center-referenced. Convert only for
        # geometry checks; the rear-axle trajectory remains the controller's
        # tracking reference.
        rear_to_center = float(self.config.rear_axle_to_center)
        eval_xy = eval_xy + rear_to_center * np.stack(
            [np.cos(eval_yaw), np.sin(eval_yaw)], axis=-1
        )

        # Circle collision approximation instead of OBB SAT
        ego_c, ego_r = self._get_circles(eval_xy, eval_yaw, self.config.ego_length, self.config.ego_width)

        collided = np.zeros(k, dtype=bool)
        min_clearance = np.full(k, np.inf, dtype=np.float32)

        # Dynamic collision: If predictions are provided (A_others, T, C), use them over time
        # shape of predictions: (A_others, T, C) where C >= 3 (x, y, yaw)
        if predictions is not None and predictions.shape[0] == len(cur_pos) and predictions.shape[1] > 0:
            # Mask out invalid agents
            pred_valid = predictions[cur_valid]
            # Match the evaluation sequence length
            pred_eval = pred_valid[:, :t_eval]

            # Pad prediction if it's shorter than the ego traj (assume constant position)
            if pred_eval.shape[1] < t_eval:
                pad_len = t_eval - pred_eval.shape[1]
                pad_arr = np.repeat(pred_eval[:, -1:], pad_len, axis=1)
                pred_eval = np.concatenate([pred_eval, pad_arr], axis=1)

            other_xy = pred_eval[..., :2]
            other_yaw = pred_eval[..., 2] if pred_eval.shape[-1] > 2 else np.zeros_like(other_xy[..., 0])

            if cur_shape is not None:
                valid_shape = cur_shape[cur_valid]
                other_width = valid_shape[:, 0]
                other_length = valid_shape[:, 1]
            else:
                other_width = np.full(len(pred_valid), self.config.agent_width)
                other_length = np.full(len(pred_valid), self.config.agent_length)
            other_c, other_r = self._get_circles(
                other_xy, other_yaw, other_length, other_width
            )
            threshold_sq = np.repeat((ego_r + other_r) ** 2, 3)

            for i in range(k):
                # ec: (T, 3, 2) matches ego steps and circles
                ec = ego_c[i]
                # we need to check frame-by-frame: ec is (T_eval, 3, 2)
                # other_c is (A, T_eval, 3, 2)
                # diff: (T_eval, 1, 3, 2) - (T_eval, A*3, 2)  => too big, just loop frames or agents
                for t_idx in range(t_eval):
                    ec_t = ec[t_idx].reshape(-1, 2) # (3, 2)
                    oc_t = other_c[:, t_idx].reshape(-1, 2) # (A*3, 2)
                    diff = ec_t[:, None, :] - oc_t[None, :, :]
                    dist_sq = diff[..., 0]**2 + diff[..., 1]**2
                    clearance = np.sqrt(dist_sq) - np.sqrt(threshold_sq)[None, :]
                    min_clearance[i] = min(
                        min_clearance[i], float(np.min(clearance))
                    )
                    if (clearance < 0.0).any():
                        collided[i] = True
        else:
            # Static fallback: assume current frame frozen
            others = cur_pos[cur_valid]
            others_yaw = cur_heading[cur_valid] if cur_heading is not None else np.zeros((len(others),), dtype=np.float32)

            if cur_shape is not None:
                valid_shape = cur_shape[cur_valid]
                other_width = valid_shape[:, 0]
                other_length = valid_shape[:, 1]
            else:
                other_width = np.full(len(others), self.config.agent_width)
                other_length = np.full(len(others), self.config.agent_length)
            other_c, other_r = self._get_circles(
                others, others_yaw, other_length, other_width
            )
            threshold_sq = np.repeat((ego_r + other_r) ** 2, 3)
            other_c_flat = other_c.reshape(-1, 2)

            for i in range(k):
                ec = ego_c[i].reshape(-1, 2)
                diff = ec[:, None, :] - other_c_flat[None, :, :]
                dist_sq = diff[..., 0]**2 + diff[..., 1]**2
                clearance = np.sqrt(dist_sq) - np.sqrt(threshold_sq)[None, :]
                min_clearance[i] = float(np.min(clearance))
                if (clearance < 0.0).any():
                    collided[i] = True

        self.last_collision_clearance = min_clearance
        return collided

    def _out_of_drivable(self, traj_xy: np.ndarray, traj_yaw: np.ndarray, input_data: Dict) -> np.ndarray:
        """Best-effort drivable-area compliance.

        UniTraj's Pluto map features are *not* full polygon meshes.

        In the Pluto feature format, `map/polygon_position` is a single anchor point per
        map element (M, 2), while the actual sampled geometry lives in
        `map/point_position` with shape (M, 3, P, 2) where the 3 corresponds to
        centerline/left/right boundaries.

        Here we build a coarse drivable polygon per lane-like element by stitching the
        left boundary forward with the right boundary reversed.

        If `map/polygon_type` exists, it can be used as a whitelist.
        """
        mp = input_data.get("map", None)
        if not isinstance(mp, dict):
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        point_pos = _to_numpy(mp.get("point_position", None))
        if point_pos is None:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        poly_type = _to_numpy(mp.get("polygon_type", None))
        # Supported shapes (best-effort, dataset dependent):
        # - point_position: (B, M, 3, P, 2) or (M, 3, P, 2)
        # - polygon_type:   (B, M) or (M,)
        if point_pos.ndim == 5:
            point_pos = point_pos[0]  # (M, 3, P, 2)
        if point_pos.ndim != 4:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)
        if poly_type is not None:
            if poly_type.ndim == 2:
                poly_type = poly_type[0]  # (P,)
            keep = np.isin(poly_type, np.array(self.config.drivable_poly_type_whitelist))
            # Apply keep on map-element axis (M)
            point_pos = point_pos[keep, ...]
        if point_pos.shape[0] == 0:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)

        # Build coarse polygons: left boundary forward + right boundary reversed
        # point_pos: (M, 3, P, 2) with indices: 0=center, 1=left, 2=right
        left = point_pos[:, 1]  # (M, P, 2)
        right = point_pos[:, 2]  # (M, P, 2)
        poly_pos = np.concatenate([left, right[:, ::-1]], axis=1).astype(np.float32)  # (M, 2P, 2)

        # Remove zero padded polygons
        valid_poly_mask = ~np.all(poly_pos == 0, axis=(1, 2))
        poly_pos = poly_pos[valid_poly_mask]
        if poly_pos.shape[0] == 0:
            return np.zeros((traj_xy.shape[0],), dtype=np.float32)

        poly_min = poly_pos.min(axis=1)  # (M, 2)
        poly_max = poly_pos.max(axis=1)  # (M, 2)

        k, t = traj_xy.shape[:2]
        outside_ratio = np.zeros((k,), dtype=np.float32)
        t_eval = t if self.config.num_frames_eval <= 0 else min(t, self.config.num_frames_eval)

        # Pre-extract for vectorized ray casting
        xp = poly_pos[:, :, 0]
        yp = poly_pos[:, :, 1]
        xj = np.roll(xp, 1, axis=1)
        yj = np.roll(yp, 1, axis=1)

        for i in range(k):
            outside_count = 0
            for ti in range(t_eval):
                x, y = traj_xy[i, ti]
                # Fast BBox pruning
                in_bbox_m = np.where((x >= poly_min[:, 0]) & (x <= poly_max[:, 0]) & \
                                     (y >= poly_min[:, 1]) & (y <= poly_max[:, 1]))[0]
                pt_inside = False

                for m in in_bbox_m:
                    # Vectorized ray casting for this polygon
                    cond1 = (yp[m] > y) != (yj[m] > y)
                    cond2 = x < (xj[m] - xp[m]) * (y - yp[m]) / (yj[m] - yp[m] + 1e-12) + xp[m]
                    if (cond1 & cond2).sum() % 2 == 1:
                        pt_inside = True
                        break

                if not pt_inside:
                    outside_count += 1
            ratio = outside_count / max(t_eval, 1)
            ratio = max(0.0, ratio - float(self.config.out_of_drivable_grace_ratio))
            outside_ratio[i] = ratio / max(1.0 - float(self.config.out_of_drivable_grace_ratio), 1e-6)
        return outside_ratio

    def _wrong_way(self, traj_xy: np.ndarray, traj_yaw: np.ndarray, input_data: Dict, candidate_ref_idx: np.ndarray) -> np.ndarray:
        ref_pos, ref_valid, ref_ori = self._reference_line_arrays(input_data)
        if ref_ori is None or ref_valid is None or ref_pos is None:
            return np.zeros((traj_xy.shape[0],), dtype=bool)

        k, t = traj_xy.shape[:2]
        wrong = np.zeros((k,), dtype=bool)
        t_eval = t if self.config.num_frames_eval <= 0 else min(t, self.config.num_frames_eval)

        eval_xy = traj_xy[:, :t_eval]
        eval_yaw = traj_yaw[:, :t_eval]

        for i in range(k):
            pts, oris = self._select_reference(ref_pos, ref_valid, ref_ori, int(candidate_ref_idx[i]))
            if pts is None or oris is None or len(pts) == 0:
                continue
            # Fully vectorized nearest neighbor search and check
            d = np.linalg.norm(pts[None, :, :] - eval_xy[i, :, None, :], axis=-1)
            idx = np.argmin(d, axis=-1)
            dyaw = _wrap_to_pi(eval_yaw[i] - oris[idx])
            if np.any(np.abs(dyaw) > self.config.max_abs_yaw_diff):
                wrong[i] = True

        return wrong
