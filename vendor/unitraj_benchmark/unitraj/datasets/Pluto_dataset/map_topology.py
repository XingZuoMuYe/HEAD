"""Utilities to convert unified MetaDrive/ScenarioNet-style map_features into a lane graph.

Why this exists
---------------
Pluto's original NuPlan feature builder relies on the NuPlan map API which provides:
- Accurate lane/lane_connector geometry (centerline + boundaries)
- Explicit topology (incoming/outgoing edges)
- A route as an ordered sequence of roadblocks

In UniTraj we only have `scenario['map_features']`, which is a dict of lane-like features.
This module offers a best-effort conversion into a lightweight lane graph that can be reused
by feature extraction (reference line generation, route id estimation, map filtering, etc.).

Assumptions (as discussed)
-------------------------
Each lane-like map feature is a dict with (some of) the following keys:
- id: lane id (or the dict key itself)
- entry / exit: lists of predecessor/successor lane ids (or roadblock ids)
- polyline: centerline polyline points [(x, y), ...] or [(x, y, heading), ...]
- polygon: boundary polygon points [(x, y), ...]

The code is defensive: it tries multiple key aliases and gracefully degrades when topology
fields are missing by inferring adjacency from geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


LANE_TYPE_HINTS = ("LANE", "LANE_CONNECTOR", "ROAD", "DRIVABLE")


def _as_int_id(x: Any) -> Optional[int]:
    if x is None:
        return None
    # preserve int
    if isinstance(x, (int, np.integer)):
        return int(x)
    # numeric strings
    s = str(x)
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        try:
            return int(s)
        except Exception:
            return None
    # fallback to stable hash (deterministic within run, handles UUIDs like nuScenes map tokens)
    return int(abs(hash(s)) % (10**9))


def _to_xy(points: Any) -> Optional[np.ndarray]:
    """Convert common polyline-like structures to (N,2) float array."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    if arr.shape[1] >= 2:
        return arr[:, :2]
    return None


def _get_first_present(d: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


@dataclass
class LaneNode:
    lane_id: int
    centerline: np.ndarray  # (N,2)
    polygon: Optional[np.ndarray]  # (M,2)
    entry: List[int]
    exit: List[int]
    left_neighbors: List[int]
    right_neighbors: List[int]
    feature_type: str = ""

    @property
    def start(self) -> np.ndarray:
        return self.centerline[0]

    @property
    def end(self) -> np.ndarray:
        return self.centerline[-1]


@dataclass
class LaneGraph:
    lanes: Dict[int, LaneNode]
    entry_to: Dict[int, Set[int]]  # lane_id -> {successors}
    exit_from: Dict[int, Set[int]]  # lane_id -> {predecessors}

    def successors(self, lane_id: int) -> List[int]:
        return sorted(self.entry_to.get(lane_id, set()))

    def predecessors(self, lane_id: int) -> List[int]:
        return sorted(self.exit_from.get(lane_id, set()))


def build_lane_graph(
    map_features: Dict[Any, Dict[str, Any]],
    *,
    infer_from_geometry: bool = True,
    geom_link_dist_m: float = 2.0,
) -> LaneGraph:
    """Build a lightweight lane graph.

    Parameters
    ----------
    map_features:
        scenario['map_features'] dict.
    infer_from_geometry:
        If explicit entry/exit lists are missing, infer adjacency by connecting lane A's end
        to lane B's start when distance < geom_link_dist_m.
    geom_link_dist_m:
        Threshold for geometry-based linking.
    """

    lanes: Dict[int, LaneNode] = {}

    # 1) Parse lane-like features
    for raw_id, feat in (map_features or {}).items():
        if not isinstance(feat, dict):
            continue

        # filter by type hint (but keep permissive)
        t = str(feat.get("type", "")).upper()
        if t and not any(h in t for h in LANE_TYPE_HINTS):
            continue

        # In our unified format, lane_id is the dict key (string form)
        lane_id = _as_int_id(raw_id)
        if lane_id is None:
            # fall back to stable hash (still deterministic within run)
            lane_id = int(abs(hash(str(raw_id))) % (10**9))

        # polyline points are (x, y, heading). We only keep (x, y) here.
        centerline = _to_xy(_get_first_present(feat, ["polyline", "centerline", "baseline_path", "line"]))
        if centerline is None:
            # Some datasets encode lane centerline as polygon boundary; we can't use it reliably.
            continue

        polygon = _to_xy(_get_first_present(feat, ["polygon", "boundary", "poly"]))

        entry = _get_first_present(
            feat,
            [
                "entry_lanes",
                "entry",
                "entries",
                "predecessors",
                "incoming",
                "in",
            ],
        )
        exit_ = _get_first_present(
            feat,
            [
                "exit_lanes",
                "exit",
                "exits",
                "successors",
                "outgoing",
                "out",
            ],
        )

        left_n = _get_first_present(feat, ["left_neighbor", "left_neighbors", "left"])
        right_n = _get_first_present(feat, ["right_neighbor", "right_neighbors", "right"])

        def _ids_list(v: Any) -> List[int]:
            if v is None:
                return []
            if isinstance(v, (int, np.integer, str)):
                v = [v]
            out: List[int] = []
            for x in list(v):
                xid = _as_int_id(x)
                if xid is not None:
                    out.append(xid)
            return out

        lanes[lane_id] = LaneNode(
            lane_id=lane_id,
            centerline=centerline,
            polygon=polygon,
            entry=_ids_list(entry),
            exit=_ids_list(exit_),
            left_neighbors=_ids_list(left_n),
            right_neighbors=_ids_list(right_n),
            feature_type=t,
        )

    # 2) Build adjacency from explicit topology
    succ: Dict[int, Set[int]] = {lid: set() for lid in lanes.keys()}
    pred: Dict[int, Set[int]] = {lid: set() for lid in lanes.keys()}

    for lid, node in lanes.items():
        for s in node.exit:
            if s in lanes:
                succ[lid].add(s)
                pred[s].add(lid)
        for p in node.entry:
            if p in lanes:
                pred[lid].add(p)
                succ[p].add(lid)

    # 3) Optional geometry-based linking if explicit data is weak
    if infer_from_geometry:
        lane_ids = list(lanes.keys())
        if lane_ids:
            ends = np.stack([lanes[lid].end for lid in lane_ids], axis=0)  # (L,2)
            starts = np.stack([lanes[lid].start for lid in lane_ids], axis=0)  # (L,2)

            # Compute pairwise distances end->start
            # For L up to ~1000 this is fine; if larger, consider KDTree.
            d2 = (
                (ends[:, None, 0] - starts[None, :, 0]) ** 2
                + (ends[:, None, 1] - starts[None, :, 1]) ** 2
            )
            thr2 = float(geom_link_dist_m) ** 2
            src_idx, dst_idx = np.where(d2 < thr2)
            for i, j in zip(src_idx.tolist(), dst_idx.tolist()):
                a = lane_ids[i]
                b = lane_ids[j]
                # don't create self loops; keep existing links
                if a != b and b not in succ[a]:
                    succ[a].add(b)
                    pred[b].add(a)

    return LaneGraph(lanes=lanes, entry_to=succ, exit_from=pred)


def estimate_route_lane_ids(
    ego_positions_xy: np.ndarray,
    lane_graph: LaneGraph,
    *,
    max_dist_m: float = 1.5,
    min_hold_frames: int = 3,
    no_backtrack: bool = True,
    include_lateral_neighbors: bool = False,
    lateral_hops: int = 1,
) -> List[int]:
    """Estimate ordered route lane ids by matching ego positions to nearest lane polygon/centerline.

    This mirrors the idea used in `pluto_utils.get_route_roadblock_ids`, but operates on lane_graph.
    The output is a forward ordered sequence (with debouncing) and can be used as a route prior.

    Notes
    -----
    - If polygons are missing, we fall back to distance-to-centerline endpoints (rough).
    - For full fidelity you would want point-to-linestring distance; here we use a cheap proxy.
    """
    if ego_positions_xy is None:
        return []
    pos = np.asarray(ego_positions_xy, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[0] == 0:
        return []

    lane_ids = list(lane_graph.lanes.keys())
    if not lane_ids:
        return []

    # Precompute poly samples for distance approximation
    # Use downsampled centerline points to speed up.
    samples = []
    for lid in lane_ids:
        cl = lane_graph.lanes[lid].centerline
        if cl.shape[0] > 30:
            idx = np.linspace(0, cl.shape[0] - 1, 30).astype(int)
            cl = cl[idx]
        samples.append(cl)

    def match_lane_id(xy: np.ndarray) -> Optional[int]:
        best = None
        best_d = float("inf")
        for lid, cl in zip(lane_ids, samples):
            # min dist to sampled centerline points
            d = np.min(np.linalg.norm(cl - xy[None, :], axis=1))
            if d < best_d:
                best_d = d
                best = lid
        if best_d <= max_dist_m:
            return best
        return None

    route_seq: List[int] = []
    committed_set: Set[int] = set()

    last_committed: Optional[int] = None
    candidate: Optional[int] = None
    candidate_count = 0

    for xy in pos:
        lid = match_lane_id(xy)
        if lid is None:
            candidate = None
            candidate_count = 0
            continue

        if lid == last_committed:
            candidate = None
            candidate_count = 0
            continue

        if candidate is None or lid != candidate:
            candidate = lid
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count < min_hold_frames:
            continue

        def lane_close_to_expert(lid, expert_xy, max_d=2.0, ratio_th=0.6):
            cl = lane_graph.lanes[lid].centerline
            d = np.linalg.norm(cl[:, None, :] - expert_xy[None, :, :], axis=-1)
            min_d = d.min(axis=1)
            return (min_d < max_d).mean() >= ratio_th
        if not lane_close_to_expert(lid, pos, max_d=2.0, ratio_th=0.6):
            continue

        # commit
        if last_committed is None:
            route_seq.append(lid)
            committed_set.add(lid)
            last_committed = lid
        else:
            # no_backtrack: do not revisit an earlier lane id
            if no_backtrack and lid in committed_set:
                # If topology indicates it's a valid successor chain, keep; else skip.
                if lid in lane_graph.successors(last_committed):
                    route_seq.append(lid)
                    last_committed = lid
                # else ignore to avoid loops
            else:
                route_seq.append(lid)
                committed_set.add(lid)
                last_committed = lid

        candidate = None
        candidate_count = 0

    if not include_lateral_neighbors:
        return route_seq

    # Expand each committed lane with lateral neighbors *as provided by map_features*.
    # We do NOT artificially choose “N lanes left/right”. Instead we include however many
    # neighbor ids exist on each LaneNode.
    #
    # Note: if lateral_hops > 1, we only traverse to neighbor-of-neighbor relationships that
    # are explicitly present in the provided neighbor lists (still fully data-driven).
    hops = int(max(1, lateral_hops))

    expanded: List[int] = []
    expanded_set: Set[int] = set()

    def _add(cid: int) -> None:
        if cid in lane_graph.lanes and cid not in expanded_set:
            expanded_set.add(cid)
            expanded.append(cid)

    def _ordered_lateral(nid: int) -> List[int]:
        node = lane_graph.lanes.get(nid)
        if node is None:
            return []
        # Preserve dataset-provided order: left neighbors first, then right neighbors.
        out: List[int] = []
        out.extend([x for x in (node.left_neighbors or [])])
        out.extend([x for x in (node.right_neighbors or [])])
        return out

    for lid in route_seq:
        _add(lid)

        frontier = [lid]
        for _ in range(hops):
            nxt: List[int] = []
            for cur in frontier:
                for nb in _ordered_lateral(cur):
                    if nb in lane_graph.lanes and nb not in expanded_set:
                        nxt.append(nb)
            for nb in nxt:
                _add(nb)
            frontier = nxt
            if not frontier:
                break

    return expanded

