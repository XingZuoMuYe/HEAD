from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from torch.nn.utils.rnn import pad_sequence
from unitraj.datasets.Pluto_dataset.utils import to_device, to_numpy, to_tensor
import numpy as np

def calculate_additional_ego_states(current_state, prev_state, ego_params, dt=0.1):
    cur_velocity = current_state['velocity']
    angle_diff = current_state['heading'] - prev_state['heading']
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    speed = (cur_velocity[0]**2 + cur_velocity[1]**2)**0.5
    if speed < 0.2:
        return 0.0, 0.0  # if the car is almost stopped, the yaw rate is unreliable
    else:
        steering_angle = np.arctan(
            yaw_rate * ego_params['wheel_base'] / speed
        )
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)

        return steering_angle, yaw_rate

def get_route_roadblock_ids(
        ego_state_list,
        map_feature_list,
        max_dist: float = 1.0,
        min_hold_frames: int = 3,
        no_backtrack: bool = True,
):
    from shapely.geometry import Point, Polygon

    map_polygons = {}
    left_neighbors = {}
    right_neighbors = {}
    for map_id, feature in (map_feature_list or {}).items():
        if not _is_lane_like(str(feature.get("type", ""))):
            continue
        poly_points = feature.get("polygon", None)
        if poly_points is None or len(poly_points) < 3:
            continue

        map_polygons[map_id] = Polygon(poly_points)

        def _parse_neighbor_ids(v):
            if v is None:
                return []
            if isinstance(v, dict):
                v = [v]
            if isinstance(v, (str, int, np.integer)):
                v = [v]
            out = []
            for item in list(v):
                if isinstance(item, dict):
                    # common keys in unified format
                    item = item.get('feature_id', item.get('id', item.get('lane_id', None)))
                if item is None:
                    continue
                out.append(str(item))
            return out

        left_neighbors[str(map_id)] = _parse_neighbor_ids(
            feature.get('left_neighbor', feature.get('left_neighbors', feature.get('left', None)))
        )
        right_neighbors[str(map_id)] = _parse_neighbor_ids(
            feature.get('right_neighbor', feature.get('right_neighbors', feature.get('right', None)))
        )

    if not map_polygons:
        return []

    # 2) helper: choose best polygon id for a point
    def match_route_id(x, y):
        pt = Point(float(x), float(y))
        best_id = None
        best_d = float("inf")
        for rid, poly in map_polygons.items():
            # distance is 0 if inside; otherwise boundary distance
            d = poly.distance(pt)
            if d < best_d:
                best_d = d
                best_id = rid
        if best_d <= max_dist:
            return best_id
        return None

    # 3) generate ordered ids with debouncing
    route_seq = []
    committed_set = set()

    last_committed = None
    candidate = None
    candidate_count = 0

    for state in ego_state_list:
        pos = state.get("position", None)
        if pos is None:
            continue
        x, y = (pos[0], pos[1]) if isinstance(pos, (list, tuple, np.ndarray)) else (pos.x, pos.y)

        rid = match_route_id(x, y)
        if rid is None:
            # reset candidate when not on any lane polygon
            candidate = None
            candidate_count = 0
            continue

        if rid == last_committed:
            # stable, keep going
            candidate = None
            candidate_count = 0
            continue

        # if forbidding backtrack (heuristic)
        if no_backtrack and rid in committed_set:
            # ignore returning to an already committed id
            continue

        # debounce switching: rid must persist min_hold_frames
        if rid != candidate:
            candidate = rid
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= max(1, min_hold_frames):
            route_seq.append(candidate)
            committed_set.add(candidate)
            last_committed = candidate
            candidate = None
            candidate_count = 0
    # 4) Expand each committed lane id with its left/right neighbors.
    expanded = []
    expanded_set = set()

    for rid in route_seq:
        rid_s = str(rid)
        candidates = [rid_s]
        candidates += left_neighbors.get(rid_s, [])
        candidates += right_neighbors.get(rid_s, [])

        for cid in candidates:
            if cid is None:
                continue
            if cid in expanded_set:
                continue
            expanded_set.add(cid)
            expanded.append(cid)

    return expanded


def interpolate_polyline(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """
    Interpolate a polyline to a fixed number of points.

    Args:
        polyline: shape (N, 2) or (N, 3)
        num_points: int, number of returned points

    Returns:
        interpolated polyline of shape (num_points, polyline.shape[1])
    """
    if len(polyline) == 0:
        return np.zeros((num_points, 2))
    if len(polyline) == 1:
        return np.tile(polyline, (num_points, 1))

    diffs = polyline[1:] - polyline[:-1]
    dists = np.linalg.norm(diffs[:, :2], axis=-1)

    # Cumulative distance
    cum_dists = np.concatenate([[0.0], np.cumsum(dists)])

    if cum_dists[-1] == 0:
        return np.tile(polyline[0], (num_points, 1))

    # Desired distances
    step = cum_dists[-1] / (num_points - 1)
    targets = np.arange(num_points) * step
    targets[-1] = cum_dists[-1] # ensure exactly ends at the last point

    interpolated = np.zeros((num_points, polyline.shape[1]), dtype=polyline.dtype)
    for dim in range(polyline.shape[1]):
        interpolated[:, dim] = np.interp(targets, cum_dists, polyline[:, dim])

    return interpolated

def _is_lane_like(t: str) -> bool:
    """Return True if SD map feature type represents a drivable lane surface / connector."""
    s = (t or "").upper()
    # Common SD naming patterns:
    #   LANE / LANE_CONNECTOR
    #   LANE_SURFACE_STREET / LANE_SURFACE
    #   LANE_CONNECTOR_* (sometimes)
    if s in {"LANE", "LANE_CONNECTOR"}:
        return True
    if s.startswith("LANE_") or s.startswith("LANE_SURFACE"):
        return True
    if "LANE_CONNECTOR" in s:
        return True
    return False


def _is_crosswalk(t: str) -> bool:
    return "CROSSWALK" in (t or "").upper()


@dataclass
class PlutoFeature:
    data: Dict[str, Any]  # anchor sample
    data_p: Dict[str, Any] = None  # positive sample
    data_n: Dict[str, Any] = None  # negative sample
    data_n_info: Dict[str, Any] = None  # negative sample info

    @classmethod
    def collate(cls, feature_list: List[PlutoFeature]) -> PlutoFeature:
        batch_data = {}

        pad_keys = ["agent", "map"]
        stack_keys = ["current_state", "origin", "angle"]

        if "reference_line" in feature_list[0].data:
            pad_keys.append("reference_line")
        if "static_objects" in feature_list[0].data:
            pad_keys.append("static_objects")
        if "cost_maps" in feature_list[0].data:
            stack_keys.append("cost_maps")

        if feature_list[0].data_n is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list]
                        + [f.data_n[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            batch_data["data_n_valid_mask"] = torch.Tensor(
                [f.data_n_info["valid_mask"] for f in feature_list]
            ).bool()
            batch_data["data_n_type"] = torch.Tensor(
                [f.data_n_info["type"] for f in feature_list]
            ).long()

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list]
                    + [f.data_n[key] for f in feature_list],
                    dim=0,
                )
        elif feature_list[0].data_p is not None:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list]
                        + [f.data_p[key][k] for f in feature_list],
                        batch_first=True,
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list]
                    + [f.data_p[key] for f in feature_list],
                    dim=0,
                )
        else:
            for key in pad_keys:
                batch_data[key] = {
                    k: pad_sequence(
                        [f.data[key][k] for f in feature_list], batch_first=True
                    )
                    for k in feature_list[0].data[key].keys()
                }

            for key in stack_keys:
                batch_data[key] = torch.stack(
                    [f.data[key] for f in feature_list], dim=0
                )

        return PlutoFeature(data=batch_data)

    def to_feature_tensor(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_tensor(v)

        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_tensor(v)
        else:
            new_data_p = None

        if self.data_n is not None:
            new_data_n = {}
            new_data_n_info = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_tensor(v)
            for k, v in self.data_n_info.items():
                new_data_n_info[k] = to_tensor(v)
        else:
            new_data_n = None
            new_data_n_info = None

        return PlutoFeature(
            data=new_data,
            data_p=new_data_p,
            data_n=new_data_n,
            data_n_info=new_data_n_info,
        )

    def to_numpy(self) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_numpy(v)
        if self.data_p is not None:
            new_data_p = {}
            for k, v in self.data_p.items():
                new_data_p[k] = to_numpy(v)
        else:
            new_data_p = None
        if self.data_n is not None:
            new_data_n = {}
            for k, v in self.data_n.items():
                new_data_n[k] = to_numpy(v)
        else:
            new_data_n = None
        return PlutoFeature(data=new_data, data_p=new_data_p, data_n=new_data_n)

    def to_device(self, device: torch.device) -> PlutoFeature:
        new_data = {}
        for k, v in self.data.items():
            new_data[k] = to_device(v, device)
        return PlutoFeature(data=new_data)

    def serialize(self) -> Dict[str, Any]:
        return {"data": self.data}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> PlutoFeature:
        return PlutoFeature(data=data["data"])

    @property
    def is_valid(self) -> bool:
        if "reference_line" in self.data:
            return self.data["reference_line"]["valid_mask"].any()
        else:
            return self.data["map"]["point_position"].shape[0] > 0

    @classmethod
    def normalize(
        cls, data, first_time=False, radius=None, hist_steps=21
    ) -> PlutoFeature:
        cur_state = data["current_state"]
        center_xy, center_angle = cur_state[:2].copy(), cur_state[2].copy()

        rotate_mat = np.array(
            [
                [np.cos(center_angle), -np.sin(center_angle)],
                [np.sin(center_angle), np.cos(center_angle)],
            ],
            dtype=np.float64,
        )

        data["current_state"][:3] = 0
        data["agent"]["position"] = np.matmul(
            data["agent"]["position"] - center_xy, rotate_mat
        )
        data["agent"]["velocity"] = np.matmul(data["agent"]["velocity"], rotate_mat)
        data["agent"]["heading"] -= center_angle

        data["map"]["point_position"] = np.matmul(
            data["map"]["point_position"] - center_xy, rotate_mat
        )
        data["map"]["point_vector"] = np.matmul(data["map"]["point_vector"], rotate_mat)
        data["map"]["point_orientation"] -= center_angle

        data["map"]["polygon_center"][..., :2] = np.matmul(
            data["map"]["polygon_center"][..., :2] - center_xy, rotate_mat
        )
        data["map"]["polygon_center"][..., 2] -= center_angle
        data["map"]["polygon_position"] = np.matmul(
            data["map"]["polygon_position"] - center_xy, rotate_mat
        )
        data["map"]["polygon_orientation"] -= center_angle

        if "causal" in data:
            if len(data["causal"]["free_path_points"]) > 0:
                data["causal"]["free_path_points"][..., :2] = np.matmul(
                    data["causal"]["free_path_points"][..., :2] - center_xy, rotate_mat
                )
                data["causal"]["free_path_points"][..., 2] -= center_angle
        if "static_objects" in data:
            data["static_objects"]["position"] = np.matmul(
                data["static_objects"]["position"] - center_xy, rotate_mat
            )
            data["static_objects"]["heading"] -= center_angle
        if "route" in data:
            data["route"]["position"] = np.matmul(
                data["route"]["position"] - center_xy, rotate_mat
            )
        if "reference_line" in data:
            data["reference_line"]["position"] = np.matmul(
                data["reference_line"]["position"] - center_xy, rotate_mat
            )
            data["reference_line"]["vector"] = np.matmul(
                data["reference_line"]["vector"], rotate_mat
            )
            data["reference_line"]["orientation"] -= center_angle
        else:
            print('!!!!!!!!Reference line is empty!')

        target_position = (
            data["agent"]["position"][:, hist_steps:]
            - data["agent"]["position"][:, hist_steps - 1][:, None]
        )
        target_heading = (
            data["agent"]["heading"][:, hist_steps:]
            - data["agent"]["heading"][:, hist_steps - 1][:, None]
        )
        target = np.concatenate([target_position, target_heading[..., None]], -1)
        target[~data["agent"]["valid_mask"][:, hist_steps:].astype(bool)] = 0
        data["agent"]["target"] = target

        if first_time:
            point_position = data["map"]["point_position"]
            x_max, x_min = radius, -radius
            y_max, y_min = radius, -radius
            valid_mask = (
                (point_position[:, 0, :, 0] < x_max)
                & (point_position[:, 0, :, 0] > x_min)
                & (point_position[:, 0, :, 1] < y_max)
                & (point_position[:, 0, :, 1] > y_min)
            )
            valid_polygon = valid_mask.any(-1)
            data["map"]["valid_mask"] = valid_mask
            if '_raw_map_features' in data["map"]:
                data["map"].pop('_raw_map_features')

            for k, v in data["map"].items():
                data["map"][k] = v[valid_polygon]

            if "causal" in data:
                data["causal"]["ego_care_red_light_mask"] = data["causal"]["ego_care_red_light_mask"][valid_polygon]

        data["origin"] = center_xy
        data["angle"] = center_angle

        return PlutoFeature(data=data)

def to_numpy(v):
    if v is None:
        return v
    if isinstance(v, list) and isinstance(v[0], str):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        return v
    return v.numpy()

def to_feature_tensor_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    new_data = {}
    for k, v in data.items():
        new_data[k] = to_tensor(v)
    return new_data

def collate_pluto_dicts(feature_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_data = {}

    pad_keys = ["agent", "map"]
    stack_keys = ["current_state", "origin", "angle"]

    if "reference_line" in feature_list[0]:
        pad_keys.append("reference_line")
    if "static_objects" in feature_list[0]:
        pad_keys.append("static_objects")
    if "cost_maps" in feature_list[0]:
        stack_keys.append("cost_maps")

    for key in pad_keys:
        if key in feature_list[0]:
            batch_data[key] = {
                k: pad_sequence(
                    [f[key][k] for f in feature_list], batch_first=True
                )
                for k in feature_list[0][key].keys()
            }

    for key in stack_keys:
        if key in feature_list[0]:
            batch_data[key] = torch.stack(
                [f[key] for f in feature_list], dim=0
            )

    return batch_data
