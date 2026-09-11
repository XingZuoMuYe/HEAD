import warnings

import traceback

import os
import pickle
import numpy as np
import h5py

from head.agents.pluto.features.cost_map_manager import CostMapManager
from head.agents.pluto.features.utils import save_dict_to_hdf5
from head.agents.common.base_dataset import BaseDataset
from scenarionet.common_utils import read_scenario

from shapely.geometry import Polygon
from shapely.geometry import Point, LineString
from typing import List, Tuple
from head.agents.pluto.features.pluto_utils import calculate_additional_ego_states, \
    PlutoFeature, interpolate_polyline, _is_lane_like

from head.agents.pluto.features.map_topology import build_lane_graph, estimate_route_lane_ids, _as_int_id


def _get_ego_features(state, ego_category_idx: int = 0, present_idx: int = 20, history_samples: int = 20):
    pos = state['position']
    history_start = max(0, present_idx - history_samples)
    end = present_idx + 1

    pos = pos[history_start:end]
    T = len(pos)
    position = pos[..., :2] if pos.shape[-1] >= 2 else pos
    heading = state['heading'][history_start:end]
    vel = state['velocity'][history_start:end]
    velocity = vel[..., :2] if vel.shape[-1] >= 2 else vel

    if 'acceleration' in state:
        accel = state['acceleration'][history_start:end]
        acceleration = accel[..., :2] if accel.shape[-1] >= 2 else accel
    else:
        acceleration = np.zeros((T, 2), dtype=np.float64)

    width = np.array(state['width'][history_start:end]).reshape(-1, 1)
    length = np.array(state['length'][history_start:end]).reshape(-1, 1)
    # width = np.full((len(heading), 1), 1.8)  # 形状: (N, 1)
    # length = np.full((len(heading), 1), 4.8)  # 形状: (N, 1)
    shape = np.concatenate([width, length], axis=-1)

    valid_mask = state['valid'][history_start:end]
    category = np.array(ego_category_idx, dtype=np.int8)

    return {
        "position": position.astype(np.float64),
        "heading": heading.astype(np.float64),
        "velocity": velocity.astype(np.float64),
        "acceleration": acceleration.astype(np.float64),
        "shape": shape.astype(np.float64),
        "category": category,
        "valid_mask": valid_mask.astype(np.bool_),
    }


def _box_corners_xy(center_xy: np.ndarray, heading: float, width: float, length: float) -> np.ndarray:
    """Return oriented 2D box corners (4,2) in world frame."""
    center_xy = np.asarray(center_xy, dtype=np.float64).reshape(-1)[:2]
    dx = float(length) / 2.0
    dy = float(width) / 2.0
    corners = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], dtype=np.float64)
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return center_xy[None, :] + corners @ rot.T


def _sanitize_poly_xy(poly_xy: np.ndarray) -> np.ndarray:
    poly_xy = np.asarray(poly_xy, dtype=np.float64)
    if poly_xy.ndim != 2 or poly_xy.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    poly_xy = poly_xy[:, :2]
    valid = np.isfinite(poly_xy).all(axis=1)
    poly_xy = poly_xy[valid]
    if len(poly_xy) >= 2 and np.linalg.norm(poly_xy[0] - poly_xy[-1]) < 1e-6:
        poly_xy = poly_xy[:-1]
    if len(poly_xy) < 3:
        return np.zeros((0, 2), dtype=np.float64)

    dedup = [poly_xy[0]]
    for pt in poly_xy[1:]:
        if np.linalg.norm(pt - dedup[-1]) >= 1e-6:
            dedup.append(pt)
    poly_xy = np.asarray(dedup, dtype=np.float64)
    return poly_xy if len(poly_xy) >= 3 else np.zeros((0, 2), dtype=np.float64)


def _sample_closed_ring(poly_xy: np.ndarray, num_points: int) -> np.ndarray:
    poly_xy = _sanitize_poly_xy(poly_xy)
    if len(poly_xy) == 0:
        return np.zeros((0, 2), dtype=np.float64)

    ring = np.concatenate([poly_xy, poly_xy[:1]], axis=0)
    seg = np.linalg.norm(np.diff(ring, axis=0), axis=1)
    perimeter = float(seg.sum())
    if perimeter < 1e-6:
        return np.repeat(poly_xy[:1], num_points, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, perimeter, num_points, endpoint=False)
    sampled = np.zeros((num_points, 2), dtype=np.float64)
    for dim in range(2):
        sampled[:, dim] = np.interp(targets, cum, ring[:, dim])
    return sampled


def _compute_path_tangent(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(points) == 1:
        return np.array([[1.0, 0.0]], dtype=np.float64)

    tang = np.zeros_like(points)
    tang[0] = points[1] - points[0]
    tang[-1] = points[-1] - points[-2]
    if len(points) > 2:
        tang[1:-1] = points[2:] - points[:-2]
    norm = np.linalg.norm(tang, axis=1, keepdims=True)
    tang = tang / np.clip(norm, 1e-6, None)
    return tang


def _fill_polyline_gaps(polyline: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float64).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if len(polyline) == 0 or valid_mask.all():
        return polyline
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return polyline
    if len(valid_idx) == 1:
        polyline[:] = polyline[valid_idx[0]]
        return polyline

    query = np.arange(len(polyline), dtype=np.float64)
    for dim in range(polyline.shape[1]):
        polyline[:, dim] = np.interp(query, valid_idx.astype(np.float64), polyline[valid_idx, dim])
    return polyline


def _smooth_polyline(polyline: np.ndarray) -> np.ndarray:
    polyline = np.asarray(polyline, dtype=np.float64).copy()
    if len(polyline) < 3:
        return polyline
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for dim in range(polyline.shape[1]):
        tmp = np.convolve(polyline[:, dim], kernel, mode='same')
        tmp[0] = polyline[0, dim]
        tmp[-1] = polyline[-1, dim]
        polyline[:, dim] = tmp
    return polyline


def _smooth_series(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    if len(values) < 3:
        return values
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    tmp = np.convolve(values, kernel, mode='same')
    tmp[0] = values[0]
    tmp[-1] = values[-1]
    return tmp


def _validate_reconstructed_lane_bounds(
        centerline: np.ndarray, left_bound: np.ndarray, right_bound: np.ndarray
) -> bool:
    centerline = np.asarray(centerline, dtype=np.float64)
    left_bound = np.asarray(left_bound, dtype=np.float64)
    right_bound = np.asarray(right_bound, dtype=np.float64)
    if len(centerline) < 2 or len(left_bound) != len(centerline) or len(right_bound) != len(centerline):
        return False

    tang = _compute_path_tangent(centerline)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left_signed = np.sum((left_bound - centerline) * normals, axis=1)
    right_signed = np.sum((right_bound - centerline) * normals, axis=1)
    lane_width = left_signed - right_signed

    if np.any(left_signed < 0.2) or np.any(right_signed > -0.2):
        return False
    if np.any(lane_width < 1.5) or np.any(lane_width > 12.0):
        return False

    tang_left = _compute_path_tangent(left_bound)
    tang_right = _compute_path_tangent(right_bound)
    if np.mean(np.sum(tang_left * tang, axis=1)) < 0.5:
        return False
    if np.mean(np.sum(tang_right * tang, axis=1)) < 0.5:
        return False

    left_seg = np.linalg.norm(np.diff(left_bound, axis=0), axis=1)
    right_seg = np.linalg.norm(np.diff(right_bound, axis=0), axis=1)
    if len(left_seg) > 0 and np.mean(left_seg < 1e-3) > 0.1:
        return False
    if len(right_seg) > 0 and np.mean(right_seg < 1e-3) > 0.1:
        return False

    return True


def _extract_lane_bounds_from_polygon(
        centerline: np.ndarray, polygon_xy: np.ndarray, num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate lane widths from polygon support and reconstruct ordered bounds along centerline normals."""
    centerline = np.asarray(centerline, dtype=np.float64)
    polygon_xy = _sanitize_poly_xy(polygon_xy)
    if len(centerline) < 2 or len(polygon_xy) < 3:
        return None, None

    dense_ring = _sample_closed_ring(polygon_xy, max(128, num_points * 12))
    if len(dense_ring) == 0:
        return None, None

    tang = _compute_path_tangent(centerline)
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    seg = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    search_half_window = max(1.5, float(np.median(seg)) * 2.5 if len(seg) > 0 else 2.5)

    left_offset = np.zeros((len(centerline),), dtype=np.float64)
    right_offset = np.zeros((len(centerline),), dtype=np.float64)
    left_valid = np.zeros((len(centerline),), dtype=bool)
    right_valid = np.zeros((len(centerline),), dtype=bool)

    for i, center in enumerate(centerline):
        t = tang[i]
        n = normals[i]
        rel = dense_ring - center[None, :]
        longitudinal = rel @ t
        lateral = rel @ n

        near_mask = np.abs(longitudinal) <= search_half_window
        left_mask = near_mask & (lateral > 0.10)
        right_mask = near_mask & (lateral < -0.10)

        if left_mask.any():
            left_offset[i] = float(np.max(lateral[left_mask]))
            left_valid[i] = True
        if right_mask.any():
            right_offset[i] = float(-np.min(lateral[right_mask]))
            right_valid[i] = True

    min_valid = max(4, num_points // 3)
    if left_valid.sum() < min_valid or right_valid.sum() < min_valid:
        return None, None

    left_offset = _fill_polyline_gaps(left_offset[:, None], left_valid)[:, 0]
    right_offset = _fill_polyline_gaps(right_offset[:, None], right_valid)[:, 0]

    left_offset = _smooth_series(_smooth_series(left_offset))
    right_offset = _smooth_series(_smooth_series(right_offset))

    left_offset = np.clip(left_offset, 0.5, 6.0)
    right_offset = np.clip(right_offset, 0.5, 6.0)

    left_bound = centerline + left_offset[:, None] * normals
    right_bound = centerline - right_offset[:, None] * normals

    if not _validate_reconstructed_lane_bounds(centerline, left_bound, right_bound):
        return None, None

    return left_bound, right_bound


class PlutoDataset(BaseDataset):
    """
    Dataset class for Pluto feature format reading NuPlan db files (ScenarioDescription).
    """

    def __init__(self, config=None, is_validation=False):
        # Match pluto_feature_builder.py indexing semantics (no dependency on nuplan types)
        # agent categories: [EGO, VEHICLE, PEDESTRIAN, BICYCLE]
        self.interested_objects_types = ["EGO", "VEHICLE", "PEDESTRIAN", "BICYCLE"]
        # static obstacle categories (nuplan: [CZONE_SIGN, BARRIER, TRAFFIC_CONE, GENERIC_OBJECT])
        self.static_objects_types = ["CZONE_SIGN", "TRAFFIC_BARRIER", "TRAFFIC_CONE", "GENERIC_OBJECT"]
        # map polygon categories (nuplan: [LANE, LANE_CONNECTOR, CROSSWALK])
        self.polygon_types = ["LANE", "LANE_CONNECTOR", "CROSSWALK"]

        self.max_agents = 48
        self.max_static_obstacles = 10
        self.__collate_fn__ = PlutoFeature.collate

        self.radius = config.get('radius', 100)
        self.history_horizon = config.get('history_horizon', 2)
        self.future_horizon = config.get('future_horizon', 8)
        self.sample_interval = config.get('sample_interval', 0.1)
        self.history_samples = int(self.history_horizon / self.sample_interval)
        self.future_samples = int(self.future_horizon / self.sample_interval) - 1
        self.ego_params = None
        self.strict_route_refline = config.get("strict_route_refline", not is_validation)
        super().__init__(config, is_validation)

    def process_data_chunk(self, worker_index):
        with open(os.path.join('tmp', '{}.pkl'.format(worker_index)), 'rb') as f:
            data_chunk = pickle.load(f)
        file_list = {}
        data_path, mapping, data_list, dataset_name = data_chunk
        hdf5_path = os.path.join(self.cache_path, f'{worker_index}.h5')

        with h5py.File(hdf5_path, 'w') as f:
            for cnt, file_name in enumerate(data_list):
                if worker_index == 0 and cnt % max(int(len(data_list) / 10), 1) == 0:
                    print(f'{cnt}/{len(data_list)} data processed', flush=True)
                scenario = read_scenario(data_path, mapping, file_name)

                try:
                    output = self.preprocess(scenario)

                    pluto_feature = self.process(output)

                    output = self.postprocess(pluto_feature)

                except Exception as e:
                    print('Warning: {} in {}'.format(e, file_name))
                    traceback.print_exc()
                    output = None

                if output is None: continue

                for i, record in enumerate(output):
                    grp_name = dataset_name + '-' + str(worker_index) + '-' + str(cnt) + '-' + str(i)
                    grp = f.create_group(grp_name)
                    save_dict_to_hdf5(grp, record)

                    file_info = {'h5_path': hdf5_path}
                    file_list[grp_name] = file_info
                del scenario
                del output

        return file_list

    def _resolve_controlled_agent_id(self, scenario, controlled_agent_id=None):
        tracks = scenario.get("tracks", {})
        if controlled_agent_id is None:
            controlled_agent_id = scenario.get("metadata", {}).get("sdc_id", "ego")

        controlled_agent_id = str(controlled_agent_id)
        if controlled_agent_id in tracks:
            return controlled_agent_id

        for track_id in tracks.keys():
            if str(track_id) == controlled_agent_id:
                return track_id
        if "ego" in tracks:
            return "ego"
        raise KeyError(
            f"Controlled agent '{controlled_agent_id}' is not present in scenario tracks"
        )

    def _resolve_controlled_category_idx(self, controlled_track_id, controlled_track) -> int:
        if str(controlled_track_id) in {
            "ego", str(getattr(self, "scenario_sdc_id", "ego"))
        }:
            return self.interested_objects_types.index("EGO")

        track_type = str(controlled_track.get("type", "")).upper()
        if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS", "EGO"}:
            return self.interested_objects_types.index("VEHICLE")
        if track_type in {"PEDESTRIAN", "PEDESTRIAN_ADULT", "PEDESTRIAN_CHILD"}:
            return self.interested_objects_types.index("PEDESTRIAN")
        if track_type in {"BICYCLE", "BICYCLIST", "CYCLIST"}:
            return self.interested_objects_types.index("BICYCLE")
        return self.interested_objects_types.index("VEHICLE")

    def _controlled_actor_type(self, controlled_track_id, controlled_track) -> str:
        if str(controlled_track_id) in {
            "ego", str(getattr(self, "scenario_sdc_id", "ego"))
        }:
            return "EGO"
        return str(controlled_track.get("type", "")).upper()

    def _controlled_actor_is_vehicle(self) -> bool:
        actor_type = str(getattr(self, "controlled_actor_type", "EGO")).upper()
        return actor_type in {"EGO", "VEHICLE", "CAR", "TRUCK", "BUS"}

    def preprocess(self, scenario, current_step=20, controlled_agent_id=None):
        """
        In UniTraj, preprocess often generates intermediate dict.
        Here we skip intermediate formatting and construct pluto feature dict explicitly.
        """
        # We can either return the raw PlutoFeature or dict that HDF5 caches.
        # But HDF5 can't cache PlutoFeature directly if it contains custom objects.
        # Let's return primitive dict.
        all_tracks = scenario['tracks']
        self.scenario_sdc_id = str(scenario.get("metadata", {}).get("sdc_id", "ego"))
        controlled_track_id = self._resolve_controlled_agent_id(scenario, controlled_agent_id)
        controlled_track = all_tracks[controlled_track_id]
        self.controlled_actor_type = self._controlled_actor_type(controlled_track_id, controlled_track)
        self.ego_params = dict(
            scenario.get('metadata', {}).get('ego_vehicle_parameters') or {}
        )
        if self._controlled_actor_is_vehicle():
            controlled_length = float(
                np.asarray(controlled_track["state"].get("length", [4.8]))
                .reshape(-1)[0]
            )
            self.ego_params.setdefault(
                "rear_axle_to_center",
                (float(self.ego_params['front_length']) - float(self.ego_params['rear_length'])) / 2
                if {'front_length', 'rear_length'} <= self.ego_params.keys()
                else 0.35 * controlled_length
            )
            self.ego_params.setdefault("wheel_base", 0.6 * controlled_length)
        elif not self._controlled_actor_is_vehicle():
            self.ego_params = {}

        ego_track_state = self._build_local_ego_state_for_pluto(controlled_track['state'])
        ego_state_list = self.parse_tracks_to_states(controlled_track['state'])
        map_features_list = scenario['map_features']
        traffic_light_status = scenario['dynamic_map_states']
        return [
            ego_track_state,
            ego_state_list,
            map_features_list,
            traffic_light_status,
            all_tracks,
            controlled_track_id,
        ]

    def process(self, data, current_step=20, controlled_agent_id=None):
        [
            ego_track_state,
            ego_state_list,
            map_features_list,
            traffic_light_status,
            all_tracks,
            controlled_track_id,
        ] = data

        # Start map tracking
        if not hasattr(self, '_lane_graph_cache'):
            self._lane_graph_cache = {}
        if not hasattr(self, '_route_lane_cache'):
            self._route_lane_cache = {}

        # Use current_step as the present index instead of hardcoded self.history_samples
        present_idx = current_step
        present_idx = min(present_idx, len(ego_state_list) - 1)

        present_ego_state = ego_state_list[present_idx]
        query_xy = present_ego_state['position'][:2]
        # print('query_xy', query_xy)

        # Build lane topology once and reuse in downstream feature extraction.
        # Speed optimization: cache it for the same map instance (object id)
        map_id = id(map_features_list)
        if map_id not in self._lane_graph_cache:
            lane_graph = build_lane_graph(map_features_list, infer_from_geometry=True, geom_link_dist_m=2.0)
            self._lane_graph_cache[map_id] = lane_graph
        else:
            lane_graph = self._lane_graph_cache[map_id]

        controlled_cache_key = (map_id, str(controlled_track_id))
        if controlled_cache_key not in self._route_lane_cache:
            ego_xy = np.asarray([s['position'][:2] for s in ego_state_list], dtype=np.float64)
            route_lane_seq = estimate_route_lane_ids(
                ego_xy,
                lane_graph,
                max_dist_m=1.5,
                min_hold_frames=3,
                no_backtrack=True,
                include_lateral_neighbors=False,
            )
            # print("\n================ ROUTE_SEQ ================")
            # print(route_lane_seq[:50])
            # print("===========================================\n")
            self._route_lane_cache[controlled_cache_key] = route_lane_seq

            # try:
            #     # DEBUG VISUALIZATION: Dump the lane graph mapping to json and image out
            #     import os
            #     os.environ['DUMP_LANE_GRAPH'] = '1'
            #     if os.environ.get('DUMP_LANE_GRAPH', '0') == '1':
            #         from head.agents.pluto.features.plot_lane_graph import dump_lane_graph_to_json, plot_lane_graph
            #         # dump_lane_graph_to_json(lane_graph, out_file=f"lane_graph_{map_id}.json")
            #         # plot_focused_lane_graph(
            #         #     lane_graph=lane_graph,
            #         #     route_seq=route_lane_seq,  # 路由车道ID
            #         #     ego_pos=np.array([0, 0]),  # 自车位置
            #         #     out_file="focused_map.png",
            #         #     focus_radius=200.0,  # 100米范围
            #         #     figsize=(12, 12)
            #         # )
            #         visualize_lane_graph_by_type(
            #             lane_graph=lane_graph,
            #             route_seq=route_lane_seq,
            #             ego_pos=np.array([0, 0]),
            #             out_file="lanes_only.png"
            #         )
            #         # states = visualize_lane_graph_integrity( lane_graph=lane_graph, out_file="lane_graph_check.png" )
            #         # plot_lane_graph(lane_graph, route_seq=route_lane_seq, ego_pos=query_xy, out_file=f"lane_graph_viz_{map_id}.png", map_features_list=map_features_list)
            # except Exception as e:
            #     print(f"Warning: Failed to dump lane graph: {e}")

        else:
            route_lane_seq = self._route_lane_cache[controlled_cache_key]

        data = {}
        prev_idx = max(0, present_idx - 1)
        data["current_state"] = self._get_ego_current_state(
            ego_state_list[present_idx], ego_state_list[prev_idx]
        )

        controlled_features = _get_ego_features(
            ego_track_state,
            ego_category_idx=self._resolve_controlled_category_idx(
                controlled_track_id,
                all_tracks[controlled_track_id],
            ),
            present_idx=present_idx,
            history_samples=self.history_samples,
        )
        # print('controlled features', controlled_features['position'])
        agent_features, agent_tokens, agents_polygon = self._get_agent_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks,
            controlled_agent_id=controlled_track_id,
        )

        data["agent"] = {}
        for k in agent_features.keys():
            data["agent"][k] = np.concatenate(
                [controlled_features[k][np.newaxis, ...], agent_features[k]], axis=0
            )
        agent_tokens = [str(controlled_track_id)] + agent_tokens

        is_validation = getattr(self, 'is_validation', True)

        if is_validation:
            data["agent_tokens"] = agent_tokens
            # data["controlled_agent_id"] = str(controlled_track_id)

        data["static_objects"], static_objects = self._get_static_objects_features(
            query_xy=query_xy,
            present_idx=present_idx,
            all_tracks=all_tracks
        )

        data["map"], map_polygon_tokens = self._get_map_features(
            map_features_list=map_features_list,
            query_xy=query_xy,
            route_roadblock_ids=route_lane_seq,
            traffic_light_status=traffic_light_status,
            radius=self.radius,
            present_idx=present_idx,
            lane_graph=lane_graph,
        )

        if not is_validation:
            data["causal"] = self.scenario_casual_reasoning_preprocess(
                ego_features=controlled_features,
                agent_features=agent_features,
                agents_tokens=agent_tokens,
                map_polygon_tokens=map_polygon_tokens,
                map_features=data["map"],
                present_idx=self.history_samples,
            )
            data["causal"]["interaction_label"] = self._get_interaction_label(
                controlled_features, agent_features
            )
            data["agent"]["valid_mask"][0, self.history_samples + 1:] = data["causal"][
                "fixed_ego_future_valid_mask"
            ]

            cost_map_manager = CostMapManager(
                origin=controlled_features["position"][self.history_samples],
                angle=controlled_features["heading"][self.history_samples],
                height=600,
                width=600,
                resolution=0.2
            )
            cost_maps_res = cost_map_manager.build_cost_maps(
                # CostMapManager expects static objects with shape info: [x,y,heading,width,length,cat]
                static_objects=static_objects,
                agents=agent_features,
                map_features_list=map_features_list,
                agents_polygon=agents_polygon,
                traffic_light_status=traffic_light_status,
                present_idx=present_idx,
                future_steps=self.future_samples,
                # Pluto default: consider VEHICLE/PEDESTRIAN/BICYCLE as obstacles
                dynamic_obstacle_types=(
                    self.interested_objects_types.index("VEHICLE"),
                    self.interested_objects_types.index("PEDESTRIAN"),
                    self.interested_objects_types.index("BICYCLE"),
                ),
                dynamic_dilation_radius_m=1.0,
            )
            data["cost_maps"] = cost_maps_res["cost_maps"]

        data["reference_line"] = self._get_reference_line_feature(
            actor_features=controlled_features,
            map_features_list=map_features_list,
            agent_features=agent_features,
            route_roadblock_ids=route_lane_seq,
            lane_graph=lane_graph,
            training=bool(not self.is_validation)
        )

        return PlutoFeature.normalize(data, first_time=True, radius=self.radius)

    def postprocess(self, pluto_feature):
        return [pluto_feature.data]
        # return pluto_feature

    # def __getitem__(self, idx):
    #     # We fetch the dict from parent class HDF5 cache,
    #     # and then wrap it in PlutoFeature at iteration time!
    #     record = super().__getitem__(idx)
    #     # Reconstruct structured dict
    #     data = {
    #         'agent': {
    #             'position': record['agent/position'],
    #             'heading': record['agent/heading'],
    #             'velocity': record['agent/velocity'],
    #             'shape': record['agent/shape'],
    #             'category': record['agent/category'],
    #             'valid_mask': record['agent/valid_mask'],
    #         },
    #         'map': {
    #             'point_position': record['map/point_position'],
    #             'point_vector': record['map/point_vector'],
    #             'polygon_center': record['map/polygon_center'],
    #             'polygon_type': record['map/polygon_type'],
    #             'valid_mask': record['map/valid_mask']
    #         },
    #         'current_state': record['current_state']
    #     }
    #     return PlutoFeature(data=data)

    def parse_tracks_to_states(self, controlled_track):
        """
        Parameters:
            scenario (dict): Contains 'tracks' key, which is a dictionary where keys are object IDs and values are
                dictionaries containing 'state' information.
                'state' contains 'position', 'heading', 'velocity', 'valid', 'length', 'width', 'height' fields.
                Each field is a 2D array of shape (T, ...) where T is the number of time frames.

        Returns:
            ego_state_list (list): List of ego states, one per frame, each element is a dictionary of the ego state.
            tracked_objects_list (list): List of tracked objects states, one per frame, each element is a list of
                dictionaries of the states of all tracked objects.
                Structure: [ [frame_0_ego_state], [frame_1_ego_state], ... ]
                                              and [ [frame_0_obj1_state, frame_0_obj2_state, ...], [frame_1_obj1_state, frame_1_obj2_state, ...], ... ]
        """
        ego_state = controlled_track
        T = len(ego_state['position'])
        ego_state_list = []

        # 逐帧提取状态
        for frame_idx in range(T):
            heading_val = ego_state['heading'][frame_idx].item()
            position_val = ego_state['position'][frame_idx].tolist()
            if self._controlled_actor_is_vehicle():
                position_val = self._shift_center_to_rear_xy(position_val, heading_val).tolist()
            ego_frame_state = {
                'position': position_val,
                'heading': heading_val,
                'velocity': ego_state['velocity'][frame_idx].tolist(),
                'valid': ego_state['valid'][frame_idx].item(),
                'length': ego_state['length'][frame_idx].item(),
                'width': ego_state['width'][frame_idx].item(),
                'height': ego_state['height'][frame_idx].item()
                # 'length': ego_state.get('length', [4.8] * T)[frame_idx].item() if 'length' in ego_state else 4.8,
                # 'width': ego_state.get('width', [1.8] * T)[frame_idx].item() if 'width' in ego_state else 1.8,
                # 'height': ego_state.get('height', [1.5] * T)[frame_idx].item() if 'height' in ego_state else 1.5
            }
            ego_state_list.append(ego_frame_state)
        return ego_state_list

    def _get_rear_axle_to_center(self) -> float:
        if not self._controlled_actor_is_vehicle():
            return 0.0
        params = self.ego_params or {}
        for key in ("rear_axle_to_center", "rear_axle_to_center_dist", "cog_position_from_rear_axle"):
            if key in params:
                return float(params[key])
        return 1.67

    def _shift_center_to_rear_xy(self, position, heading):
        pos = np.asarray(position, dtype=np.float64).copy()
        d = self._get_rear_axle_to_center()
        pos[:2] -= d * np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        return pos

    def _build_local_ego_state_for_pluto(self, ego_state):
        ego_state = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in ego_state.items()}
        if not self._controlled_actor_is_vehicle():
            return ego_state
        pos = np.asarray(ego_state["position"], dtype=np.float64).copy()
        heading = np.asarray(ego_state["heading"], dtype=np.float64)

        d = self._get_rear_axle_to_center()
        offset = np.stack([np.cos(heading), np.sin(heading)], axis=-1) * d
        pos[..., :2] = pos[..., :2] - offset
        ego_state["position"] = pos
        return ego_state

    def _get_ego_current_state(self, ego_state, prev_state):
        state = np.zeros(7, dtype=np.float64)
        state[0:2] = ego_state['position'][:2]
        state[2] = ego_state['heading']
        heading = float(ego_state['heading'])
        heading_vec = np.array([np.cos(heading), np.sin(heading)], dtype=np.float64)
        cur_velocity = np.asarray(ego_state['velocity'][:2], dtype=np.float64)
        longitudinal_velocity = float(np.dot(cur_velocity, heading_vec))

        if 'acceleration' in ego_state:
            cur_acceleration = np.asarray(ego_state['acceleration'][:2], dtype=np.float64)
            longitudinal_acceleration = float(np.dot(cur_acceleration, heading_vec))
        else:
            prev_heading = float(prev_state['heading'])
            prev_heading_vec = np.array([np.cos(prev_heading), np.sin(prev_heading)], dtype=np.float64)
            prev_velocity = np.asarray(prev_state['velocity'][:2], dtype=np.float64)
            prev_longitudinal_velocity = float(np.dot(prev_velocity, prev_heading_vec))
            longitudinal_acceleration = (
                                                longitudinal_velocity - prev_longitudinal_velocity
                                        ) / max(float(self.sample_interval), 1e-6)

        state[3] = longitudinal_velocity
        state[4] = longitudinal_acceleration

        if self._controlled_actor_is_vehicle():
            steering_angle, yaw_rate = calculate_additional_ego_states(
                ego_state, prev_state, self.ego_params
            )
        else:
            angle_diff = float(ego_state['heading'] - prev_state['heading'])
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            steering_angle = 0.0
            yaw_rate = angle_diff / max(float(self.sample_interval), 1e-6)
        state[5] = steering_angle
        state[6] = yaw_rate
        return state

    def _get_agent_features(
            self,
            query_xy,
            present_idx: int,
            all_tracks,
            controlled_agent_id='ego',
    ):
        # Find valid non-ego agents at present_idx
        present_agents = []
        for obj_id, track in all_tracks.items():
            if str(obj_id) == str(controlled_agent_id) or str(
                    track.get("type", "")).upper() not in self.interested_objects_types:
                continue
            state = track['state']
            if state['valid'][present_idx]:
                pos = state['position'][present_idx][:2]
                dist = np.linalg.norm(np.array(pos) - np.array(query_xy))
                present_agents.append((dist, obj_id, track))

        present_agents.sort(key=lambda x: x[0])
        present_agents = present_agents[:self.max_agents]

        N, T = min(len(present_agents), self.max_agents), self.history_samples + 1

        position = np.zeros((N, T, 2), dtype=np.float64)
        heading = np.zeros((N, T), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        polygon = [None] * N

        agent_tokens = []

        if N == 0:
            return (
                {
                    "position": position,
                    "heading": heading,
                    "velocity": velocity,
                    "shape": shape,
                    "category": category,
                    "valid_mask": valid_mask,
                },
                [],
                [],
            )

        for idx, (_dist, obj_id, track) in enumerate(present_agents):
            agent_tokens.append(obj_id)
            track_state = track['state']

            # Minimal, fast path: slice/pad to fixed horizon.
            history_start = max(0, present_idx - self.history_samples)
            end = present_idx + 1

            pos_seq = np.asarray(track_state['position'], dtype=np.float64)[history_start:end, :2]
            vel_seq = np.asarray(track_state['velocity'], dtype=np.float64)[history_start:end, :2]
            hdg_seq = np.asarray(track_state['heading'], dtype=np.float64)[history_start:end]
            vld_seq = np.asarray(track_state['valid'], dtype=bool)[history_start:end]

            position[idx] = pos_seq
            velocity[idx] = vel_seq
            heading[idx] = hdg_seq
            valid_mask[idx] = vld_seq

            # width/length: keep constant, expand to all timesteps
            w0 = float(np.asarray(track_state.get('width', 0.0)).reshape(-1)[0])
            l0 = float(np.asarray(track_state.get('length', 0.0)).reshape(-1)[0])
            shape[idx, :, 0] = w0
            shape[idx, :, 1] = l0

            track_type = str(track.get('type', '')).upper()
            if track_type in {"VEHICLE", "CAR", "TRUCK", "BUS"}:
                cat = self.interested_objects_types.index("VEHICLE")
            elif track_type in {"PEDESTRIAN", "PEDESTRIAN_ADULT", "PEDESTRIAN_CHILD"}:
                cat = self.interested_objects_types.index("PEDESTRIAN")
            elif track_type in {"BICYCLE", "BICYCLIST", "CYCLIST"}:
                cat = self.interested_objects_types.index("BICYCLE")
            else:
                # fallback: treat unknown dynamic agent as VEHICLE
                cat = self.interested_objects_types.index("VEHICLE")
            category[idx] = cat

            # Build present-time polygon for this agent (used by cost map parked-agent logic).
            center_now = pos_seq[self.history_samples]
            heading_now = float(hdg_seq[self.history_samples])
            poly_xy = _box_corners_xy(center_now, heading_now, float(w0), float(l0))
            polygon[idx] = Polygon(poly_xy)

        agent_features = {
            "position": position,
            "heading": heading,
            "velocity": velocity,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
        }

        return agent_features, agent_tokens, polygon

    def _get_static_objects_features(
            self,
            query_xy,
            present_idx: int,
            all_tracks,
    ):
        static_objects = []
        # dynamic_types = {'VEHICLE', 'PEDESTRIAN', 'BICYCLE'}

        for obj_id, track in all_tracks.items():
            track_type = str(track.get('type', '')).upper()
            if track_type in self.interested_objects_types:
                continue

            state = track['state']
            if not state['valid'][present_idx]:
                continue

            pos = state['position'][present_idx][:2]
            if np.linalg.norm(np.array(pos) - np.array(query_xy)) > self.radius:
                continue

            heading = state['heading'][present_idx]
            length = state['length'][present_idx] if isinstance(state['length'], (list, np.ndarray)) else state[
                'length']
            width = state['width'][present_idx] if isinstance(state['width'], (list, np.ndarray)) else state[
                'width']

            # Map static obstacle type to pluto_feature_builder indices
            if "BARRIER" in track_type or "TRAFFIC_BARRIER" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_BARRIER")
            elif "CONE" in track_type or "TRAFFIC_CONE" in track_type:
                cat = self.static_objects_types.index("TRAFFIC_CONE")
            elif "SIGN" in track_type or "CZONE_SIGN" in track_type:
                cat = self.static_objects_types.index("CZONE_SIGN")
            else:
                cat = self.static_objects_types.index("GENERIC_OBJECT")
            static_objects.append([pos[0], pos[1], heading, float(width), float(length), cat])

        if len(static_objects) > 0:
            static_objects = np.array(static_objects, dtype=np.float64)
            valid_mask = np.ones(len(static_objects), dtype=np.bool_)
        else:
            static_objects = np.zeros((0, 6), dtype=np.float64)
            valid_mask = np.zeros(0, dtype=np.bool_)

        return {
            "position": static_objects[:, :2],
            "heading": static_objects[:, 2],
            "shape": static_objects[:, 3:5],
            "category": static_objects[:, -1].astype(np.int8),
            "valid_mask": valid_mask,
        }, static_objects

    def _get_map_features(
            self,
            map_features_list: dict,
            query_xy,
            route_roadblock_ids: list,
            traffic_light_status,
            radius: float,
            sample_points: int = 20,
            present_idx: int = None,
            lane_graph=None,
    ):
        present_idx_use = present_idx if present_idx is not None else self.history_samples
        route_ids = set(str(route_id) for route_id in route_roadblock_ids)

        # traffic_light_status (SD) is typically a dict keyed by lane_connector_id/lane_id.
        # We convert it to lane_id -> numeric status at present_idx.
        # IMPORTANT: Align with NuPlan TrafficLightStatusType used by Pluto pretrained weights:
        #   GREEN=0, YELLOW=1, RED=2, UNKNOWN=3
        state_mapping = {
            "TRAFFIC_LIGHT_GREEN": 0,
            "TRAFFIC_LIGHT_YELLOW": 1,
            "TRAFFIC_LIGHT_RED": 2,
            "TRAFFIC_LIGHT_UNKNOWN": 3,
        }

        def _tl_to_int(x) -> int:
            if x is None:
                return 3
            s = str(x).upper()
            return int(state_mapping.get(s, 3))

        tls = {}
        for lane_id, tl_info in (traffic_light_status or {}).items():
            state = tl_info.get('state', {}) if isinstance(tl_info.get('state', None), dict) else {}
            obj_state = state.get('object_state', None)
            cur_state = None
            if obj_state is not None and len(obj_state) > present_idx_use:
                cur_state = obj_state[present_idx_use]
            tls[str(lane_id)] = _tl_to_int(cur_state)

        lane_objects = []
        crosswalk_objects = []
        query_pos = query_xy

        for map_id, map_feat in map_features_list.items():
            # Check distance
            polyline = map_feat.get('polyline', None)
            polygon = map_feat.get('polygon', None)

            pts = polyline if polyline is not None and len(polyline) > 0 else polygon
            if pts is None or len(pts) == 0:
                continue

            center = np.mean(pts[:, :2], axis=0)
            if np.linalg.norm(center - query_pos) > radius:
                continue

            obj_type = str(map_feat.get('type', '')).upper()
            if 'CROSSWALK' in obj_type:
                crosswalk_objects.append((map_id, map_feat))
            elif _is_lane_like(obj_type):
                lane_objects.append((map_id, map_feat))

        # Build per-polygon records first to avoid shape/broadcast issues when
        # some polygons are invalid and must be skipped.
        P = sample_points
        rec_point_position = []
        rec_point_vector = []
        rec_point_side = []
        rec_point_orientation = []
        rec_polygon_center = []
        rec_polygon_position = []
        rec_polygon_orientation = []
        rec_polygon_type = []
        rec_polygon_on_route = []
        rec_polygon_tl_status = []
        rec_polygon_speed_limit = []
        rec_polygon_has_speed_limit = []
        rec_polygon_road_block_id = []

        kept_object_ids = []

        for map_id, lane in lane_objects:
            polyline = lane.get('polyline', np.zeros((2, 2)))
            if len(polyline) < 2:
                polyline = np.concatenate([polyline, np.zeros((2 - len(polyline), polyline.shape[-1]))])

            # Sample discrete path from polyline
            centerline = interpolate_polyline(polyline[:, :2], sample_points + 1)

            left_bound = None
            right_bound = None

            polygon = lane.get('polygon', None)
            poly_xy = None
            if polygon is not None:
                try:
                    poly_xy = _sanitize_poly_xy(np.asarray(polygon, dtype=np.float64))
                except Exception:
                    poly_xy = None

                if poly_xy is not None and len(poly_xy) >= 3:
                    left_bound, right_bound = _extract_lane_bounds_from_polygon(
                        centerline, poly_xy, sample_points + 1
                    )

                if left_bound is None or right_bound is None:
                    try:
                        lid = int(map_id) if str(map_id).isdigit() else None
                    except Exception:
                        lid = None

                    if lid is not None and lid in lane_graph.lanes:
                        node = lane_graph.lanes[lid]
                        if node.left_neighbors:
                            nb = node.left_neighbors[0]
                            if nb in lane_graph.lanes:
                                left_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)
                        if node.right_neighbors:
                            nb = node.right_neighbors[0]
                            if nb in lane_graph.lanes:
                                right_bound = interpolate_polyline(lane_graph.lanes[nb].centerline, sample_points + 1)

                if left_bound is None or right_bound is None:
                    if poly_xy is not None:
                        d = np.min(np.linalg.norm(poly_xy[None, :, :] - centerline[:, None, :], axis=-1), axis=1)
                        half_w = float(np.clip(np.median(d), 0.5, 6.0))
                        # Build left/right offset using local tangent normals.
                        tang = np.diff(centerline, axis=0, prepend=centerline[0:1])
                        nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
                        nrm_norm = np.linalg.norm(nrm, axis=1, keepdims=True)
                        nrm = nrm / np.clip(nrm_norm, 1e-6, None)
                        if left_bound is None:
                            left_bound = centerline + half_w * nrm
                        if right_bound is None:
                            right_bound = centerline - half_w * nrm

            # DEBUG: plot lane boundaries to check if they are correct
            dist_to_ego = np.min(np.linalg.norm(centerline - np.asarray(query_pos)[None, :], axis=1))
            if dist_to_ego < 20.0 and getattr(self, "_debug_lane_plot_cnt", 0) < 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(centerline[:, 0], centerline[:, 1], 'k--', label='centerline')
                plt.plot(left_bound[:, 0], left_bound[:, 1], 'b-', label='left_bound')
                plt.plot(right_bound[:, 0], right_bound[:, 1], 'r-', label='right_bound')
                if poly_xy is not None:
                    # append the first point to close the polygon
                    plot_poly = np.vstack([poly_xy, poly_xy[0]])
                    plt.plot(plot_poly[:, 0], plot_poly[:, 1], 'g:', label='original_poly')
                plt.legend()
                plt.title(f"Lane {map_id} Extraction")
                plt.axis('equal')
                plt.show()
                self._debug_lane_plot_cnt = getattr(self, "_debug_lane_plot_cnt", 0) + 1

            try:
                edges = np.stack([centerline, left_bound, right_bound], axis=0)
            except Exception:
                continue

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))

            rec_polygon_center.append(
                np.concatenate(
                    [
                        centerline[int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(centerline[0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            # - MetaDriveType.LANE_SURFACE_STREET for ROADBLOCK interior edges (treat as LANE)
            # - MetaDriveType.LANE_SURFACE_UNSTRUCTURE for ROADBLOCK_CONNECTOR interior edges (treat as LANE_CONNECTOR)
            lane_type = str(lane.get('type', '')).upper()
            if "LANE_SURFACE_UNSTRUCTURE" in lane_type:
                rec_polygon_type.append(int(self.polygon_types.index("LANE_CONNECTOR")))
            else:
                rec_polygon_type.append(int(self.polygon_types.index("LANE")))
            rec_polygon_on_route.append(bool(str(map_id) in route_ids))
            rec_polygon_tl_status.append(int(tls.get(str(map_id), 3)))
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            # if (
            #         bool(str(map_id) in route_ids)
            #         and getattr(self, "_debug_lane_geom_log_cnt", 0) < 8
            # ):
            #     # #region debug-point E:lane-geometry-snapshot
            #     import json, urllib.request;
            #     _p = '.dbg/pluto-turning-drift.env';
            #     _u, _s = 'http://127.0.0.1:7777/event', 'pluto-turning-drift';
            #     exec(
            #         "try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass")
            #     _dbg_n = int(min(10, centerline.shape[0]))
            #     _dbg_lane = {
            #         "map_id": str(map_id),
            #         "query_pos": np.asarray(query_pos, dtype=np.float64).tolist(),
            #         "dist_to_ego": float(dist_to_ego),
            #         "on_route": True,
            #         "centerline_first10": np.asarray(centerline[:_dbg_n], dtype=np.float64).tolist(),
            #         "left_bound_first10": np.asarray(left_bound[:_dbg_n], dtype=np.float64).tolist(),
            #         "right_bound_first10": np.asarray(right_bound[:_dbg_n], dtype=np.float64).tolist(),
            #         "center_orientation_first10": np.asarray(ori[0, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #         "left_orientation_first10": np.asarray(ori[1, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #         "right_orientation_first10": np.asarray(ori[2, :min(10, ori.shape[1])], dtype=np.float64).tolist(),
            #     }
            #     urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps(
            #         {"sessionId": _s, "runId": "pre", "hypothesisId": "E",
            #          "location": "unitraj/datasets/Pluto_dataset/Pluto_dataset.py:_get_map_features",
            #          "msg": "[DEBUG] lane geometry snapshot", "data": _dbg_lane}).encode(),
            #                                                   headers={"Content-Type": "application/json"})).read()
            #     self._debug_lane_geom_log_cnt = getattr(self, "_debug_lane_geom_log_cnt", 0) + 1
            #     # #endregion

            kept_object_ids.append(map_id)

        for map_id, crosswalk in crosswalk_objects:
            polygon = crosswalk.get('polygon', np.zeros((4, 2)))
            if len(polygon) < 3:
                continue

            try:
                from shapely.geometry import Polygon
                import shapely
                poly = Polygon(polygon[:, :2])
                bbox = shapely.minimum_rotated_rectangle(poly)
                coords = np.array(bbox.exterior.coords)
                edge1 = coords[[3, 0]]  # right boundary
                edge2 = coords[[2, 1]]  # left boundary

                edges_geom = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)  # [3, 2, 2]
                v = edges_geom[:, 1] - edges_geom[:, 0]  # [3, 2]
                steps = np.linspace(0, 1, sample_points + 1, endpoint=True)[None, :]
                edges = edges_geom[:, 0][:, None, :] + v[:, None, :] * steps[:, :, None]  # [3, P+1, 2]

                # DEBUG: plot crosswalk edges
                dist_to_ego = np.min(np.linalg.norm(edges[0] - np.asarray(query_pos)[None, :], axis=1))
                if dist_to_ego < 20.0 and getattr(self, "_debug_cw_plot_cnt", 0) < 0:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(edges[0, :, 0], edges[0, :, 1], 'k--', label='center')
                    plt.plot(edges[1, :, 0], edges[1, :, 1], 'b-', label='left_edge (edge2)')
                    plt.plot(edges[2, :, 0], edges[2, :, 1], 'r-', label='right_edge (edge1)')
                    plot_poly = np.vstack([polygon[:, :2], polygon[0, :2]])
                    plt.plot(plot_poly[:, 0], plot_poly[:, 1], 'g:', label='original_poly')
                    plt.legend()
                    plt.title(f"Crosswalk {map_id} Extraction")
                    plt.axis('equal')
                    plt.show()
                    self._debug_cw_plot_cnt = getattr(self, "_debug_cw_plot_cnt", 0) + 1

            except Exception as e:
                edges = np.tile(interpolate_polyline(polygon[:, :2], sample_points + 1)[None, :], (3, 1, 1))

            vec = edges[:, 1:] - edges[:, :-1]
            pos = edges[:, :-1]
            ori = np.arctan2(vec[:, :, 1], vec[:, :, 0])

            rec_point_vector.append(vec)
            rec_point_position.append(pos)
            rec_point_orientation.append(ori)
            rec_point_side.append(np.arange(3, dtype=np.int8))
            rec_polygon_center.append(
                np.concatenate(
                    [
                        edges[0, int(sample_points / 2)],
                        [ori[0, int(sample_points / 2)]],
                    ],
                    axis=-1,
                )
            )
            rec_polygon_position.append(edges[0, 0])
            rec_polygon_orientation.append(float(ori[0, 0]))
            rec_polygon_type.append(int(self.polygon_types.index("CROSSWALK")))
            rec_polygon_on_route.append(False)
            rec_polygon_tl_status.append(3)
            rec_polygon_has_speed_limit.append(False)
            rec_polygon_speed_limit.append(0.0)
            rec_polygon_road_block_id.append(int(map_id))

            kept_object_ids.append(map_id)

        # Stack records
        M = len(rec_point_position)
        point_position = np.stack(rec_point_position, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_vector = np.stack(rec_point_vector, axis=0) if M > 0 else np.zeros((0, 3, P, 2), dtype=np.float64)
        point_orientation = np.stack(rec_point_orientation, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        point_side = np.stack(rec_point_side, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.int8)
        polygon_center = np.stack(rec_polygon_center, axis=0) if M > 0 else np.zeros((0, 3), dtype=np.float64)
        polygon_position = np.stack(rec_polygon_position, axis=0) if M > 0 else np.zeros((0, 2), dtype=np.float64)
        polygon_orientation = np.asarray(rec_polygon_orientation, dtype=np.float64) if M > 0 else np.zeros((0,),
                                                                                                           dtype=np.float64)
        polygon_type = np.asarray(rec_polygon_type, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_on_route = np.asarray(rec_polygon_on_route, dtype=np.bool_) if M > 0 else np.zeros((0,), dtype=np.bool_)
        polygon_tl_status = np.asarray(rec_polygon_tl_status, dtype=np.int8) if M > 0 else np.zeros((0,), dtype=np.int8)
        polygon_speed_limit = np.asarray(rec_polygon_speed_limit, dtype=np.float64) if M > 0 else np.zeros((0,),
                                                                                                           dtype=np.float64)
        polygon_has_speed_limit = np.asarray(rec_polygon_has_speed_limit, dtype=np.bool_) if M > 0 else np.zeros((0,),
                                                                                                                 dtype=np.bool_)
        polygon_road_block_id = np.asarray(rec_polygon_road_block_id, dtype=np.int32) if M > 0 else np.zeros((0,),
                                                                                                             dtype=np.int32)

        object_ids = kept_object_ids

        map_features = {
            "point_position": point_position,
            "point_vector": point_vector,
            "point_orientation": point_orientation,
            "point_side": point_side,
            "polygon_center": polygon_center,
            "polygon_position": polygon_position,
            "polygon_orientation": polygon_orientation,
            "polygon_type": polygon_type,
            "polygon_on_route": polygon_on_route,
            "polygon_tl_status": polygon_tl_status,
            "polygon_has_speed_limit": polygon_has_speed_limit,
            "polygon_speed_limit": polygon_speed_limit,
            "polygon_road_block_id": polygon_road_block_id,
            # allow causal reasoning to access raw polygons by map_id
            "_raw_map_features": map_features_list,
        }

        return map_features, object_ids

    def scenario_casual_reasoning_preprocess(
            self,
            ego_features,
            agent_features,
            agents_tokens,
            map_polygon_tokens,
            map_features=None,
            present_idx=None,
    ):
        """Heuristic causal reasoning without nuPlan ScenarioManager.

        找到自车前方最近的动态车辆，识别对自车有影响的红灯区域，计算自车在考虑前方障碍物和红灯后的可行驶路径，判断自车未来轨迹是否会进入红灯区域

        Limitations vs nuplan:
        - no drivable-area query
        - no lane-graph-based leading-object inference
        - no precise stop-line occupancy; we use map polygon containment as proxy
        """
        if present_idx is None:
            present_idx = self.history_samples

        num_agents = len(agents_tokens)
        num_maps = len(map_polygon_tokens)
        T_future = self.future_samples

        leading_agent_mask = np.zeros(num_agents, dtype=bool)
        leading_distance = np.zeros(num_agents, dtype=np.float64)
        ego_care_red_light_mask = np.zeros(num_maps, dtype=bool)
        fixed_ego_future_valid_mask = np.ones(T_future, dtype=bool)

        # present ego pose
        ego_pos = ego_features["position"][present_idx]
        ego_heading = float(ego_features["heading"][present_idx])
        ego_dir = np.array([np.cos(ego_heading), np.sin(ego_heading)], dtype=np.float64)

        # --- leading dynamic agents (forward cone in ego heading) ---
        nearest_leading_agent_idx = None
        nearest_leading_agent_dist = None

        if agent_features is not None and agent_features["position"].shape[0] > 0:
            # agent_features are non-ego agents only, aligned with agents_tokens[1:]
            pos_now = agent_features["position"][:, present_idx]
            valid_now = agent_features["valid_mask"][:, present_idx]
            rel = pos_now - ego_pos[None, :]
            forward = rel @ ego_dir  # projection
            lateral = np.abs(rel[:, 0] * (-ego_dir[1]) + rel[:, 1] * ego_dir[0])

            # candidates in front within lateral band
            cand = valid_now & (forward > 0.0) & (lateral < 3.5)
            if cand.any():
                # sort by forward distance
                order = np.argsort(forward + (~cand) * 1e6)
                for j in order:
                    if not cand[j]:
                        continue
                    # mark this as leading
                    idx_token = j + 1  # shift because ego at 0 in agents_tokens
                    leading_agent_mask[idx_token] = True
                    leading_distance[idx_token] = float(forward[j])
                    if nearest_leading_agent_idx is None:
                        nearest_leading_agent_idx = idx_token
                        nearest_leading_agent_dist = float(forward[j])
                    # also mark other close-in-front agents as leading
                    if forward[j] < 30.0:
                        continue
                    break

        # --- red light polygons ---
        nearest_red_poly = None
        nearest_red_poly_dist = None

        def _is_red(x):
            # numeric encoding: RED==2,UNKNOWN==3
            if isinstance(x, (np.integer, int)):
                return int(x) == 2
            if isinstance(x, (bytes, str)):
                return "RED" in str(x).upper()
            return False

        if map_features is not None and "polygon_tl_status" in map_features:
            tl_status = map_features["polygon_tl_status"]
            poly_pos = map_features.get("polygon_position", None)
            for i in range(min(num_maps, len(tl_status))):
                if _is_red(tl_status[i]):
                    ego_care_red_light_mask[i] = True
                    if poly_pos is not None:
                        d = float(np.linalg.norm(poly_pos[i] - ego_pos))
                        if nearest_red_poly_dist is None or d < nearest_red_poly_dist:
                            nearest_red_poly_dist = d
                            nearest_red_poly = i

        # nuplan builder: "waiting for red light without lead" means nearest leading object is red light
        is_waiting_for_red_light_without_lead = bool(nearest_red_poly is not None and nearest_leading_agent_idx is None)

        # future valid mask: if ego future enters nearest red polygon, invalidate remaining future
        if nearest_red_poly is not None and map_features is not None:
            token = map_polygon_tokens[nearest_red_poly] if nearest_red_poly < len(map_polygon_tokens) else None
            poly = None
            raw_map = map_features.get("_raw_map_features", None)
            if isinstance(raw_map, dict) and token is not None:
                raw = raw_map.get(str(token), raw_map.get(token))
                if isinstance(raw, dict):
                    poly = raw.get("polygon", None)

            if poly is not None and len(poly) >= 3:
                shp = Polygon(poly[:, :2])
                future_pos = ego_features["position"][present_idx + 1: present_idx + 1 + T_future][:, :2]
                for i in range(min(T_future, len(future_pos))):
                    pt = future_pos[i]
                    if shp.contains(Polygon([pt, pt, pt]).centroid):
                        fixed_ego_future_valid_mask[i:] = False
                        break

        # free path points along ego heading, with end limited by nearest lead and red light
        ego_speed = float(np.linalg.norm(ego_features["velocity"][present_idx])) if "velocity" in ego_features else 0.0
        free_path_start = ego_speed ** 2 / (2 * 5.0) + 2.5
        free_path_end = max(7.0, ego_speed ** 2 / (2 * 1.5))
        if nearest_leading_agent_dist is not None:
            free_path_end = min(free_path_end, nearest_leading_agent_dist)
        if nearest_red_poly_dist is not None:
            free_path_end = min(free_path_end, nearest_red_poly_dist)

        if free_path_end <= free_path_start:
            free_path_points = np.zeros((0, 3), dtype=np.float64)
            free_path_points_angle = np.zeros((0,), dtype=np.float64)
        else:
            n = max(int((free_path_end - free_path_start) / 1.0), 2)
            ds = np.linspace(free_path_start + 3.0, max(free_path_start + 3.0, free_path_end - 3.0), n)
            points = (ego_pos[None, :] + ds[:, None] * ego_dir[None, :]).astype(np.float64)
            headings = np.full((points.shape[0], 1), ego_heading, dtype=np.float64)
            free_path_points = np.hstack([points, headings])

        return {
            "is_waiting_for_red_light_without_lead": is_waiting_for_red_light_without_lead,
            "leading_agent_mask": leading_agent_mask,
            "leading_distance": leading_distance,
            "ego_care_red_light_mask": ego_care_red_light_mask,
            "fixed_ego_future_valid_mask": fixed_ego_future_valid_mask,
            "free_path_points": free_path_points,
        }

    def _get_interaction_label(self, ego, agents):
        """Compute interaction label between ego and each agent.

        This is a dependency-free approximation of PlutoFeatureBuilder's interaction label.
        We:
        1) Find nearest ego-agent distance over future horizon (t > history_samples)
        2) If below threshold and boxes intersect, mark as interaction
        3) Label is time difference (ego_t - agent_t) in the closest pair, clipped, with 0 meaning no interaction.

        Output shape matches builder usage: (N_agents + 1,) including ego at index 0.
        """
        start = self.history_samples + 1
        ego_heading = ego["heading"][start:]
        ego_position = ego["position"][start:]

        agents_position = agents["position"][:, start:]
        agents_heading = agents["heading"][:, start:]
        agents_shape = agents["shape"][:, start:]
        agents_valid = agents["valid_mask"][:, start:]

        if agents_position.shape[0] == 0 or agents_position.shape[1] == 0 or ego_position.shape[0] == 0:
            return np.zeros(1, dtype=np.int64)

        N, T = agents_position.shape[:2]
        Te = ego_position.shape[0]

        # pairwise distances for each agent across time pairs (agent_t, ego_t)
        # Build (N, T, Te) distances in numpy for speed/memory constraints: do incremental min
        min_dist = np.full((N,), 1e9, dtype=np.float64)
        min_idx = np.full((N,), -1, dtype=np.int64)

        for i in range(N):
            if not agents_valid[i].any():
                continue
            # restrict to valid timesteps
            valid_ts = np.where(agents_valid[i])[0]
            if len(valid_ts) == 0:
                continue
            # compute cdist for valid agent steps to all ego steps
            a = agents_position[i, valid_ts]  # (Tv,2)
            e = ego_position  # (Te,2)
            d = np.linalg.norm(a[:, None, :] - e[None, :, :], axis=-1)  # (Tv,Te)
            flat = d.reshape(-1)
            j = int(flat.argmin())
            md = float(flat[j])
            if md < min_dist[i]:
                min_dist[i] = md
                # encode back to agent_t, ego_t in full-T coordinates
                agent_t = int(valid_ts[j // Te])
                ego_t = int(j % Te)
                min_idx[i] = agent_t * Te + ego_t

        interact_flag = min_dist < 4.0

        # collision check with oriented boxes
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            agent_shape = agents_shape[i, agent_t]
            agent_box = self._build_agent_bbox(
                agents_position[i, agent_t],
                agents_heading[i, agent_t],
                float(agent_shape[0]),
                float(agent_shape[1]),
            )
            ego_box = self._build_ego_bbox(ego_position[ego_t], float(ego_heading[ego_t]))
            if not agent_box.intersects(ego_box):
                interact_flag[i] = False

        # label
        interact_label = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            if not interact_flag[i] or min_idx[i] < 0:
                continue
            agent_t = int(min_idx[i] // Te)
            ego_t = int(min_idx[i] % Te)
            interact_label[i] = ego_t - agent_t

        # prepend ego
        return np.concatenate([np.zeros(1, dtype=np.int64), interact_label])

    @staticmethod
    def _get_interact_type(index, T=80):
        row, col = index // T, index % T
        if row == col:
            return 0  # collision or self
        return col - row

    def _build_agent_bbox(self, xy, angle, width, length):
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
        ])
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        center = xy[:2] if (isinstance(xy, np.ndarray) and xy.shape[-1] >= 2) else xy
        return Polygon(center + corners @ rot.T)

    def _build_ego_bbox(self, xy, angle):
        center = xy + 1.67 * np.array([np.cos(angle), np.sin(angle)])
        width = getattr(self, 'width', 2.0)
        length = getattr(self, 'length', 5.0)
        return self._build_agent_bbox(center, angle, width, length)

    def _get_ego_head_position(self, xy, angle):
        ego_len = getattr(self, 'length', 5.0)
        return xy + ego_len * np.array([np.cos(angle), np.sin(angle)]) / 2

    def _get_reference_line_feature(
            self,
            actor_features=None,
            map_features_list=None,
            agent_features=None,
            route_roadblock_ids=None,
            lane_graph=None,
            training=False,
    ):
        actor_pos = actor_features["position"][-1]
        actor_heading = actor_features["heading"][-1]
        actor_speed = float(np.linalg.norm(actor_features["velocity"][-1]))

        radius = self.radius

        # 路由ID转换
        route_seq = []
        for x in (route_roadblock_ids or []):
            mapped_x = _as_int_id(x)
            if mapped_x is not None:
                route_seq.append(mapped_x)

        def wrap_to_pi(angle):
            return (angle + np.pi) % (2 * np.pi) - np.pi

        # Build a working lane set within radius to avoid excessive computation.
        # If route is available, restrict to route lanes (plus a small neighbor set) to reduce branching.
        lane_ids = []
        lane_polys = {}
        lane_xyz = {}
        from typing import Set
        neighbor_extra: Set[int] = set()

        if route_seq:
            # 拓展左边车道
            for lid in route_seq:
                node = lane_graph.lanes.get(lid)
                if node is None:
                    continue
                neighbor_extra.update(node.left_neighbors)
                neighbor_extra.update(node.right_neighbors)

        allowed_lanes = None
        if route_seq:
            # Keep a deterministic iteration order to avoid subtle randomness in candidate selection.
            if self.strict_route_refline:
                allowed_lanes = set(route_seq)
            else:
                allowed_lanes = set(route_seq) | neighbor_extra # 不严格限制贴ref_line，路由车道 + 邻居车道
            allowed_lanes_ordered = []
            seen = set()
            for lid in route_seq:
                if lid in allowed_lanes and lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
            for lid in sorted(allowed_lanes):
                if lid not in seen:
                    allowed_lanes_ordered.append(lid)
                    seen.add(lid)
        else:
            allowed_lanes_ordered = None

        iter_lids = allowed_lanes_ordered if allowed_lanes_ordered is not None else list(lane_graph.lanes.keys())
        for lid in iter_lids:
            node = lane_graph.lanes.get(lid)
            if node is None:
                continue
            cl = node.centerline
            if cl is None or len(cl) < 2:
                continue
            # 距离过滤
            min_dist = float(np.min(np.linalg.norm(cl - actor_pos[None, :2], axis=1)))
            if min_dist > radius:
                continue
            lane_ids.append(lid)
            lane_polys[lid] = cl
            lane_xyz[lid] = node

        # 候选起始车道选择
        candidates: List[Tuple[int, float]] = []
        ego_dir = np.array([np.cos(actor_heading), np.sin(actor_heading)], dtype=np.float64)

        allow_neighbor_refs = bool(training)
        if route_seq and agent_features is not None and agent_features.get("position", None) is not None:
            try:
                pos_now = np.asarray(agent_features["position"][:, -1], dtype=np.float64)
                valid_now = np.asarray(agent_features["valid_mask"][:, -1], dtype=np.bool_)
                vel_now = (
                    np.asarray(agent_features["velocity"][:, -1], dtype=np.float64)
                    if "velocity" in agent_features and agent_features["velocity"].shape[0] > 0
                    else np.zeros((len(pos_now), 2), dtype=np.float64)
                )
                if len(pos_now) > 0:
                    rel = pos_now - actor_pos[None, :2]
                    forward = rel @ ego_dir
                    lateral = np.abs(rel[:, 0] * (-ego_dir[1]) + rel[:, 1] * ego_dir[0])
                    speed_now = np.linalg.norm(vel_now[..., :2], axis=-1)
                    blocking = valid_now & (forward > 0.0) & (forward < 12.0) & (lateral < 3.5)
                    if blocking.any():
                        nearest = int(np.argmin(np.where(blocking, forward, 1e6)))
                        speed_threshold = max(2.0, actor_speed * 0.35)
                        allow_neighbor_refs = bool(speed_now[nearest] < speed_threshold)
            except Exception:
                allow_neighbor_refs = bool(training)

        # 车道必须同时满足距离近、朝向匹配、在前方三个条件才能成为候选
        DIST_TH = 15.0
        HEADING_TH = 1.05
        FORWARD_TH = -2.0

        for lid in lane_ids:
            poly = lane_polys[lid]
            dists = np.linalg.norm(poly - actor_pos[None, :2], axis=1)
            k = int(np.argmin(dists))
            dist = float(dists[k])
            if dist > DIST_TH:
                continue
            # local tangent
            if k == 0:
                p0, p1 = poly[0], poly[1]
            elif k >= len(poly) - 1:
                p0, p1 = poly[-2], poly[-1]
            else:
                p0, p1 = poly[k - 1], poly[k + 1]
            traj_vec = (p1 - p0).astype(np.float64)
            n = float(np.linalg.norm(traj_vec))
            if n < 1e-3:
                continue
            traj_heading = float(np.arctan2(traj_vec[1], traj_vec[0]))
            heading_diff = abs(wrap_to_pi(traj_heading - actor_heading))
            if heading_diff > HEADING_TH:
                continue
            vec_near = (poly[k] - actor_pos[:2]).astype(np.float64)
            forward = float(np.dot(vec_near, ego_dir))
            if forward <= FORWARD_TH:
                continue
            candidates.append((lid, dist))

        # 从路由序列中找到当前位置对应的车道，只关心未来的路由
        if route_seq and candidates:
            start_idx = None
            for lid, _ in sorted(candidates, key=lambda x: x[1]):
                if lid in route_seq:
                    start_idx = route_seq.index(lid)
                    break

            if start_idx is None:
                best_dist = float('inf')
                best_idx = 0
                for i, r_lid in enumerate(route_seq):
                    # 找到目前 route_seq 里离主车最近的车道
                    node = lane_graph.lanes.get(r_lid)
                    if node and node.centerline is not None and len(node.centerline) > 0:
                        dist = float(np.min(np.linalg.norm(node.centerline - actor_pos[None, :2], axis=1)))
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = i
                start_idx = best_idx

            if start_idx is not None:
                route_seq = route_seq[start_idx:]
        route_set = set(route_seq)

        if len(candidates) == 0:
            # reference_lines = []
            xs = actor_pos[0] + np.linspace(0, radius, int(radius)) * np.cos(actor_heading)
            ys = actor_pos[1] + np.linspace(0, radius, int(radius)) * np.sin(actor_heading)
            fake_line = np.stack([xs, ys, np.full_like(xs, actor_heading)], axis=-1)
            reference_lines = [fake_line]
            reference_line_route_ids = [route_seq[0] if route_seq else -1]
        else:
            # 优先选择在路由上的候选车道，作为起始车道
            cand_sorted = sorted(candidates, key=lambda x: x[1])
            start_lanes_on_route = [lid for lid, _ in cand_sorted if (not route_seq) or (lid in route_set)]
            start_lanes = start_lanes_on_route[:2] if start_lanes_on_route else [lid for lid, _ in cand_sorted[:6]]

            if allow_neighbor_refs:
                # 添加邻近车道，允许变道
                extra_starts = []
                for lid in list(start_lanes):
                    node = lane_graph.lanes.get(lid)
                    if node is None:
                        continue
                    for nb in (node.left_neighbors + node.right_neighbors):
                        if allowed_lanes is not None and nb not in allowed_lanes:
                            continue
                        extra_starts.append(nb)
                # unique preserve order
                seen = set(start_lanes)
                for nb in extra_starts:
                    if nb not in seen:
                        start_lanes.append(nb)
                        seen.add(nb)
                    if len(start_lanes) >= 12:
                        break
            # print("\n==== REF START LANES ====")
            # print("route_seq:", route_seq[:20])
            # print("candidates:", cand_sorted[:10])
            # print("start_lanes:", start_lanes)
            # print("=========================\n")

            def lane_length_xy(poly_xy: np.ndarray) -> float:
                return float(np.sum(np.linalg.norm(np.diff(poly_xy, axis=0), axis=1)))

            # DFS along exit_lanes topology
            all_paths: List[List[int]] = []

            # Build a quick route index for successor priority.
            route_index = {lid: i for i, lid in enumerate(route_seq)} if route_seq else {}

            def _ordered_successors(cur: int) -> List[int]:
                succs = [s for s in lane_graph.successors(cur) if s in lane_graph.lanes]
                if not succs:
                    return []
                if not route_seq or cur not in route_index:
                    return succs
                i = route_index[cur]
                # preferred = route_seq[i + 1] if i + 1 < len(route_seq) else None
                # if preferred is not None and preferred in succs:
                #     # Put the route successor first, keep others as alternatives.
                #     others = [s for s in succs if s != preferred]
                #     return [preferred] + others

                # 允许跳过少量 topology 缺失，但不能大跳到出口
                MAX_SKIP = 3
                future_window = route_seq[i + 1: i + 1 + MAX_SKIP]
                ordered = [s for s in future_window if s in succs]
                # print(
                #     "CUR:", cur,
                #     "ROUTE_NEXT:",
                #     route_seq[route_index[cur] + 1]
                #     if cur in route_index and route_index[cur] + 1 < len(route_seq)
                #     else None,
                #     "ALL_SUCC:",
                #     ordered
                # )
                return ordered
                # return succs

            def dfs(path: List[int], acc_len: float, offroute_budget: int):
                cur = path[-1]
                if acc_len >= radius:
                    all_paths.append(path)
                    return
                succs = _ordered_successors(cur)
                if not succs:
                    all_paths.append(path)
                    return
                expanded = False
                for nxt in succs:
                    if nxt in path:
                        continue
                    # 保持在路由上; 允许邻边拓展
                    if route_seq and nxt not in route_set:
                        # allow neighbor lanes only
                        allow_offroute = (nxt in neighbor_extra) and (offroute_budget > 0)
                        if not allow_offroute:
                            continue
                    poly_xy = lane_graph.lanes[nxt].centerline
                    if poly_xy is None or len(poly_xy) < 2:
                        continue
                    # keep within radius by endpoint proximity
                    if float(np.min(np.linalg.norm(poly_xy - actor_pos[None, :2], axis=1))) > radius * 1.2:
                        continue
                    expanded = True
                    next_budget = offroute_budget
                    if route_seq and (nxt not in route_set) and (nxt not in neighbor_extra):
                        next_budget -= 1
                    dfs(path + [nxt], acc_len + lane_length_xy(poly_xy), next_budget)
                if not expanded:
                    all_paths.append(path)

            offroute_budget = 1 if allow_neighbor_refs else 0
            for lid in start_lanes:
                # if route_seq exists, we can require start lane in trimmed route.
                if route_seq and lid not in route_set:
                    continue
                poly_xy = lane_graph.lanes[lid].centerline
                if poly_xy is None or len(poly_xy) < 2:
                    continue
                dfs([lid], lane_length_xy(poly_xy), offroute_budget=offroute_budget)

            # Merge lane centerlines into continuous polylines. Preserve original heading from polyline if present.
            reference_lines = []
            reference_line_route_ids = []

            def _dedup_append(lines: List[np.ndarray], ids: List[int], line: np.ndarray, route_id: int,
                              eps: float = 1.0) -> None:
                """Append line if it's not covered by existing ones. Remove existing ones covered by this line."""
                if line is None or len(line) < 2:
                    return
                to_remove = []
                for i, ex in enumerate(lines):
                    diff = line[:, None, :2] - ex[None, :, :2]
                    dists = np.linalg.norm(diff, axis=-1)

                    dist_line_to_ex = np.max(np.min(dists, axis=1))
                    dist_ex_to_line = np.max(np.min(dists, axis=0))

                    if dist_line_to_ex < eps:
                        # 'line' is completely covered by 'ex' (within eps), so it provides no new path info
                        return

                    if dist_ex_to_line < eps:
                        # 'ex' is completely covered by 'line', mark 'ex' for replacement
                        to_remove.append(i)

                # Pop out in reverse to maintain indices
                for i in reversed(to_remove):
                    lines.pop(i)
                    ids.pop(i)

                lines.append(line)
                ids.append(route_id)

            for path in all_paths:
                # print("\n==== DFS PATH ====")
                # print(path)
                # print("==================\n")
                merged = []
                for idx, lid in enumerate(path):
                    # We still have original polyline with heading in map_features_list.
                    raw = map_features_list.get(str(lid), None)
                    poly = None
                    if isinstance(raw, dict):
                        poly = raw.get('polyline', None)
                    if poly is None:
                        # fallback to xy-only centerline
                        poly_xy = lane_graph.lanes[lid].centerline
                        diff_y = np.diff(poly_xy[:, 1])
                        diff_x = np.diff(poly_xy[:, 0])
                        seg_hdg = np.arctan2(diff_y, diff_x)
                        heading = np.pad(seg_hdg, (1, 0), mode='edge')
                        poly = np.concatenate([poly_xy, heading[:, None]], axis=1)
                    poly = np.asarray(poly)
                    if idx > 0 and len(poly) > 1:
                        poly = poly[1:]
                    merged.append(poly)
                if not merged:
                    continue
                # Recorded lanes can be XY while a missing raw lane falls back
                # to XY+heading. Preserve homogeneous paths exactly; harmonize
                # only the previously crashing mixed representation.
                widths = {piece.shape[1] for piece in merged}
                if widths == {2, 3}:
                    normalized = []
                    for piece in merged:
                        if piece.shape[1] == 2:
                            segments = np.diff(piece, axis=0)
                            heading = np.arctan2(segments[:, 1], segments[:, 0])
                            heading = (np.r_[heading, heading[-1]] if len(heading)
                                       else np.full(len(piece), actor_heading))
                            piece = np.column_stack([piece, heading])
                        normalized.append(piece)
                    merged = normalized
                line = np.concatenate(merged, axis=0)
                _dedup_append(reference_lines, reference_line_route_ids, line, int(path[0]))

                # Cap number of reference lines to keep tensors bounded
                if len(reference_lines) >= 12:
                    break

            if len(reference_lines) == 0:
                # Last resort: allow unfiltered expansion (ignore route_set)
                for lid in start_lanes:
                    poly = map_features_list.get(str(lid), {}).get('polyline', None)
                    if poly is None:
                        continue
                    reference_lines.append(np.asarray(poly))
                    reference_line_route_ids.append(int(lid))

        # =========================
        # 6. 转成 feature tensor
        # =========================
        n_points = int(radius)
        M = len(reference_lines)

        position = np.zeros((M, n_points, 2))
        vector = np.zeros((M, n_points, 2))
        orientation = np.zeros((M, n_points))
        valid_mask = np.zeros((M, n_points), dtype=bool)

        future_projection = np.zeros((M, 8, 2))

        # route id per reference line
        route_id = np.full((M,), -1, dtype=np.int64)
        if 'reference_line_route_ids' in locals() and len(reference_line_route_ids) == M:
            route_id[:] = np.asarray(reference_line_route_ids, dtype=np.int64)

        future_samples = []
        actor_future = actor_features["position"][self.history_samples + 1:]
        if len(actor_future) > 0:
            future_samples = actor_future[9::10]  # 1Hz
            future_samples = [Point(xy) for xy in future_samples]

        def _crop_reference_line_from_ego(line: np.ndarray, ego_xy: np.ndarray, ego_heading: float) -> np.ndarray:
            line = np.asarray(line, dtype=np.float64)
            if line.ndim != 2 or line.shape[0] < 2:
                return line

            line_xy = line[:, :2]
            seg_vec = np.diff(line_xy, axis=0)
            seg_len = np.linalg.norm(seg_vec, axis=1)
            if not np.any(seg_len > 1e-6):
                return line

            linestring = LineString(line_xy)
            proj_dist = float(linestring.project(Point(np.asarray(ego_xy[:2], dtype=np.float64))))

            cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
            proj_dist = float(np.clip(proj_dist, 0.0, cum_len[-1]))
            seg_idx = int(np.searchsorted(cum_len[1:], proj_dist, side="right"))
            seg_idx = int(np.clip(seg_idx, 0, len(seg_len) - 1))

            seg_start = float(cum_len[seg_idx])
            seg_total = float(seg_len[seg_idx])
            if seg_total > 1e-6:
                alpha = float(np.clip((proj_dist - seg_start) / seg_total, 0.0, 1.0))
                proj_xy = line_xy[seg_idx] + alpha * seg_vec[seg_idx]
                proj_heading = float(np.arctan2(seg_vec[seg_idx, 1], seg_vec[seg_idx, 0]))
            else:
                alpha = 0.0
                proj_xy = line_xy[seg_idx].copy()
                proj_heading = float(line[seg_idx, 2] if line.shape[1] >= 3 else ego_heading)

            tail = line[seg_idx + 1:]
            if line.shape[1] >= 3:
                first = np.array([[proj_xy[0], proj_xy[1], proj_heading]], dtype=np.float64)
            else:
                first = np.array([[proj_xy[0], proj_xy[1]]], dtype=np.float64)

            cropped = np.concatenate([first, tail], axis=0)
            if cropped.shape[0] < 2:
                # Keep at least two points for downstream vector/orientation construction.
                cropped = line[max(seg_idx, 0): seg_idx + 2]
            return cropped

        def _resample_reference_line(
            line: np.ndarray,
            target_points: int,
            sample_spacing: float = 1.0,
        ) -> np.ndarray:
            line = np.asarray(line, dtype=np.float64)
            if line.ndim != 2 or line.shape[0] == 0:
                return line
            if line.shape[0] == 1:
                base_xy = np.repeat(line[:, :2], target_points, axis=0)
                if line.shape[1] >= 3:
                    base_hdg = np.repeat(line[:, 2], target_points, axis=0)
                else:
                    base_hdg = np.zeros((target_points,), dtype=np.float64)
                return np.concatenate([base_xy, base_hdg[:, None]], axis=1)

            line_xy = line[:, :2]
            seg_vec = np.diff(line_xy, axis=0)
            seg_len = np.linalg.norm(seg_vec, axis=1)
            valid_seg = seg_len > 1e-6
            if not np.any(valid_seg):
                base_xy = np.repeat(line_xy[:1], target_points, axis=0)
                base_hdg = np.full((target_points,), float(actor_heading), dtype=np.float64)
                return np.concatenate([base_xy, base_hdg[:, None]], axis=1)

            cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
            sample_dist = np.arange(target_points, dtype=np.float64) * float(sample_spacing)
            out_xy = np.zeros((target_points, 2), dtype=np.float64)
            out_hdg = np.zeros((target_points,), dtype=np.float64)

            if line.shape[1] >= 3:
                base_ori = np.asarray(line[:, 2], dtype=np.float64)
            else:
                base_ori = np.zeros((line.shape[0],), dtype=np.float64)
                base_ori[:-1] = np.arctan2(seg_vec[:, 1], seg_vec[:, 0])
                base_ori[-1] = base_ori[-2]

            last_valid_idx = int(np.where(valid_seg)[0][-1])
            last_heading = float(np.arctan2(seg_vec[last_valid_idx, 1], seg_vec[last_valid_idx, 0]))
            last_xy = line_xy[last_valid_idx + 1]
            total_len = float(cum_len[-1])

            for j, dist in enumerate(sample_dist):
                if dist <= total_len:
                    seg_idx = int(np.searchsorted(cum_len[1:], dist, side="right"))
                    seg_idx = int(np.clip(seg_idx, 0, len(seg_len) - 1))
                    if seg_len[seg_idx] > 1e-6:
                        alpha = float((dist - cum_len[seg_idx]) / seg_len[seg_idx])
                        alpha = float(np.clip(alpha, 0.0, 1.0))
                        out_xy[j] = line_xy[seg_idx] + alpha * seg_vec[seg_idx]
                        out_hdg[j] = float(np.arctan2(seg_vec[seg_idx, 1], seg_vec[seg_idx, 0]))
                    else:
                        out_xy[j] = line_xy[seg_idx]
                        out_hdg[j] = float(base_ori[seg_idx])
                else:
                    extra = dist - total_len
                    out_xy[j] = last_xy + extra * np.array(
                        [np.cos(last_heading), np.sin(last_heading)], dtype=np.float64
                    )
                    out_hdg[j] = last_heading
                    # 外推还是重复末端点
                    # out_xy[j] = last_xy
                    # out_hdg[j] = last_heading

            return np.concatenate([out_xy, out_hdg[:, None]], axis=1)

        for i, line in enumerate(reference_lines):
            # print(
            #     "REF_LINE",
            #     i,
            #     "route_id:",
            #     route_id[i],
            # )
            line = np.asarray(line, dtype=np.float64)
            cropped_line = _crop_reference_line_from_ego(line, actor_pos[:2], actor_heading)
            subsample = _resample_reference_line(cropped_line, n_points + 1, sample_spacing=1.0)
            n_valid = len(subsample)

            position[i, : n_valid - 1] = subsample[:-1, :2]
            vector[i, : n_valid - 1] = np.diff(subsample[:, :2], axis=0)

            if cropped_line.shape[1] >= 3:
                orientation[i, : n_valid - 1] = subsample[:-1, 2]
            else:
                orientation[i, : n_valid - 1] = np.arctan2(
                    vector[i, : n_valid - 1, 1],
                    vector[i, : n_valid - 1, 0],
                )
            valid_mask[i, : n_valid - 1] = True

            # =========================
            # Pad the rest with the last valid point to prevent bent lines
            # =========================
            if 0 < n_valid - 1 < n_points:
                position[i, n_valid - 1:] = position[i, n_valid - 2]
                vector[i, n_valid - 1:] = vector[i, n_valid - 2]
                orientation[i, n_valid - 1:] = orientation[i, n_valid - 2]

            if len(actor_future) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    linestring = LineString(cropped_line[:, :2])

                    for j, future_sample in enumerate(future_samples[:8]):
                        future_projection[i, j, 0] = linestring.project(future_sample)
                        future_projection[i, j, 1] = linestring.distance(future_sample)

            # if i == 0 and getattr(self, "_debug_ref_geom_log_cnt", 0) < 8:
            #     # #region debug-point E:reference-line-snapshot
            #     import json, urllib.request;
            #     _p = '.dbg/pluto-turning-drift.env';
            #     _u, _s = 'http://127.0.0.1:7777/event', 'pluto-turning-drift';
            #     exec(
            #         "try:\n with open(_p) as f: c=f.read(); _u=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SERVER_URL=')),_u); _s=next((l.split('=',1)[1] for l in c.split('\\n') if l.startswith('DEBUG_SESSION_ID=')),_s)\nexcept: pass")
            #     _dbg_n = int(min(10, max(n_valid - 1, 0)))
            #     _dbg_ref = {
            #         "ref_index": int(i),
            #         "route_id": int(route_id[i]) if len(route_id) > i else -1,
            #         "ego_pos": np.asarray(actor_pos[:2], dtype=np.float64).tolist(),
            #         "ego_heading": float(actor_heading),
            #         "position_first10": np.asarray(position[i, :_dbg_n], dtype=np.float64).tolist(),
            #         "vector_first10": np.asarray(vector[i, :_dbg_n], dtype=np.float64).tolist(),
            #         "orientation_first10": np.asarray(orientation[i, :_dbg_n], dtype=np.float64).tolist(),
            #         "future_projection": np.asarray(future_projection[i], dtype=np.float64).tolist(),
            #         "valid_points": int(n_valid - 1),
            #     }
            #     urllib.request.urlopen(urllib.request.Request(_u, data=json.dumps(
            #         {"sessionId": _s, "runId": "pre", "hypothesisId": "E",
            #          "location": "unitraj/datasets/Pluto_dataset/Pluto_dataset.py:_get_reference_line_feature",
            #          "msg": "[DEBUG] reference line snapshot", "data": _dbg_ref}).encode(),
            #                                                   headers={"Content-Type": "application/json"})).read()
            #     self._debug_ref_geom_log_cnt = getattr(self, "_debug_ref_geom_log_cnt", 0) + 1
            #     # #endregion

        return {
            "position": position,
            "vector": vector,
            "orientation": orientation,
            "valid_mask": valid_mask,
            "future_projection": future_projection,
            "route_id": route_id,
        }
