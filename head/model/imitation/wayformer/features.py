import time
from collections import defaultdict
from torch.utils.data import Dataset
import os
os.environ["MPLBACKEND"] = "Agg"   # 离线保存最稳
import torch
from head.model.imitation.common.common_utils import get_polyline_dir, generate_mask, get_kalman_difficulty, get_trajectory_type, interpolate_polyline
import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType
from head.model.imitation.common.my_types import object_type, polyline_type
from head.model.imitation.wayformer.map_utils import get_map_data, get_manually_split_map_data
from head.model.imitation.wayformer.agent_utils import get_agent_data, get_interested_agents, trajectory_filter

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)


def create_batch_dict(ret_list):
    """Create batch dictionary from list of processed data."""
    batch_size = len(ret_list)
    key_to_list = {}
    for key in ret_list[0].keys():
        key_to_list[key] = [ret_list[bs_idx][key] for bs_idx in range(batch_size)]

    input_dict = {}
    for key, val_list in key_to_list.items():
        try:
            input_dict[key] = torch.from_numpy(np.stack(val_list, axis=0))
        except:
            input_dict[key] = val_list

    return {'batch_size': batch_size, 'input_dict': input_dict, 'batch_sample_count': batch_size}


class UnitrajTestDataset(Dataset):
    def __init__(self, cfg):
        self.global_cfg = cfg

    def process_scenario(self, scenario, current_step):
        t1 = time.time()
        info = self.process_scenario_data(scenario, current_step)
        t2 = time.time()
        ret_list = self.prepare_agent_and_map_data(info)
        batch_dict = create_batch_dict(ret_list)

        return batch_dict, self.center_objects

    def process_scenario_data(self, scenario, current_step):
        """Process scenario data to extract track and map information."""
        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']

        # 改变时间逻辑，滑动更新
        past_length = self.global_cfg['past_len']
        future_length = self.global_cfg['future_len']
        # The network window size is not the closed-loop episode horizon.
        # Keep the historical prefix unchanged, but retain the current frame
        # after index80 instead of indexing outside an81-frame array.
        total_steps = max(past_length + future_length, int(current_step) + 1)
        starting_frame = 0
        ending_frame = starting_frame + total_steps
        trajectory_sample_interval = self.global_cfg['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)
        track_infos = self.extract_track_infos(tracks, starting_frame, ending_frame, total_steps, frequency_mask)
        map_infos, cached = self.get_cached_map_infos(scenario)
        dynamic_map_infos = self.extract_dynamic_map_infos(traffic_lights, total_steps)
        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        # Never truncate the live simulator's metadata in place: later calls
        # need timestamps beyond the first prediction window.
        ret['timestamps_seconds'] = ret.pop('ts')[:total_steps]
        ret['current_time_index'] = current_step
        ret['sdc_track_index'] = track_infos['object_id'].index(ret['sdc_id'])

        ret = self.prepare_tracks_to_predict(ret, track_infos)
        ret['map_center'] = scenario['metadata'].get('map_center', np.zeros(3))[np.newaxis]
        ret['track_length'] = total_steps

        return ret

    def extract_track_infos(self, tracks, starting_frame, ending_frame, total_steps, frequency_mask):
        """Extract track information from raw tracks data."""
        track_infos = {'object_id': [], 'object_type': [], 'trajs': []}
        for k, v in tracks.items():
            state = v['state']

            for key in state:
                if isinstance(state[key], list):
                    state[key] = np.array(state[key])
                if len(state[key].shape) == 1:
                    state[key] = np.expand_dims(state[key], axis=-1)
            all_state = np.concatenate([
                state['position'], state['length'], state['width'], state['height'],
                state['heading'], state['velocity'], state['valid']
            ], axis=-1)
            if all_state.shape[0] < ending_frame:
                all_state = np.pad(all_state, ((ending_frame - all_state.shape[0], 0), (0, 0)))
            all_state = all_state[starting_frame:ending_frame]
            assert all_state.shape[0] == total_steps, f'Error: {all_state.shape[0]} != {total_steps}'

            track_infos['object_id'].append(k)
            track_infos['object_type'].append(object_type[v['type']])
            track_infos['trajs'].append(all_state)

        track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)
        track_infos['trajs'][..., -1] *= frequency_mask[np.newaxis]
        return track_infos

    def extract_map_infos(self, map_feat):
        """Extract map information from map features."""
        map_infos = {
            'lane': [], 'road_line': [], 'road_edge': [], 'stop_sign': [],
            'crosswalk': [], 'speed_bump': [],
        }
        polylines = []
        point_cnt = 0
        for k, v in map_feat.items():
            polyline_type_ = polyline_type[v['type']]
            if polyline_type_ == 0:
                continue
            cur_info, polyline = self.process_single_map_feature(v, polyline_type_)
            if polyline is not None:
                if polyline.shape[-1] == 2:
                    polyline = np.concatenate((polyline, np.zeros((polyline.shape[0], 1))), axis=-1)
                try:
                    cur_polyline_dir = get_polyline_dir(polyline)
                    type_array = np.full((polyline.shape[0], 1), polyline_type_)
                    cur_polyline = np.concatenate((polyline, cur_polyline_dir, type_array), axis=-1)
                except:
                    cur_polyline = np.zeros((0, 7), dtype=np.float32)
                polylines.append(cur_polyline)
                cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
                point_cnt += len(cur_polyline)
                map_infos[self.get_map_category(polyline_type_)].append(cur_info)

        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
        map_infos['all_polylines'] = polylines
        return map_infos

    # ========== 地图缓存辅助函数 ==========
    def get_cached_map_infos(self, scenario):
        """从缓存中获取或构建地图信息"""
        scenario_id = scenario["metadata"].get("scenario_id", None)
        map_feat = scenario["map_features"]

        # 首次使用该类时可能没有缓存属性
        if not hasattr(self, "map_cache"):
            self.map_cache = {}

        # 检查缓存命中
        if scenario_id in self.map_cache:
            # ✅ 命中缓存
            # print(f"[MapCache] Hit for scenario: {scenario_id}")
            return self.map_cache[scenario_id], True

        # ❌ 缓存未命中 → 重新提取并存储
        t0 = time.time()
        map_infos = self.extract_map_infos(map_feat)
        self.map_cache[scenario_id] = map_infos
        # print(f"[MapCache] Built new map for {scenario_id}, time={time.time() - t0:.3f}s")

        return map_infos, False

    def process_single_map_feature(self, v, polyline_type_):
        """Process a single map feature based on its type."""
        cur_info = {'id': v.get('id'), 'type': v['type']}
        polyline = None

        get_val = v.get
        # --- Lane 类特征 ---
        if polyline_type_ <= 3:  # [1, 2, 3]
            cur_info.update({
                'speed_limit_mph': get_val('speed_limit_mph'),
                'interpolating': get_val('interpolating'),
                'entry_lanes': get_val('entry_lanes'),
            })

            try:
                cur_info['left_boundary'] = [{'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                                              'feature_id': x['feature_id'], 'boundary_type': 'UNKNOWN'}
                                             for x in v['left_neighbor']]
                cur_info['right_boundary'] = [{'start_index': x['self_start_index'], 'end_index': x['self_end_index'],
                                               'feature_id': x['feature_id'], 'boundary_type': 'UNKNOWN'}
                                              for x in v['right_neighbor']]
            except:
                cur_info['left_boundary'] = []
                cur_info['right_boundary'] = []

            polyline_data = get_val('polyline')
            if polyline_data is not None and len(polyline_data) > 1:
                polyline = interpolate_polyline(polyline_data)
            else:
                polyline = np.asarray(polyline_data, dtype=np.float32)

        # --- road_line / edge ---
        elif 6 <= polyline_type_ <= 13:
            polyline = interpolate_polyline(get_val('polyline', get_val('polygon')))

        # --- special markings ---
        elif 15 <= polyline_type_ <= 16:
            polyline = interpolate_polyline(get_val('polyline'))
            cur_info['type'] = 7  # 保留原行为

        # --- stop_sign ---
        elif polyline_type_ == 17:
            cur_info['lane_ids'] = v['lane']
            cur_info['position'] = v['position']
            polyline = np.expand_dims(cur_info['position'], 0)

        # --- crosswalk/polygon 类 ---
        elif polyline_type_ in (18, 19):
            polyline = np.array(v['polygon'], dtype=np.float32)

        return cur_info, polyline


    def get_map_category(self, polyline_type_):
        """Map polyline type to category string."""
        _category_map = {
            **{k: 'lane' for k in [1, 2, 3]},
            **{k: 'road_line' for k in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]},
            17: 'stop_sign',
            18: 'crosswalk',
            19: 'crosswalk'
        }
        return _category_map.get(polyline_type_, 'others')


    def extract_dynamic_map_infos(self, traffic_lights, total_steps):
        """Extract dynamic map information (e.g., traffic lights)."""
        dynamic_map_infos = {'lane_id': [], 'state': [], 'stop_point': []}
        for k, v in traffic_lights.items():
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:
                lane_id.append(str(v['lane']) if v.get('lane') else k)
                state.append(cur_signal)
                if type(v['stop_point']) == list:
                    stop_point.append(v['stop_point'])
                else:
                    stop_point.append(v['stop_point'].tolist())
            lane_id = lane_id[:total_steps]
            state = state[:total_steps]
            stop_point = stop_point[:total_steps]
            dynamic_map_infos['lane_id'].append(np.array([lane_id]))
            dynamic_map_infos['state'].append(np.array([state]))
            dynamic_map_infos['stop_point'].append(np.array([stop_point]))
        return dynamic_map_infos

    def prepare_tracks_to_predict(self, ret, track_infos):
        """Prepare tracks to predict based on configuration."""
        if self.global_cfg['only_train_on_ego']:
            tracks_to_predict = {
                'track_index': [ret['sdc_track_index']],
                'difficulty': [0],
                'object_type': [MetaDriveType.VEHICLE]
            }
        elif ret.get('tracks_to_predict', None) is None:
            filtered_tracks = trajectory_filter(ret)
            sample_list = list(filtered_tracks.keys())
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list
                                if
                                id in track_infos['object_id']],
            }
        else:
            sample_list = list(ret['tracks_to_predict'].keys())
            sample_list = list(set(sample_list))
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list
                                if
                                id in track_infos['object_id']],
            }
        ret['tracks_to_predict'] = tracks_to_predict
        return ret

    def prepare_agent_and_map_data(self, info):
        """Prepare agent data and map data for model input."""
        scene_id = info['scenario_id']
        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']

        timestamps = np.array(info['timestamps_seconds'][current_time_index - 21:current_time_index],
                              dtype=np.float32)
        track_infos = info['track_infos']
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']
        obj_trajs_past = obj_trajs_full[:, current_time_index - 21: current_time_index]
        # print('last_po',obj_trajs_past[int(track_index_to_predict),-1:,:2])
        obj_trajs_future = obj_trajs_full[:, current_time_index:]
        # print('next_po', obj_trajs_future[int(track_index_to_predict), 0, :2])

        center_objects, track_index_to_predict = get_interested_agents(
            self.global_cfg, track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full, current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
        self.center_objects = center_objects

        if center_objects is None:
            return None

        sample_num = center_objects.shape[0]

        (obj_trajs_data, obj_trajs_mask, obj_trajs_pos, obj_trajs_last_pos, obj_trajs_future_state,
         obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask, center_gt_final_valid_idx,
         track_index_to_predict_new) = get_agent_data(
            self.global_cfg, center_objects=center_objects, obj_trajs_past=obj_trajs_past,
            obj_trajs_future=obj_trajs_future, track_index_to_predict=track_index_to_predict,
            sdc_track_index=sdc_track_index, timestamps=timestamps, obj_types=obj_types
        )

        ret_dict = {
            'scenario_id': np.array([scene_id] * len(track_index_to_predict)),
            'obj_trajs': obj_trajs_data,
            'obj_trajs_mask': obj_trajs_mask,
            'track_index_to_predict': track_index_to_predict_new,
            'obj_trajs_pos': obj_trajs_pos,
            'obj_trajs_last_pos': obj_trajs_last_pos,

            'center_objects_world': center_objects,
            'center_objects_id': np.array(track_infos['object_id'])[track_index_to_predict],
            'center_objects_type': np.array(track_infos['object_type'])[track_index_to_predict],
            'map_center': info['map_center'],

            'obj_trajs_future_state': obj_trajs_future_state,
            'obj_trajs_future_mask': obj_trajs_future_mask,
            'center_gt_trajs': center_gt_trajs,
            'center_gt_trajs_mask': center_gt_trajs_mask,
            'center_gt_final_valid_idx': center_gt_final_valid_idx,
            'center_gt_trajs_src': obj_trajs_full[track_index_to_predict]
        }

        if info['map_infos']['all_polylines'].__len__() == 0:
            info['map_infos']['all_polylines'] = np.zeros((2, 7), dtype=np.float32)
            print(f'Warning: empty HDMap {scene_id}')

        if self.global_cfg.manually_split_lane:
            map_polylines_data, map_polylines_mask, map_polylines_center = get_manually_split_map_data(
                self.global_cfg, center_objects=center_objects, map_infos=info['map_infos'])
        else:
            map_polylines_data, map_polylines_mask, map_polylines_center = get_map_data(
                self.global_cfg, center_objects=center_objects, map_infos=info['map_infos'])

        ret_dict['map_polylines'] = map_polylines_data
        ret_dict['map_polylines_mask'] = map_polylines_mask.astype(bool)
        ret_dict['map_polylines_center'] = map_polylines_center

        self.mask_attributes(ret_dict)
        for k, v in ret_dict.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float64:
                ret_dict[k] = v.astype(np.float32)

        ret_dict['map_center'] = ret_dict['map_center'].repeat(sample_num, axis=0)
        ret_dict['dataset_name'] = [info['dataset']] * sample_num

        ret_list = []
        for i in range(sample_num):
            ret_dict_i = {k: v[i] for k, v in ret_dict.items()}
            ret_list.append(ret_dict_i)

        get_kalman_difficulty(ret_list)
        get_trajectory_type(ret_list)
        return ret_list

    def mask_attributes(self, ret_dict):
        """Mask out unused attributes."""
        masked_attributes = self.global_cfg['masked_attributes']
        if 'z_axis' in masked_attributes:
            ret_dict['obj_trajs'][..., 2] = 0
            ret_dict['map_polylines'][..., 2] = 0
        if 'size' in masked_attributes:
            ret_dict['obj_trajs'][..., 3:6] = 0
        if 'velocity' in masked_attributes:
            ret_dict['obj_trajs'][..., 25:27] = 0
        if 'acceleration' in masked_attributes:
            ret_dict['obj_trajs'][..., 27:29] = 0
        if 'heading' in masked_attributes:
            ret_dict['obj_trajs'][..., 23:25] = 0
