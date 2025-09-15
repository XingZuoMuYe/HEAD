import time
import os
os.environ["MPLBACKEND"] = "Agg"   # 离线保存最稳
from datetime import datetime
import torch
# main_head.py
from collections import defaultdict
from head.manager.config_manager import get_final_config
from head.manager.imitation_selector import resolve_imitation_strategy
from head.evolution_engine.env_builder.env import make_env
from head.policy.imitation_policy.dataset_builder.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType
from head.policy.imitation_policy.dataset_builder.types import object_type, polyline_type
default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)
from head.policy.imitation_policy.dataset_builder import common_utils
from head.policy.imitation_policy.utils import visualization
from head.policy.imitation_policy.utils.map_utils import get_map_data, get_manually_split_map_data
from head.policy.imitation_policy.utils.agent_utils import transform_trajs_to_center_coords, get_agent_data, get_interested_agents, trajectory_filter

class UnitrajInference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.global_cfg = None
        self.imitation_algo = None
        self.env = None
        self.device = "cuda"
        self.initialize_model_and_env()

    def initialize_model_and_env(self):
        """Initialize the imitation model and environment."""
        self.imitation_algo, self.global_cfg = resolve_imitation_strategy(self.cfg)
        self.env = make_env(self.cfg)
        ckpt = torch.load(self.global_cfg.ckpt_path, map_location=self.device, weights_only=False)
        self.imitation_algo.load_state_dict(ckpt["state_dict"])
        self.imitation_algo = self.imitation_algo.to(self.device)

    def process_scenario_data(self, scenario):
        """Process scenario data to extract track and map information."""
        traffic_lights = scenario['dynamic_map_states']
        tracks = scenario['tracks']
        map_feat = scenario['map_features']

        past_length = self.global_cfg['past_len']
        future_length = self.global_cfg['future_len']
        total_steps = past_length + future_length
        starting_frame = 0
        ending_frame = starting_frame + total_steps
        trajectory_sample_interval = self.global_cfg['trajectory_sample_interval']
        frequency_mask = generate_mask(past_length - 1, total_steps, trajectory_sample_interval)

        track_infos = self.extract_track_infos(tracks, starting_frame, ending_frame, total_steps, frequency_mask)
        scenario['metadata']['ts'] = scenario['metadata']['ts'][:total_steps]

        map_infos = self.extract_map_infos(map_feat)
        dynamic_map_infos = self.extract_dynamic_map_infos(traffic_lights, total_steps)

        ret = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        ret.update(scenario['metadata'])
        ret['timestamps_seconds'] = ret.pop('ts')
        ret['current_time_index'] = self.global_cfg['past_len'] - 1
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

    def process_single_map_feature(self, v, polyline_type_):
        """Process a single map feature based on its type."""
        cur_info = {'id': v.get('id', None), 'type': v['type']}
        polyline = None
        if polyline_type_ in [1, 2, 3]:
            cur_info.update({key: v.get(key, None) for key in ['speed_limit_mph', 'interpolating', 'entry_lanes']})
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
            polyline = interpolate_polyline(v['polyline'])
        elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13]:
            polyline = interpolate_polyline(v.get('polyline', v.get('polygon', None)))
        elif polyline_type_ in [15, 16]:
            polyline = interpolate_polyline(v['polyline'])
            cur_info['type'] = 7
        elif polyline_type_ in [17]:
            cur_info['lane_ids'] = v['lane']
            cur_info['position'] = v['position']
            polyline = v['position'][np.newaxis]
        elif polyline_type_ in [18, 19]:
            polyline = v['polygon']
        return cur_info, polyline

    def get_map_category(self, polyline_type_):
        """Map polyline type to category string."""
        if polyline_type_ in [1, 2, 3]:
            return 'lane'
        elif polyline_type_ in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16]:
            return 'road_line'
        elif polyline_type_ in [17]:
            return 'stop_sign'
        elif polyline_type_ in [18, 19]:
            return 'crosswalk'
        else:
            return 'others'

    def extract_dynamic_map_infos(self, traffic_lights, total_steps):
        """Extract dynamic map information (e.g., traffic lights)."""
        dynamic_map_infos = {'lane_id': [], 'state': [], 'stop_point': []}
        for k, v in traffic_lights.items():
            lane_id, state, stop_point = [], [], []
            for cur_signal in v['state']['object_state']:
                lane_id.append(str(v['lane']))
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
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }
        else:
            sample_list = list(ret['tracks_to_predict'].keys())
            sample_list = list(set(sample_list))
            tracks_to_predict = {
                'track_index': [track_infos['object_id'].index(id) for id in sample_list if
                                id in track_infos['object_id']],
                'object_type': [track_infos['object_type'][track_infos['object_id'].index(id)] for id in sample_list if
                                id in track_infos['object_id']],
            }
        ret['tracks_to_predict'] = tracks_to_predict
        return ret

    def prepare_agent_and_map_data(self, info):
        """Prepare agent data and map data for model input."""
        scene_id = info['scenario_id']
        sdc_track_index = info['sdc_track_index']
        current_time_index = info['current_time_index']
        timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)
        track_infos = info['track_infos']
        track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
        obj_types = np.array(track_infos['object_type'])
        obj_trajs_full = track_infos['trajs']
        obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
        obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

        center_objects, track_index_to_predict = get_interested_agents(
            self.global_cfg, track_index_to_predict=track_index_to_predict,
            obj_trajs_full=obj_trajs_full, current_time_index=current_time_index,
            obj_types=obj_types, scene_id=scene_id
        )
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

    def create_batch_dict(self, ret_list):
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

    def run_inference(self, to_device_func, num_iterations=20):
        """Run inference and return the last batch and prediction."""
        last_batch_dict = None
        last_prediction = None

        for i in range(num_iterations):
            t1 = time.time()
            scenario_data = self.env.engine.data_manager._scenarios
            scenario = next(iter(scenario_data.values()))
            info = self.process_scenario_data(scenario)
            t2 = time.time()

            ret_list = self.prepare_agent_and_map_data(info)
            if ret_list is None:
                continue

            batch_dict = self.create_batch_dict(ret_list)
            batch_dict = to_device_func(batch_dict, self.device)
            t3 = time.time()

            prediction, loss = self.imitation_algo.forward(batch_dict)
            last_batch_dict = batch_dict
            last_prediction = prediction
            t4 = time.time()

            self.env.head_renderer.render(
                screen_record=False,
                scaling=6,
                film_size=(6000, 400),
                mode="topdown"
            )
            self.env.step(([0, 0]))

            print(f"迭代 {i}:")
            print(f"时间1 (数据处理): {t2 - t1:.4f}s")
            print(f"时间2 (智能体准备): {t3 - t2:.4f}s")
            print(f"时间3 (推理): {t4 - t3:.4f}s")
            print(f"总时间: {t4 - t1:.4f}s")

        return last_batch_dict, last_prediction