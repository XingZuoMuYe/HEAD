"""
Author: ShuaixiPan
Date: 2025-09-11
Description: Agent related utility functions for processing agent data.
"""

import numpy as np
from collections import defaultdict
from head.policy.imitation_policy.dataset_builder import common_utils
from head.policy.imitation_policy.dataset_builder.types import object_type

default_value = 0
object_type = defaultdict(lambda: default_value, object_type)


def transform_trajs_to_center_coords(obj_trajs, center_xyz, center_heading, heading_index,
                                     rot_vel_index=None):
    """
    Args:
        obj_trajs (num_objects, num_timestamps, num_attrs):
            first three values of num_attrs are [x, y, z] or [x, y]
        center_xyz (num_center_objects, 3 or 2): [x, y, z] or [x, y]
        center_heading (num_center_objects):
        heading_index: the index of heading angle in the num_attr-axis of obj_trajs
    """
    num_objects, num_timestamps, num_attrs = obj_trajs.shape
    num_center_objects = center_xyz.shape[0]
    assert center_xyz.shape[0] == center_heading.shape[0]
    assert center_xyz.shape[1] in [3, 2]

    obj_trajs = np.tile(obj_trajs[None, :, :, :], (num_center_objects, 1, 1, 1))
    obj_trajs[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    obj_trajs[:, :, :, 0:2] = common_utils.rotate_points_along_z(
        points=obj_trajs[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
        angle=-center_heading
    ).reshape(num_center_objects, num_objects, num_timestamps, 2)

    obj_trajs[:, :, :, heading_index] -= center_heading[:, None, None]

    # rotate direction of velocity
    if rot_vel_index is not None:
        assert len(rot_vel_index) == 2
        obj_trajs[:, :, :, rot_vel_index] = common_utils.rotate_points_along_z(
            points=obj_trajs[:, :, :, rot_vel_index].reshape(num_center_objects, -1, 2),
            angle=-center_heading
        ).reshape(num_center_objects, num_objects, num_timestamps, 2)

    return obj_trajs


def get_agent_data(
        config, center_objects, obj_trajs_past, obj_trajs_future, track_index_to_predict, sdc_track_index, timestamps,
        obj_types
):
    num_center_objects = center_objects.shape[0]
    num_objects, num_timestamps, box_dim = obj_trajs_past.shape
    obj_trajs = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_past,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )

    object_onehot_mask = np.zeros((num_center_objects, num_objects, num_timestamps, 5))
    object_onehot_mask[:, obj_types == 1, :, 0] = 1
    object_onehot_mask[:, obj_types == 2, :, 1] = 1
    object_onehot_mask[:, obj_types == 3, :, 2] = 1
    object_onehot_mask[np.arange(num_center_objects), track_index_to_predict, :, 3] = 1
    object_onehot_mask[:, sdc_track_index, :, 4] = 1

    object_time_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, num_timestamps + 1))
    for i in range(num_timestamps):
        object_time_embedding[:, :, i, i] = 1
    object_time_embedding[:, :, :, -1] = timestamps

    object_heading_embedding = np.zeros((num_center_objects, num_objects, num_timestamps, 2))
    object_heading_embedding[:, :, :, 0] = np.sin(obj_trajs[:, :, :, 6])
    object_heading_embedding[:, :, :, 1] = np.cos(obj_trajs[:, :, :, 6])

    vel = obj_trajs[:, :, :, 7:9]
    vel_pre = np.roll(vel, shift=1, axis=2)
    acce = (vel - vel_pre) / 0.1
    acce[:, :, 0, :] = acce[:, :, 1, :]

    obj_trajs_data = np.concatenate([
        obj_trajs[:, :, :, 0:6],
        object_onehot_mask,
        object_time_embedding,
        object_heading_embedding,
        obj_trajs[:, :, :, 7:9],
        acce,
    ], axis=-1)

    obj_trajs_mask = obj_trajs[:, :, :, -1]
    obj_trajs_data[obj_trajs_mask == 0] = 0

    obj_trajs_future = obj_trajs_future.astype(np.float32)
    obj_trajs_future = transform_trajs_to_center_coords(
        obj_trajs=obj_trajs_future,
        center_xyz=center_objects[:, 0:3],
        center_heading=center_objects[:, 6],
        heading_index=6, rot_vel_index=[7, 8]
    )
    obj_trajs_future_state = obj_trajs_future[:, :, :, [0, 1, 7, 8]]  # (x, y, vx, vy)
    obj_trajs_future_mask = obj_trajs_future[:, :, :, -1]
    obj_trajs_future_state[obj_trajs_future_mask == 0] = 0

    center_obj_idxs = np.arange(len(track_index_to_predict))
    center_gt_trajs = obj_trajs_future_state[center_obj_idxs, track_index_to_predict]
    center_gt_trajs_mask = obj_trajs_future_mask[center_obj_idxs, track_index_to_predict]
    center_gt_trajs[center_gt_trajs_mask == 0] = 0

    assert obj_trajs_past.__len__() == obj_trajs_data.shape[1]
    valid_past_mask = np.logical_not(obj_trajs_past[:, :, -1].sum(axis=-1) == 0)

    obj_trajs_mask = obj_trajs_mask[:, valid_past_mask]
    obj_trajs_data = obj_trajs_data[:, valid_past_mask]
    obj_trajs_future_state = obj_trajs_future_state[:, valid_past_mask]
    obj_trajs_future_mask = obj_trajs_future_mask[:, valid_past_mask]

    obj_trajs_pos = obj_trajs_data[:, :, :, 0:3]
    num_center_objects, num_objects, num_timestamps, _ = obj_trajs_pos.shape
    obj_trajs_last_pos = np.zeros((num_center_objects, num_objects, 3), dtype=np.float32)
    for k in range(num_timestamps):
        cur_valid_mask = obj_trajs_mask[:, :, k] > 0
        obj_trajs_last_pos[cur_valid_mask] = obj_trajs_pos[:, :, k, :][cur_valid_mask]

    center_gt_final_valid_idx = np.zeros((num_center_objects), dtype=np.float32)
    for k in range(center_gt_trajs_mask.shape[1]):
        cur_valid_mask = center_gt_trajs_mask[:, k] > 0
        center_gt_final_valid_idx[cur_valid_mask] = k

    max_num_agents = config['max_num_agents']
    object_dist_to_center = np.linalg.norm(obj_trajs_data[:, :, -1, 0:2], axis=-1)

    object_dist_to_center[obj_trajs_mask[..., -1] == 0] = 1e10
    topk_idxs = np.argsort(object_dist_to_center, axis=-1)[:, :max_num_agents]

    topk_idxs = np.expand_dims(topk_idxs, axis=-1)
    topk_idxs = np.expand_dims(topk_idxs, axis=-1)

    obj_trajs_data = np.take_along_axis(obj_trajs_data, topk_idxs, axis=1)
    obj_trajs_mask = np.take_along_axis(obj_trajs_mask, topk_idxs[..., 0], axis=1)
    obj_trajs_pos = np.take_along_axis(obj_trajs_pos, topk_idxs, axis=1)
    obj_trajs_last_pos = np.take_along_axis(obj_trajs_last_pos, topk_idxs[..., 0], axis=1)
    obj_trajs_future_state = np.take_along_axis(obj_trajs_future_state, topk_idxs, axis=1)
    obj_trajs_future_mask = np.take_along_axis(obj_trajs_future_mask, topk_idxs[..., 0], axis=1)
    track_index_to_predict_new = np.zeros(len(track_index_to_predict), dtype=np.int64)

    obj_trajs_data = np.pad(obj_trajs_data, ((0, 0), (0, max_num_agents - obj_trajs_data.shape[1]), (0, 0), (0, 0)))
    obj_trajs_mask = np.pad(obj_trajs_mask, ((0, 0), (0, max_num_agents - obj_trajs_mask.shape[1]), (0, 0)))
    obj_trajs_pos = np.pad(obj_trajs_pos, ((0, 0), (0, max_num_agents - obj_trajs_pos.shape[1]), (0, 0), (0, 0)))
    obj_trajs_last_pos = np.pad(obj_trajs_last_pos,
                                ((0, 0), (0, max_num_agents - obj_trajs_last_pos.shape[1]), (0, 0)))
    obj_trajs_future_state = np.pad(obj_trajs_future_state,
                                    ((0, 0), (0, max_num_agents - obj_trajs_future_state.shape[1]), (0, 0), (0, 0)))
    obj_trajs_future_mask = np.pad(obj_trajs_future_mask,
                                   ((0, 0), (0, max_num_agents - obj_trajs_future_mask.shape[1]), (0, 0)))

    return (obj_trajs_data, obj_trajs_mask.astype(bool), obj_trajs_pos, obj_trajs_last_pos,
            obj_trajs_future_state, obj_trajs_future_mask, center_gt_trajs, center_gt_trajs_mask,
            center_gt_final_valid_idx,
            track_index_to_predict_new)


def get_interested_agents(config, track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
    center_objects_list = []
    track_index_to_predict_selected = []
    selected_type = config['object_type']
    selected_type = [object_type[x] for x in selected_type]
    for k in range(len(track_index_to_predict)):
        obj_idx = track_index_to_predict[k]

        if obj_trajs_full[obj_idx, current_time_index, -1] == 0:
            print(f'Warning: obj_idx={obj_idx} is not valid at time step {current_time_index}, scene_id={scene_id}')
            continue
        if obj_types[obj_idx] not in selected_type:
            continue

        center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
        track_index_to_predict_selected.append(obj_idx)
    if len(center_objects_list) == 0:
        print(f'Warning: no center objects at time step {current_time_index}, scene_id={scene_id}')
        return None, []
    center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
    track_index_to_predict = np.array(track_index_to_predict_selected)
    return center_objects, track_index_to_predict


def trajectory_filter(data):

        trajs = data['track_infos']['trajs']
        current_idx = data['current_time_index']
        obj_summary = data['object_summary']

        tracks_to_preidct = {}
        for idx,(k,v) in enumerate(obj_summary.items()):
            type = v['type']
            positions = trajs[idx, :, 0:2]
            validity = trajs[idx, :, -1]
            if type not in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']: continue
            valid_ratio = v['valid_length']/v['track_length']
            if valid_ratio < 0.5: continue
            moving_distance = v['moving_distance']
            if moving_distance < 2.0 and type=='VEHICLE': continue
            is_valid_at_m = validity[current_idx]>0
            if not is_valid_at_m: continue

            # past_traj = positions[:current_idx+1, :]  # Time X (x,y)
            # gt_future = positions[current_idx+1:, :]
            # valid_past = count_valid_steps_past(validity[:current_idx+1])


            future_mask =validity[current_idx+1:]
            future_mask[-1]=0
            idx_of_first_zero = np.where(future_mask == 0)[0]
            idx_of_first_zero = len(future_mask) if len(idx_of_first_zero) == 0 else idx_of_first_zero[0]

            #past_trajectory_valid = past_traj[-valid_past:, :]  # Time(valid) X (x,y)

            # try:
            #     kalman_traj = estimate_kalman_filter(past_trajectory_valid, idx_of_first_zero)  # (x,y)
            #     kalman_diff = calculate_epe(kalman_traj, gt_future[idx_of_first_zero-1])
            # except:
            #     continue
            # if kalman_diff < 20: continue

            tracks_to_preidct[k] = {'track_index': idx, 'track_id': k, 'difficulty': 0, 'object_type': type}

        return tracks_to_preidct