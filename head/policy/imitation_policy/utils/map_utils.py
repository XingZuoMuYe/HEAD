"""
Author: ShuaixiPan
Date: 2025-09-11
Description: Map related utility functions for processing polylines and map data.
"""

import numpy as np
from head.policy.imitation_policy.dataset_builder import common_utils


def get_map_data(config, center_objects, map_infos):
    num_center_objects = center_objects.shape[0]

    def transform_to_center_coordinates(neighboring_polylines):
        neighboring_polylines[:, :, 0:3] -= center_objects[:, None, 0:3]
        neighboring_polylines[:, :, 0:2] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, 0:2],
            angle=-center_objects[:, 6]
        )
        neighboring_polylines[:, :, 3:5] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, 3:5],
            angle=-center_objects[:, 6]
        )

        return neighboring_polylines

    polylines = np.expand_dims(map_infos['all_polylines'].copy(), axis=0).repeat(num_center_objects, axis=0)

    map_polylines = transform_to_center_coordinates(neighboring_polylines=polylines)
    num_of_src_polylines = config['max_num_roads']
    map_infos['polyline_transformed'] = map_polylines

    all_polylines = map_infos['polyline_transformed']
    max_points_per_lane = config.get('max_points_per_lane', 20)
    line_type = config.get('line_type', [])
    map_range = config.get('map_range', None)
    center_offset = config.get('center_offset_of_map', (30.0, 0))
    num_agents = all_polylines.shape[0]
    polyline_list = []
    polyline_mask_list = []

    for k, v in map_infos.items():
        if k == 'all_polylines' or k not in line_type:
            continue
        if len(v) == 0:
            continue
        for polyline_dict in v:
            polyline_index = polyline_dict.get('polyline_index', None)
            polyline_segment = all_polylines[:, polyline_index[0]:polyline_index[1]]
            polyline_segment_x = polyline_segment[:, :, 0] - center_offset[0]
            polyline_segment_y = polyline_segment[:, :, 1] - center_offset[1]
            in_range_mask = (abs(polyline_segment_x) < map_range) * (abs(polyline_segment_y) < map_range)

            segment_index_list = []
            for i in range(polyline_segment.shape[0]):
                segment_index_list.append(common_utils.find_true_segments(in_range_mask[i]))
            max_segments = max([len(x) for x in segment_index_list])

            segment_list = np.zeros([num_agents, max_segments, max_points_per_lane, 7], dtype=np.float32)
            segment_mask_list = np.zeros([num_agents, max_segments, max_points_per_lane], dtype=np.int32)

            for i in range(polyline_segment.shape[0]):
                if in_range_mask[i].sum() == 0:
                    continue
                segment_i = polyline_segment[i]
                segment_index = segment_index_list[i]
                for num, seg_index in enumerate(segment_index):
                    segment = segment_i[seg_index]
                    if segment.shape[0] > max_points_per_lane:
                        segment_list[i, num] = segment[
                            np.linspace(0, segment.shape[0] - 1, max_points_per_lane, dtype=int)]
                        segment_mask_list[i, num] = 1
                    else:
                        segment_list[i, num, :segment.shape[0]] = segment
                        segment_mask_list[i, num, :segment.shape[0]] = 1

            polyline_list.append(segment_list)
            polyline_mask_list.append(segment_mask_list)
    if len(polyline_list) == 0: return np.zeros((num_agents, 0, max_points_per_lane, 7)), np.zeros(
        (num_agents, 0, max_points_per_lane))
    batch_polylines = np.concatenate(polyline_list, axis=1)
    batch_polylines_mask = np.concatenate(polyline_mask_list, axis=1)

    polyline_xy_offsetted = batch_polylines[:, :, :, 0:2] - np.reshape(center_offset, (1, 1, 1, 2))
    polyline_center_dist = np.linalg.norm(polyline_xy_offsetted, axis=-1).sum(-1) / np.clip(
        batch_polylines_mask.sum(axis=-1).astype(float), a_min=1.0, a_max=None)
    polyline_center_dist[batch_polylines_mask.sum(-1) == 0] = 1e10
    topk_idxs = np.argsort(polyline_center_dist, axis=-1)[:, :num_of_src_polylines]

    # Ensure topk_idxs has the correct shape for indexing
    topk_idxs = np.expand_dims(topk_idxs, axis=-1)
    topk_idxs = np.expand_dims(topk_idxs, axis=-1)
    map_polylines = np.take_along_axis(batch_polylines, topk_idxs, axis=1)
    map_polylines_mask = np.take_along_axis(batch_polylines_mask, topk_idxs[..., 0], axis=1)

    # pad map_polylines and map_polylines_mask to num_of_src_polylines
    map_polylines = np.pad(map_polylines,
                           ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
    map_polylines_mask = np.pad(map_polylines_mask,
                                ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

    temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(float)).sum(
        axis=-2)  # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1).astype(float)[:, :, None], a_min=1.0,
                                              a_max=None)  # (num_center_objects, num_polylines, 3)

    xy_pos_pre = map_polylines[:, :, :, 0:3]
    xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
    xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]

    map_types = map_polylines[:, :, :, -1]
    map_polylines = map_polylines[:, :, :, :-1]
    # one-hot encoding for map types, 14 types in total, use 20 for reserved types
    map_types = np.eye(20)[map_types.astype(int)]

    map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)
    map_polylines[map_polylines_mask == 0] = 0

    return map_polylines, map_polylines_mask, map_polylines_center


def get_manually_split_map_data(config, center_objects, map_infos):
    """
    Args:
        center_objects (num_center_objects, 10): [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
        map_infos (dict):
            all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
        center_offset (2):, [offset_x, offset_y]
    Returns:
        map_polylines (num_center_objects, num_topk_polylines, num_points_each_polyline, 9): [x, y, z, dir_x, dir_y, dir_z, global_type, pre_x, pre_y]
        map_polylines_mask (num_center_objects, num_topk_polylines, num_points_each_polyline)
    """
    num_center_objects = center_objects.shape[0]
    center_offset = config.get('center_offset_of_map', (30.0, 0))

    # transform object coordinates by center objects
    def transform_to_center_coordinates(neighboring_polylines, neighboring_polyline_valid_mask):
        neighboring_polylines[:, :, :, 0:3] -= center_objects[:, None, None, 0:3]
        neighboring_polylines[:, :, :, 0:2] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 0:2].reshape(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)
        neighboring_polylines[:, :, :, 3:5] = common_utils.rotate_points_along_z(
            points=neighboring_polylines[:, :, :, 3:5].reshape(num_center_objects, -1, 2),
            angle=-center_objects[:, 6]
        ).reshape(num_center_objects, -1, batch_polylines.shape[1], 2)

        # use pre points to map
        # (num_center_objects, num_polylines, num_points_each_polyline, num_feat)
        xy_pos_pre = neighboring_polylines[:, :, :, 0:3]
        xy_pos_pre = np.roll(xy_pos_pre, shift=1, axis=-2)
        xy_pos_pre[:, :, 0, :] = xy_pos_pre[:, :, 1, :]
        neighboring_polylines = np.concatenate((neighboring_polylines, xy_pos_pre), axis=-1)

        neighboring_polylines[neighboring_polyline_valid_mask == 0] = 0
        return neighboring_polylines, neighboring_polyline_valid_mask

    polylines = map_infos['all_polylines'].copy()
    center_objects = center_objects

    point_dim = polylines.shape[-1]

    point_sampled_interval = config['point_sampled_interval']
    vector_break_dist_thresh = config['vector_break_dist_thresh']
    num_points_each_polyline = config['num_points_each_polyline']

    sampled_points = polylines[::point_sampled_interval]
    sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
    buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]),
                                   axis=-1)  # [ed_x, ed_y, st_x, st_y]
    buffer_points[0, 2:4] = buffer_points[0, 0:2]

    break_idxs = \
        (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4],
                        axis=-1) > vector_break_dist_thresh).nonzero()[0]
    polyline_list = np.array_split(sampled_points, break_idxs, axis=0)
    ret_polylines = []
    ret_polylines_mask = []

    def append_single_polyline(new_polyline):
        cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
        cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
        cur_polyline[:len(new_polyline)] = new_polyline
        cur_valid_mask[:len(new_polyline)] = 1
        ret_polylines.append(cur_polyline)
        ret_polylines_mask.append(cur_valid_mask)

    for k in range(len(polyline_list)):
        if polyline_list[k].__len__() <= 0:
            continue
        for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
            append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

    batch_polylines = np.stack(ret_polylines, axis=0)
    batch_polylines_mask = np.stack(ret_polylines_mask, axis=0)

    # collect a number of closest polylines for each center objects
    num_of_src_polylines = config['max_num_roads']

    if len(batch_polylines) > num_of_src_polylines:
        # Sum along a specific axis and divide by the minimum clamped sum
        polyline_center = np.sum(batch_polylines[:, :, 0:2], axis=1) / np.clip(
            np.sum(batch_polylines_mask, axis=1)[:, None].astype(float), a_min=1.0, a_max=None)
        # Convert the center_offset to a numpy array and repeat it for each object
        center_offset_rot = np.tile(np.array(center_offset, dtype=np.float32)[None, :], (num_center_objects, 1))

        center_offset_rot = common_utils.rotate_points_along_z(
            points=center_offset_rot[:, None, :],
            angle=center_objects[:, 6]
        )

        pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot[:, 0]

        dist = np.linalg.norm(pos_of_map_centers[:, None, :] - polyline_center[None, :, :], axis=-1)

        # Getting the top-k smallest distances and their indices
        topk_idxs = np.argsort(dist, axis=1)[:, :num_of_src_polylines]
        map_polylines = batch_polylines[
            topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
        map_polylines_mask = batch_polylines_mask[
            topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)

    else:
        map_polylines = batch_polylines[None, :, :, :].repeat(num_center_objects, 0)
        map_polylines_mask = batch_polylines_mask[None, :, :].repeat(num_center_objects, 0)

        map_polylines = np.pad(map_polylines,
                               ((0, 0), (0, num_of_src_polylines - map_polylines.shape[1]), (0, 0), (0, 0)))
        map_polylines_mask = np.pad(map_polylines_mask,
                                    ((0, 0), (0, num_of_src_polylines - map_polylines_mask.shape[1]), (0, 0)))

    map_polylines, map_polylines_mask = transform_to_center_coordinates(
        neighboring_polylines=map_polylines,
        neighboring_polyline_valid_mask=map_polylines_mask
    )

    temp_sum = (map_polylines[:, :, :, 0:3] * map_polylines_mask[:, :, :, None].astype(np.float32)).sum(
        axis=-2)  # (num_center_objects, num_polylines, 3)
    map_polylines_center = temp_sum / np.clip(map_polylines_mask.sum(axis=-1)[:, :, np.newaxis].astype(float),
                                              a_min=1.0, a_max=None)

    map_types = map_polylines[:, :, :, 6]
    xy_pos_pre = map_polylines[:, :, :, 7:]
    map_polylines = map_polylines[:, :, :, :6]
    # one-hot encoding for map types, 14 types in total, use 20 for reserved types
    map_types = np.eye(20)[map_types.astype(int)]

    map_polylines = np.concatenate((map_polylines, xy_pos_pre, map_types), axis=-1)

    return map_polylines, map_polylines_mask, map_polylines_center