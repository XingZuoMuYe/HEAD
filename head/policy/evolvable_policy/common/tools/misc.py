#!/usr/bin/env python

import math
import numpy as np
from config import cfg

def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=50):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.x) - 2 - f_idx else len(fpath.x) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


def key_actor(actors_batch, fpath, ego_speed, ego_s, Tc):
    def get_vehicle_speed(vehicle_info):
        vehicle_speed = math.sqrt(
            vehicle_info.actor_vel.x ** 2 + vehicle_info.actor_vel.y ** 2 + vehicle_info.actor_vel.z ** 2)
        return vehicle_speed

    safe_s_ahead = ego_speed * 3.0
    safe_s_follow = ego_speed * 2.0
    # safe_s_ahead = 20.0
    # safe_s_follow = 20.0

    car_length = float(cfg.CARLA.CAR_LENGTH)
    car_width = float(cfg.CARLA.CAR_WIDTH)
    actor_ahead = []
    actor_follow = []
    vehicle_ahead = []
    vehicle_follow = []

    for i, actor in enumerate(actors_batch):
        actor['delta_s'] = actor['Obj_Frenet_state'][0] - ego_s

    for i, actor in enumerate(actors_batch):
        actor_ahead_a = False
        for j in range(len(fpath.s)):
            if abs(actor['Obj_Frenet_state'][0] - fpath.s[j]) <= car_length \
                    and abs(actor['Obj_Frenet_state'][1] - fpath.d[j]) <= car_width \
                    and actor['Obj_Frenet_state'][0] > ego_s:
                actor['safe_acc'] = 2 * (
                        actor['delta_s'] - safe_s_ahead + Tc * (
                            get_vehicle_speed(actor['Vehicle_Info']) - ego_speed)) / (
                                            Tc ** 2)
                actor_ahead.append(actor)
                actor_ahead_a = True
                break
        if not actor_ahead_a:
            for j in range(len(fpath.s)):
                if fpath.s[j] - actor['Obj_Frenet_state'][0] > 0 and abs(
                        actor['Obj_Frenet_state'][1] - fpath.d[j]) <= car_width:
                    actor['safe_acc'] = 2 * (actor['delta_s'] + safe_s_follow - Tc * abs(
                        get_vehicle_speed(actor['Vehicle_Info']) - ego_speed)) / (Tc ** 2)
                    actor['s_path_obj'] = fpath.s[j] - actor['Obj_Frenet_state'][0]
                    actor_follow.append(actor)
                    break

    # vehicle_ahead 关键前车
    if actor_ahead:
        min_delta_s = min(actor_ahead[i]['delta_s'] for i in range(len(actor_ahead)))
        for i, actor in enumerate(actor_ahead):
            if actor['delta_s'] == min_delta_s:
                vehicle_ahead = actor
                if vehicle_ahead['delta_s'] > safe_s_ahead:
                    vehicle_ahead = None
                break
    # vehicle_follow 关键后车
    if actor_follow:
        min_s_path_obj = min(actor_follow[i]['s_path_obj'] for i in range(len(actor_follow)))
        for i, actor in enumerate(actor_follow):
            if actor['s_path_obj'] == min_s_path_obj:
                vehicle_follow = actor
                if vehicle_follow['delta_s'] < -safe_s_follow:
                    vehicle_follow = None
                break

    return vehicle_ahead, vehicle_follow
