import time

from scipy.interpolate import splrep, splev
import matplotlib.pyplot as plt

import math
import numpy as np
from collections import namedtuple

import os
from pathlib import Path
import math
import importlib
import itertools
import copy
from typing import Tuple, Dict, Callable, List, Optional, Union, Sequence

# Useful types
Vector = Union[np.ndarray, Sequence[float]]
Matrix = Union[np.ndarray, Sequence[Sequence[float]]]
Interval = Union[np.ndarray,
                 Tuple[Vector, Vector],
                 Tuple[Matrix, Matrix],
                 Tuple[float, float],
                 List[Vector],
                 List[Matrix],
                 List[float]]

BLACK = (0, 0, 0)
LANE_LINE_COLOR = (35 / 255, 35 / 255, 35 / 255)
GREY = (172 / 255, 172 / 255, 172 / 255)


def rgb_normalize(color):
    return color[0] / 255, color[1] / 255, color[2] / 255


def smooth_curve(waypoint, dis=0.2):
    if len(waypoint) < 4:
        # print(f"[WARN] smooth_curve: too few points ({len(waypoint)}), using linear fallback.")
        fx = waypoint[:, 0]
        fy = waypoint[:, 1]
        dx = np.diff(fx)
        dy = np.diff(fy)
        ftheta = np.arctan2(dy, dx)
        ftheta = np.append(ftheta, ftheta[-1])
        fdL = np.sqrt(dx ** 2 + dy ** 2)
        fdL = np.append(fdL, fdL[-1])
        fkappa = np.zeros_like(fx)
        return [fx, fy, ftheta, fkappa]
    i = 1
    a = len(waypoint)
    index = []
    for j in range(a - 1):
        if waypoint[j + 1, 0] == waypoint[j, 0] and waypoint[j + 1, 1] == waypoint[j, 1]:
            index.append(j)
            i += 1

    x = waypoint[:, 0]
    y = waypoint[:, 1]

    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.sqrt(dx ** 2 + dy ** 2)
    s = np.concatenate(([0], np.cumsum(ds)))

    ss = np.arange(0, s[-1], dis)
    xx = splev(ss, splrep(s, x))
    yy = splev(ss, splrep(s, y))

    # 根据插值点计算ftheta和fkappa
    fx = xx
    fy = yy
    num_fx = len(xx)
    ftheta = np.zeros(num_fx)
    fdL = np.zeros(num_fx)
    fkappa = np.zeros(num_fx)

    for i in range(num_fx - 1):
        dx = fx[i + 1] - fx[i]
        dy = fy[i + 1] - fy[i]
        ftheta[i] = np.arctan2(dy, dx)
        fdL[i] = np.sqrt(dx ** 2 + dy ** 2)

    ftheta[num_fx - 1] = ftheta[num_fx - 2]
    fdL[num_fx - 1] = fdL[num_fx - 2]

    for i in range(num_fx - 1):
        fkappa[i] = (ftheta[i + 1] - ftheta[i]) / fdL[i]

    fkappa[num_fx - 1] = fkappa[num_fx - 2]
    return [fx, fy, ftheta, fkappa]


def convert_to_debug_plot(global_path, i):
    color = (0.3, 0.5, 0.8)
    x = global_path[0][i]
    y = global_path[1][i]
    z = 0.85
    points = [(x, y, z)]

    colors = [np.array([color, 1])]
    return points, colors


def plot_vis(path, position, plt_drawer):
    lane_width = 3.5
    fig, ax = plt_drawer
    # Plot the path
    ax.plot(path[0], path[1], marker='o', markersize=4, color='blue', linestyle='-', label='Path')
    # Plot the current position
    ax.plot(position[0], position[1], marker='o', color='red', markersize=4, label='Current Position')
    # Add title and legend
    ax.set_title('Visualization of Path and Current Position')
    ax.legend()
    # Show grid
    ax.grid(True)

    # 绘制边界线
    ax.hlines(3.5 * lane_width, -300, 300, color=BLACK, linewidth=1.5)
    ax.hlines(- 0.5 * lane_width, -300, 300, color=BLACK, linewidth=1.5)
    # 绘制中心线（虚线，红色）
    center_line_x = np.linspace(-300, 300, 10)
    ax.plot(center_line_x, np.full_like(center_line_x, 0.5 * lane_width), color=GREY, linestyle='--')
    ax.plot(center_line_x, np.full_like(center_line_x, 1.5 * lane_width), color=GREY, linestyle='--')
    ax.plot(center_line_x, np.full_like(center_line_x, 2.5 * lane_width), color=GREY, linestyle='--')

    # Set axis limits
    ax.set_xlim(position[0] - 20, position[0] + 100)
    ax.set_ylim(- lane_width - 1, 4 * lane_width + 1)
    ax.set_aspect('equal')
    plt.draw()  # Update the plot
    plt.pause(0.001)  # Add a delay to simulate scrolling effect
    plt.cla()  # Clear the current plot


class EGO_POSE:
    def __init__(self, x, y, v, yaw, a, s=0, s_d=0, d=0):
        self.x = x
        self.y = y
        self.speed = v
        self.yaw = yaw
        self.acc = a
        self.s = s
        self.s_d = s_d
        self.d = d


class Obstacles:
    def __init__(self, perception_data):
        objects_num = perception_data.num
        perception_objects = perception_data.Perceptionobjects
        self.id = []
        self.position_x = []
        self.position_y = []
        self.relative_x = []
        self.relative_y = []
        self.velocity_x = []
        self.velocity_y = []
        self.theta = []
        self.type = []
        for object in perception_objects:
            self.id.append(object.ID)
            self.position_x.append(object.xg)
            self.position_y.append(object.yg)
            self.relative_x.append(object.x)
            self.relative_y.append(object.y)
            self.velocity_x.append(object.v_xg)
            self.velocity_y.append(object.v_yg)
            self.theta.append(object.heading)
            self.type.append(object.type)


def driving_area(lane_data, ego_pose):
    lane_info = [120, 120, 1, 1, 0]
    return lane_info, 3


def calc_max_curv(csp, ego_pose_s):
    min_index = 1
    max_c = 0
    s_index = int(max(0, np.floor(ego_pose_s / 2) - 2))
    for i in range(s_index, len(csp.s)):
        if csp.s[i] - ego_pose_s >= -2:
            min_index = i
            for j in range(min_index, len(csp.s)):
                if csp.s[j] - ego_pose_s <= 30:
                    c_j = calc_curvature(csp, csp.s[j])
                    if abs(c_j) > max_c:
                        max_c = abs(c_j)
                else:
                    break
            return max_c
    return max_c


def get_obs_info(ego_pose, obstacles, csp, obs_index):
    obs_x = obstacles.position_x[obs_index]
    obs_y = obstacles.position_y[obs_index]
    obs_v = math.sqrt(obstacles.velocity_x[obs_index] ** 2 + obstacles.velocity_y[obs_index] ** 2)
    obs_yaw = obstacles.theta[obs_index]
    min_dist = float("inf")
    min_index = 1
    for j in range(len(csp.s)):
        x, y = calc_position(csp, csp.s[j])
        dist = np.sqrt((obs_x - x) ** 2 + (obs_y - y) ** 2)
        if dist < min_dist:
            min_dist = dist
            min_index = j

    rx, ry = calc_position(csp, csp.s[min_index])
    rtheta = calc_yaw(csp, csp.s[min_index])
    rkappa = calc_curvature(csp, csp.s[min_index])
    s, s_d, d, d_d = cartesian_to_frenet3D(csp.s[min_index], rx, ry, rtheta, rkappa, obs_x, obs_y, obs_v, obs_yaw)

    obs_s = s - ego_pose.s
    obs_s_d = s_d

    return obs_s, obs_s_d


def cartesian_to_frenet3D(rs, rx, ry, rtheta, rkappa, x, y, v, theta):
    dx = x - rx
    dy = y - ry
    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = np.sqrt(dx * dx + dy * dy) * np.sign(cross_rd_nd)
    delta_theta = theta - rtheta
    sin_delta_theta = np.sin(delta_theta)
    cos_delta_theta = np.cos(delta_theta)

    one_minus_kappa_r_d = 1 - rkappa * d
    d_d = v * sin_delta_theta

    s = rs + np.dot([dx, dy], [np.cos(rtheta), np.sin(rtheta)])
    s_d = v * cos_delta_theta / one_minus_kappa_r_d

    return s, s_d, d, d_d


def frenet_to_cartesian3D(rs, rx, ry, rtheta, rkappa, rdkappa, s_condition, d_condition):
    if abs(rs - s_condition[0]) >= 1.0e-6:
        print("The reference point s and s_condition[0] don't match")

    a = 0
    theta = 0
    kappa = 0

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    x = rx - sin_theta_r * d_condition[0]
    y = ry + cos_theta_r * d_condition[0]

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    # tan_delta_theta = d_condition[1] / one_minus_kappa_r_d
    # delta_theta = math.atan2(d_condition[1], one_minus_kappa_r_d)
    # cos_delta_theta = math.cos(delta_theta)

    # theta = NormalizeAngle(delta_theta + rtheta)
    # kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]

    # kappa = ((((d_condition[2] + kappa_r_d_prime * tan_delta_theta) * cos_delta_theta**2) /
    #           one_minus_kappa_r_d + rkappa) * cos_delta_theta / one_minus_kappa_r_d)

    d_dot = d_condition[1] * s_condition[1]

    v = math.sqrt(one_minus_kappa_r_d * one_minus_kappa_r_d * s_condition[1] * s_condition[1] + d_dot * d_dot)

    # delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    # a = (s_condition[2] * one_minus_kappa_r_d / cos_delta_theta + s_condition[1] * s_condition[1] /
    #      cos_delta_theta * (d_condition[1] * delta_theta_prime - kappa_r_d_prime))

    return x, y, v, a, theta, kappa


def calc_min_index(X, Y, ego_pose):
    index = 0
    min_dist = np.inf
    for i in range(np.size(X) - 5):
        dist = math.sqrt((X[i] - ego_pose.x) ** 2 + (Y[i] - ego_pose.y) ** 2)
        if dist < min_dist:
            min_dist = dist
            index = i

    return index


def calc_distance(x1, y1, x2, y2):
    y = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return y


# calc initial s
def calc_cur_s(csp, ego_pose, index):
    min_dist = np.inf
    min_index = 0
    increase_count = 0
    dist_tmp = np.inf
    dist_mux = []

    for i in range(index, len(csp.s)):
        global_x, global_y = calc_position(csp, csp.s[i])
        dist = calc_distance(ego_pose.x, ego_pose.y, global_x, global_y)
        dist_mux.append(dist)

    min_index = dist_mux.index(min(dist_mux))
    s_match = csp.s[min_index]
    yaw_match = calc_yaw(csp, s_match)
    x_match, y_match = calc_position(csp, s_match)

    delta_x = ego_pose.x - x_match
    delta_y = ego_pose.y - y_match
    delta_s = np.dot([delta_x, delta_y], [np.cos(yaw_match), np.sin(yaw_match)])

    s = max(0.01, np.round(s_match + delta_s, 2))

    return s, min_index


def calc_cur_fpath_s(fpath, ego_pose, index):
    min_dist = np.inf
    min_index = 0
    increase_count = 0
    dist_tmp = np.inf
    for i in range(index, len(fpath.s)):
        global_x, global_y = fpath.x[i], fpath.y[i]
        dist = calc_distance(ego_pose.x, ego_pose.y, global_x, global_y)
        if dist > dist_tmp:
            increase_count = increase_count + 1
        if increase_count > 5:
            break
        if dist < min_dist:
            min_dist = dist
            min_index = i
        dist_tmp = dist

    s_match = fpath.s[min_index]
    yaw_match = fpath.yaw[min_index]
    x_match, y_match = fpath.x[min_index], fpath.y[min_index]

    delta_x = ego_pose.x - x_match
    delta_y = ego_pose.y - y_match
    delta_s = np.dot([delta_x, delta_y], [np.cos(yaw_match), np.sin(yaw_match)])

    s = np.round(s_match + delta_s, 2)

    return s, min_index


# calc lateral error
def calc_cur_d(ego_pose, csp, cur_s):
    x_ref, y_ref = calc_position(csp, cur_s)
    yaw_ref = calc_yaw(csp, cur_s)

    delta_x = ego_pose.x - x_ref
    delta_y = ego_pose.y - y_ref
    cur_d = np.sqrt(delta_x ** 2 + delta_y ** 2) * np.sign(delta_y * np.cos(yaw_ref) - delta_x * np.sin(yaw_ref))

    return cur_d


# normalize angle
def NormalizeAngle(theta):
    a = (theta + np.pi) % (2 * np.pi)
    if a < 0:
        a += 2 * np.pi
    normalized_ang = a - np.pi
    return normalized_ang


# get everage curv from front global route within 50m
def calc_curv_50(csp, localization):
    min_dist = float("inf")
    min_index = 1
    ave_c = 0
    ego_poses = namedtuple('ego_poses', ['x', 'y'])
    ego_pose = ego_poses(localization.lon, localization.lat)
    for i in range(len(csp.s)):
        global_x, global_y = calc_position(csp, csp.s[i])
        dist = calc_distance(ego_pose.x, ego_pose.y, global_x, global_y)

        if dist < min_dist:
            min_dist = dist
            min_index = i

    i = min_index

    while i < len(csp['s']) - 1:
        if csp.s[i] - csp.s[min_index] > 50:
            break
        ave_c += calc_curvature(csp, csp.s[i])
        i += 1

    if i > min_index:
        ave_c /= (i - min_index + 1)

    return ave_c


def calc_index(ego_d, width):
    index = 0
    if abs(ego_d) <= 2 * width / 3:
        index = 0
    elif ego_d < -2 * width / 3:
        index = -1
    elif ego_d > 2 * width / 3:
        index = 1
    return index


# Spline2D function
def calc_position(csp, s):
    # calc positon
    x = calc(csp.sx, s)
    y = calc(csp.sy, s)
    return x, y


def calc_curvature(csp, s):
    dx = calcd(csp.sx, s)
    ddx = calcdd(csp.sx, s)
    dy = calcd(csp.sy, s)
    ddy = calcdd(csp.sy, s)

    denominator = (dx ** 2 + dy ** 2) ** 1.5
    k = (ddy * dx - ddx * dy) / denominator

    return k


def calc_d_curvature(csp, s):
    dx = calcd(csp.sx, s)
    ddx = calcdd(csp.sx, s)
    dddx = calcddd(csp.sx, s)
    dy = calcd(csp.sy, s)
    ddy = calcdd(csp.sy, s)
    dddy = calcddd(csp.sy, s)

    a = dx * ddy - dy * ddx
    b = dx * dddy - dy * dddx
    c = dx * ddx + dy * ddy
    d = dx * dx + dy * dy

    dk = (b * d - 3.0 * a * c) / (d ** 3)

    return dk


def calc_yaw(csp, s):
    dx = calcd(csp.sx, s)
    dy = calcd(csp.sy, s)
    delta = 0
    if dx <= 0:
        if dy <= 0:
            dx = -dx
            dy = -dy
            delta = -np.pi
        else:
            delta = np.pi

    yaw = np.arctan(dy / dx) + delta

    return yaw


# Spline function
def calc(sp, t):
    if t < sp.x[0]:
        result = sp.a[0]
        # i = search_index(sp, t)
        # dx = t - sp.x[i]
        # result = sp.a[i] + sp.b[i] * dx + sp.c[i] * dx**2 + sp.d[i] * dx**3
    elif t > sp.x[-1]:

        result = sp.a[-1]
    else:
        i = search_index(sp, t)
        dx = t - sp.x[i]
        result = sp.a[i] + sp.b[i] * dx + sp.c[i] * dx ** 2 + sp.d[i] * dx ** 3

    return result


def calcd(sp, t):
    if t < sp.x[0]:
        result = sp.b[0]
        # i = search_index(sp, t)
        # dx = t - sp.x[i]
        # result = sp.b[i] + 2.0 * sp.c[i] * dx + 3.0 * sp.d[i] * dx**2
    elif t > sp.x[-1]:
        result = sp.b[-1]
    else:
        i = search_index(sp, t)
        dx = t - sp.x[i]
        result = sp.b[i] + 2.0 * sp.c[i] * dx + 3.0 * sp.d[i] * dx ** 2

    return result


def calcdd(sp, t):
    if t < sp.x[0]:
        result = None
        # i = search_index(sp, t)
        # dx=t-sp.x[i]
        # result = 2.0 * sp.c[i] + 6.0 * sp.d[i] * dx
    elif t > sp.x[-1]:
        result = None
    else:
        i = search_index(sp, t)
        dx = t - sp.x[i]
        result = 2.0 * sp.c[i] + 6.0 * sp.d[i] * dx

    return result


def calcddd(sp, t):
    if t < sp.x[0]:
        result = None
        # i = search_index(sp, t)
        # result = 6.0 * sp.d[i]
    elif t > sp.x[-1]:
        result = None
    else:
        i = search_index(sp, t)
        result = 6.0 * sp.d[i]

    return result


def search_index(sp, x):
    min_i = 0
    max_i = len(sp.x) - 2
    index = -1  # 初始化为-1，表示未找到
    count = 1.0  # 计数器，用于限制循环次数
    mid_i = 0

    if max_i == -1:
        return 0
    elif min_i == max_i:
        return 0

    # min_i = int(max(0, math.floor(x / 2) - 2))
    # max_i = int(min(min_i + 5, max_i))

    while min_i <= max_i and count <= len(sp.x):  # 添加循环退出条件
        count += 1
        mid_i = (min_i + max_i) // 2

        if x < sp.x[mid_i]:
            max_i = mid_i - 1  # 更新最大索引为 mid_i - 1
        elif x > sp.x[mid_i]:
            min_i = mid_i + 1  # 更新最小索引为 mid_i + 1
        else:
            break
    index = mid_i
    return index


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    '''save rewards and ma_rewards
    '''
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('results saved!')


def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    '''del_empty_dir delete empty folders unders "paths"
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def lamp(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0] + 1e-10)


def closest(lst, K):
    """
    Find closes value in a list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


# 控制时计算用预瞄
class preview_point(object):
    def __init__(self, ego_state, preview_distance):
        self.x_vehicle = ego_state[0]
        self.y_vehicle = ego_state[1]
        self.phi_vehicle = ego_state[4]
        self.preview_distance = preview_distance
        self.update_xvehicle = 0
        self.update_yvehicle = 0
        self.get_preview_vehicle()

    def get_preview_vehicle(self):
        self.update_xvehicle = self.x_vehicle + math.cos(self.phi_vehicle) * self.preview_distance
        self.update_yvehicle = self.y_vehicle + math.sin(self.phi_vehicle) * self.preview_distance
        return np.array([self.update_xvehicle, self.update_yvehicle])


# 匹配规划初始位置时不用预瞄
class match_point(object):
    def __init__(self, fpath, x_vehicle, y_vehicle):
        self.x_reference_vector = fpath.x
        self.y_reference_vector = fpath.y
        self.x_vehicle = x_vehicle
        self.y_vehicle = y_vehicle

        self.d_2 = self.x_reference_vector
        self.d = 0
        self.index = 0
        self.distance = 0
        # self.find_match_point()

    def find_match_point(self):
        self.d_2 = np.multiply(np.array(self.x_reference_vector) - self.x_vehicle,
                               np.array(self.x_reference_vector) - self.x_vehicle) + np.multiply(
            np.array(self.y_reference_vector) - self.y_vehicle, np.array(self.y_reference_vector) - self.y_vehicle)
        self.d = self.d_2 ** 0.5
        self.index = np.argmin(self.d)
        self.distance = self.d[self.index]
        return self.index

    def get_parameters(self):
        if self.d == 0 and self.index == 0:
            print("init_warning")
        return [self.d_2, self.d, self.index, self.distance]


def calculate_edd(Theta_x, Theta_r_MatchPoint, Kappa_r_MatchPoint, ed):
    edd = (1 - Kappa_r_MatchPoint * ed) * math.tan(Theta_x - Theta_r_MatchPoint)
    return edd


class calculate_laterror(object):

    def __init__(self, index, fpath_rl,
                 x_vehicle, y_vehicle):
        self.X_r_MatchPoint = fpath_rl.x[index]
        self.Y_r_MatchPoint = fpath_rl.y[index]
        self.Theta_reference = fpath_rl.yaw[index]
        self.x_vehicle = x_vehicle
        self.y_vehicle = y_vehicle
        self.laterror = self.get_laterroe()

    ###carla 中坐标系是左负右正
    def get_laterroe(self):
        self.laterror = np.dot([self.x_vehicle - self.X_r_MatchPoint, self.y_vehicle - self.Y_r_MatchPoint],
                               [math.cos(math.pi / 2 + self.Theta_reference),
                                math.sin(math.pi / 2 + self.Theta_reference)])
        return self.laterror


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
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

    return min(len(fpath.x) - 2, f_idx + closest_wp_index)


def do_every(duration: float, timer: float) -> bool:
    return duration < timer


def get_class_path(cls: Callable) -> str:
    return cls.__module__ + "." + cls.__qualname__


def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object


def constrain(x: float, a: float, b: float) -> np.ndarray:
    return np.clip(x, a, b)


def not_zero(x: float, eps: float = 1e-2) -> float:
    if abs(x) > eps:
        return x
    elif x >= 0:
        return eps
    else:
        return -eps


def wrap_to_pi(x: float) -> float:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def point_in_rectangle(point: Vector, rect_min: Vector, rect_max: Vector) -> bool:
    """
    Check if a point is inside a rectangle

    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point: np.ndarray, center: np.ndarray, length: float, width: float, angle: float) \
        -> bool:
    """
    Check if a point is inside a rotated rectangle

    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    :return: is the point inside the rectangle
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, (-length / 2, -width / 2), (length / 2, width / 2))


def point_in_ellipse(point: Vector, center: Vector, angle: float, length: float, width: float) -> bool:
    """
    Check if a point is inside an ellipse

    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    :return: is the point inside the ellipse
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1: Tuple[Vector, float, float, float],
                                 rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Do two rotated rectangles intersect?

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    :return: do they?
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def rect_corners(center: np.ndarray, length: float, width: float, angle: float,
                 include_midpoints: bool = False, include_center: bool = False) -> List[np.ndarray]:
    """
    Returns the positions of the corners of a rectangle.
    :param center: the rectangle center
    :param length: the rectangle length
    :param width: the rectangle width
    :param angle: the rectangle angle
    :param include_midpoints: include middle of edges
    :param include_center: include the center of the rect
    :return: a list of positions
    """
    center = np.array(center)
    half_l = np.array([length / 2, 0])
    half_w = np.array([0, width / 2])
    corners = [- half_l - half_w,
               - half_l + half_w,
               + half_l + half_w,
               + half_l - half_w]
    if include_center:
        corners += [[0, 0]]
    if include_midpoints:
        corners += [- half_l, half_l, -half_w, half_w]

    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (rotation @ np.array(corners).T).T + np.tile(center, (len(corners), 1))


def has_corner_inside(rect1: Tuple[Vector, float, float, float],
                      rect2: Tuple[Vector, float, float, float]) -> bool:
    """
    Check if rect1 has a corner inside rect2

    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return any([point_in_rotated_rectangle(p1, *rect2)
                for p1 in rect_corners(*rect1, include_midpoints=True, include_center=True)])


def project_polygon(polygon: Vector, axis: Vector) -> Tuple[float, float]:
    min_p, max_p = None, None
    for p in polygon:
        projected = p.dot(axis)
        if min_p is None or projected < min_p:
            min_p = projected
        if max_p is None or projected > max_p:
            max_p = projected
    return min_p, max_p


def interval_distance(min_a: float, max_a: float, min_b: float, max_b: float):
    """
    Calculate the distance between [minA, maxA] and [minB, maxB]
    The distance will be negative if the intervals overlap
    """
    return min_b - max_a if min_a < min_b else min_a - max_b


def are_polygons_intersecting(a: Vector, b: Vector,
                              displacement_a: Vector, displacement_b: Vector) \
        -> Tuple[bool, bool, Optional[np.ndarray]]:
    """
    Checks if the two polygons are intersecting.

    See https://www.codeproject.com/Articles/15573/2D-Polygon-Collision-Detection

    :param a: polygon A, as a list of [x, y] points
    :param b: polygon B, as a list of [x, y] points
    :param displacement_a: velocity of the polygon A
    :param displacement_b: velocity of the polygon B
    :return: are intersecting, will intersect, translation vector
    """
    intersecting = will_intersect = True
    min_distance = np.inf
    translation, translation_axis = None, None
    for polygon in [a, b]:
        for p1, p2 in zip(polygon, polygon[1:]):
            normal = np.array([-p2[1] + p1[1], p2[0] - p1[0]])
            normal /= np.linalg.norm(normal)
            min_a, max_a = project_polygon(a, normal)
            min_b, max_b = project_polygon(b, normal)

            if interval_distance(min_a, max_a, min_b, max_b) > 0:
                intersecting = False

            velocity_projection = normal.dot(displacement_a - displacement_b)
            if velocity_projection < 0:
                min_a += velocity_projection
            else:
                max_a += velocity_projection

            distance = interval_distance(min_a, max_a, min_b, max_b)
            if distance > 0:
                will_intersect = False
            if not intersecting and not will_intersect:
                break
            if abs(distance) < min_distance:
                min_distance = abs(distance)
                d = a[:-1].mean(axis=0) - b[:-1].mean(axis=0)  # center difference
                translation_axis = normal if d.dot(normal) > 0 else -normal

    if will_intersect:
        translation = min_distance * translation_axis
    return intersecting, will_intersect, translation


def confidence_ellipsoid(data: Dict[str, np.ndarray], lambda_: float = 1e-5, delta: float = 0.1, sigma: float = 0.1,
                         param_bound: float = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute a confidence ellipsoid over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param lambda_: l2 regularization parameter
    :param delta: confidence level
    :param sigma: noise covariance
    :param param_bound: an upper-bound on the parameter norm
    :return: estimated theta, Gramian matrix G_N_lambda, radius beta_N
    """
    phi = np.array(data["features"])
    y = np.array(data["outputs"])
    g_n_lambda = 1 / sigma * np.transpose(phi) @ phi + lambda_ * np.identity(phi.shape[-1])
    theta_n_lambda = np.linalg.inv(g_n_lambda) @ np.transpose(phi) @ y / sigma
    d = theta_n_lambda.shape[0]
    beta_n = np.sqrt(2 * np.log(np.sqrt(np.linalg.det(g_n_lambda) / lambda_ ** d) / delta)) + \
             np.sqrt(lambda_ * d) * param_bound
    return theta_n_lambda, g_n_lambda, beta_n


def confidence_polytope(data: dict, parameter_box: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute a confidence polytope over the parameter theta, where y = theta^T phi

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: estimated theta, polytope vertices, Gramian matrix G_N_lambda, radius beta_N
    """
    param_bound = np.amax(np.abs(parameter_box))
    theta_n_lambda, g_n_lambda, beta_n = confidence_ellipsoid(data, param_bound=param_bound)

    values, pp = np.linalg.eig(g_n_lambda)
    radius_matrix = np.sqrt(beta_n) * np.linalg.inv(pp) @ np.diag(np.sqrt(1 / values))
    h = np.array(list(itertools.product([-1, 1], repeat=theta_n_lambda.shape[0])))
    d_theta = np.array([radius_matrix @ h_k for h_k in h])

    # Clip the parameter and confidence region within the prior parameter box.
    theta_n_lambda = np.clip(theta_n_lambda, parameter_box[0], parameter_box[1])
    for k, _ in enumerate(d_theta):
        d_theta[k] = np.clip(d_theta[k], parameter_box[0] - theta_n_lambda, parameter_box[1] - theta_n_lambda)
    return theta_n_lambda, d_theta, g_n_lambda, beta_n


def is_valid_observation(y: np.ndarray, phi: np.ndarray, theta: np.ndarray, gramian: np.ndarray,
                         beta: float, sigma: float = 0.1) -> bool:
    """
    Check if a new observation (phi, y) is valid according to a confidence ellipsoid on theta.

    :param y: observation
    :param phi: feature
    :param theta: estimated parameter
    :param gramian: Gramian matrix
    :param beta: ellipsoid radius
    :param sigma: noise covariance
    :return: validity of the observation
    """
    y_hat = np.tensordot(theta, phi, axes=[0, 0])
    error = np.linalg.norm(y - y_hat)
    eig_phi, _ = np.linalg.eig(phi.transpose() @ phi)
    eig_g, _ = np.linalg.eig(gramian)
    error_bound = np.sqrt(np.amax(eig_phi) / np.amin(eig_g)) * beta + sigma
    return error < error_bound


def is_consistent_dataset(data: dict, parameter_box: np.ndarray = None) -> bool:
    """
    Check whether a datasets {phi_n, y_n} is consistent

    The last observation should be in the confidence ellipsoid obtained by the N-1 first observations.

    :param data: a dictionary {"features": [phi_0,...,phi_N], "outputs": [y_0,...,y_N]}
    :param parameter_box: a box [theta_min, theta_max]  containing the parameter theta
    :return: consistency of the datasets
    """
    train_set = copy.deepcopy(data)
    y, phi = train_set["outputs"].pop(-1), train_set["features"].pop(-1)
    y, phi = np.array(y)[..., np.newaxis], np.array(phi)[..., np.newaxis]
    if train_set["outputs"] and train_set["features"]:
        theta, _, gramian, beta = confidence_polytope(train_set, parameter_box=parameter_box)
        return is_valid_observation(y, phi, theta, gramian, beta)
    else:
        return True


def near_split(x, num_bins=None, size_bins=None):
    """
    Split a number into several bins with near-even distribution.

    You can either set the number of bins, or their size.
    The sum of bins always equals the total.
    :param x: number to split
    :param num_bins: number of bins
    :param size_bins: size of bins
    :return: list of bin sizes
    """
    if num_bins:
        quotient, remainder = divmod(x, num_bins)
        return [quotient + 1] * remainder + [quotient] * (num_bins - remainder)
    elif size_bins:
        return near_split(x, num_bins=int(np.ceil(x / size_bins)))


def distance_to_circle(center, radius, direction):
    scaling = radius * np.ones((2, 1))
    a = np.linalg.norm(direction / scaling) ** 2
    b = -2 * np.dot(np.transpose(center), direction / np.square(scaling))
    c = np.linalg.norm(center / scaling) ** 2 - 1
    root_inf, root_sup = solve_trinom(a, b, c)
    if root_inf and root_inf > 0:
        distance = root_inf
    elif root_sup and root_sup > 0:
        distance = 0
    else:
        distance = np.infty
    return distance


def distance_to_rect(line: Tuple[np.ndarray, np.ndarray], rect: List[np.ndarray]):
    """
    Compute the intersection between a line segment and a rectangle.

    See https://math.stackexchange.com/a/2788041.
    :param line: a line segment [R, Q]
    :param rect: a rectangle [A, B, C, D]
    :return: the distance between R and the intersection of the segment RQ with the rectangle ABCD
    """
    r, q = line
    a, b, c, d = rect
    u = b - a
    v = d - a
    u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    rqu = (q - r) @ u
    rqv = (q - r) @ v
    interval_1 = [(a - r) @ u / rqu, (b - r) @ u / rqu]
    interval_2 = [(a - r) @ v / rqv, (d - r) @ v / rqv]
    interval_1 = interval_1 if rqu >= 0 else list(reversed(interval_1))
    interval_2 = interval_2 if rqv >= 0 else list(reversed(interval_2))
    if interval_distance(*interval_1, *interval_2) <= 0 \
            and interval_distance(0, 1, *interval_1) <= 0 \
            and interval_distance(0, 1, *interval_2) <= 0:
        return max(interval_1[0], interval_2[0]) * np.linalg.norm(q - r)
    else:
        return np.inf


def solve_trinom(a, b, c):
    delta = b ** 2 - 4 * a * c
    if delta >= 0:
        return (-b - np.sqrt(delta)) / (2 * a), (-b + np.sqrt(delta)) / (2 * a)
    else:
        return None, None


def find_value_in_2DTable(x_table, y_table, x_value):
    if len(x_table) != len(y_table):
        print("x_table and y_table have different lengths")
        return None

    if x_value < x_table[0]:
        return y_table[0]
    elif x_value > x_table[-1]:
        return y_table[-1]
    else:
        for i in range(len(x_table) - 1):
            if x_table[i] <= x_value < x_table[i + 1]:
                k = (y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i])
                y_value = y_table[i] + k * (x_value - x_table[i])
                return y_value