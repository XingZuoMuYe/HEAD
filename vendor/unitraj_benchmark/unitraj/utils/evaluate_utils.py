import os

import numpy as np
from numpy.ma.extras import average
from typing import List, Tuple
from metadrive.utils import Config
from metadrive.utils.math import compute_angular_velocity
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG


class EvaluateMetrics:
    def __init__(self):
        self.total_speed = []
        self.total_scores = []  # 存储每回合的总得分

        self.round_averages = []  # 存储每回合的平均得分
        self.last_angular_velocity = 0
        self.progress = 0
        self.total_scenario_duration = 0
        self.speed_limits = []
        self.config = Config(BASE_DEFAULT_CONFIG)
        self.previous_info = None
        self.speed_data = []
        self.hist_traj = []
        self.dt = self.config['physics_world_step_size'] * self.config['decision_repeat']

        # 初始化参数
        self.max_miss_rate_threshold = 0.3
        self.max_collisions = 1
        self.max_deviation = 0.3
        self.max_reversed_distance = 6
        self.min_progress = 0.2
        self.score_progress_threshold = 0.1
        self.max_overspeed_value_threshold = 2.23  # m/s, 5mph
        self.comfort_thresholds = {
            'min_longitudinal_acc': -4.05,  # m/s²
            'max_longitudinal_acc': 2.40,  # m/s²
            'max_lateral_acc': 4.89,  # m/s²
            'max_yaw_acc': 1.93,  # rad/s²
            'max_yaw_rate': 0.95,  # rad/s
            'max_longitudinal_impact': 4.13,  # m/s³
            'max_impact_strength': 8.37,  # m/s³
        }
        self.max_speed_m_s = 80 / 3.6
        self.crash_counter = 0
        self.has_crashed = False
        self.crash_threshold = 5  # 连续5帧碰撞才算真实碰撞
        self.no_collision = []
        self.round_scores = {
            'no_collision': [],
            'area_compliance': [],
            'direction_compliance': [],
            'ttc': [],
            'speed_compliance': [],
            'progress': [],
            'comfort': [],

        }

        self.I_round_scores = {
            'I_progress': [],
            'I_comfort': [],
        }

        self.score = {
            'no_collision': [],
            'area_compliance': [],
            'direction_compliance': [],
            'ttc': [],
            'speed_compliance': [],
            'progress': [],
            'comfort': []
        }

        self.I_scores = {
            'I_comfort': [],
        }

    def step(self, info, obs, i_step, env):
        self.speed_data.append(info['velocity'])
        current_yaw_rate = obs[17]  # 259/161的观察是第18位，其他观察需更改
        self.hist_traj.append(env.vehicle.position)
        info["route_completion"] = env.vehicle.navigation.route_completion

        # 计算每一步得分
        self.score['no_collision'].append(self.calculate_no_collision_score(info))
        self.score['area_compliance'].append(self.calculate_area_compliance_score(info))
        self.score['direction_compliance'].append(self.calculate_direction_compliance_score())
        self.score['ttc'].append(self.calculate_ttc_score(env))
        if i_step > 0:
            self.score['comfort'].append(self.calculate_comfort_score(info, current_yaw_rate))
            # self.score['comfort'].append(self.calculate_I_comfort_score(info, current_yaw_rate))
        self.previous_info = info  # 存储上一step的数据

        # 计算I_score的每一步得分
        self.I_scores['I_comfort'].append(self.calculate_I_comfort_score(info, current_yaw_rate))

        if self.crash_counter >= self.crash_threshold:
            self.has_crashed = True

    def update_scores(self, score):
        """
        对字典中的每个键值对进行检查，如果值是列表且包含0，就将该键值设置为0。
        :param score: 字典，包含多个键值对，每个值应该是一个列表
        :return: 更新后的字典
        """
        updated_dict = {}
        for key, value in score.items():
            if isinstance(value, list):  # 确保值是一个列表
                if key == 'ttc':
                    updated_dict[key] = sum(value) / len(value) if value else 1.0
                    continue
                if not value:
                    updated_dict[key] = 0.0
                    continue
                # 检查列表中是否包含0，如果有就将该键值对的值设置为0
                if any(score == 0 for score in value):
                    updated_dict[key] = 0
                elif any(score == 0.5 for score in value):
                    updated_dict[key] = 0.5
                elif any(score == 1 for score in value):
                    updated_dict[key] = 1
                else:
                    if len(value) == 1:
                        updated_dict[key] = float(value[0])  # 只有一个值，直接返回该值
                    else:
                        print("有错误")
            else:
                updated_dict[key] = value
                print('self.score的value存在非列表')
        return updated_dict

    # def calculate_scene_score(self,env):
    #     """计算实际行驶轨迹的场景得分"""
    #     self.score['speed_compliance'].append(self.calculate_speed_compliance_score())
    #     self.score['progress'].append(self.calculate_progress_score(env))
    #     final_score = self.update_scores(self.score)
    #
    #     if final_score['no_collision'] == 0 or final_score['area_compliance'] == 0 or final_score[
    #         'direction_compliance'] == 0 or final_score['progress'] < self.min_progress:
    #         # final_score['progress'] = 0
    #         # final_score['speed_compliance'] = 0
    #         # final_score['comfort'] = 0
    #         # final_score['ttc'] = 0
    #         # final_score['no_collision'] = 0
    #         # final_score['area_compliance'] = 0
    #         # final_score['direction_compliance'] = 0
    #
    #         return 0, final_score
    #
    #     multi = 1
    #     if (final_score['no_collision'] == 0.5 or
    #             final_score['direction_compliance'] == 0.5):
    #         multi = 0.5
    #     weighted_score = multi * (
    #             final_score['direction_compliance'] * 5 +
    #             final_score['ttc'] * 5 +
    #             final_score['speed_compliance'] * 4 +
    #             final_score['progress'] * 5 +
    #             final_score['comfort'] * 2
    #     ) / (5 + 5 + 4 + 5 + 2)
    #     return weighted_score, final_score
    def calculate_scene_score(self,env):
        """计算实际行驶轨迹的场景得分"""
        self.score['speed_compliance'].append(self.calculate_speed_compliance_score())
        self.score['progress'].append(self.calculate_progress_score(env))
        final_score = self.update_scores(self.score)

        if final_score['no_collision'] == 0 or final_score['area_compliance'] == 0 :
            return 0, final_score

        weighted_score = final_score['no_collision'] * final_score['area_compliance'] * (
                final_score['ttc'] * 5 +
                final_score['progress'] * 5 +
                final_score['comfort'] * 2
        ) / (5 + 5 + 2)
        return weighted_score, final_score


    def calculate_no_collision_score(self, info):
        """无责任碰撞得分"""
        if info['crash_vehicle'] or info['crash_human']:
            self.crash_counter += 1
            return 0
        elif info['crash_object'] or info['crash_building'] or info['crash_sidewalk']:
            self.crash_counter += 1
            num_crash = info['crash_object'] + info['crash_building'] + info['crash_sidewalk']
            return 0.5 if num_crash == 1 else 0
        return 1

    def calculate_area_compliance_score(self, info):
        """可驾驶区域合规性得分"""
        return 0 if info['out_of_road'] else 1

    def calculate_direction_compliance_score(self, window = 5, min_move = 0.1) -> float:
        """
        行驶方向合规性得分
        :param path: 最近 → 最远 的路径点
        :param window: 用最近多少步
        :param min_move: 最小有效位移（米）
        :return: 0 / 0.5 / 1
        """
        path = self.hist_traj
        if len(path) < window:
            return 1.0

        recent = np.array(path[-window:])
        vec = np.diff(recent, axis=0)  # (window-1, 2)
        dist = np.linalg.norm(vec, axis=1)

        valid = dist > min_move
        if not valid.any():
            return 1.0

        ref_vec = vec[valid][-1]
        ref_vec /= np.linalg.norm(ref_vec)

        # 实际行驶方向 = 总位移方向（最近 window 首尾）
        actual_vec = recent[-1] - recent[0]
        if np.linalg.norm(actual_vec) == 0:
            return 1.0
        actual_vec /= np.linalg.norm(actual_vec)

        cos_theta = float(np.dot(actual_vec, ref_vec))
        if cos_theta >= 0.2:
            return 1.0
        elif cos_theta >= -0.2:
            return 0.5
        return 0.0

    def calculate_ttc_score(self, env, ttc_max=4.0):
        """碰撞时间得分"""
        ego_pos = env.vehicle.position
        ego_vel = env.vehicle.velocity
        def select_relevant_vehicles(ego_pos, v_ego, env, max_lat=3.5, max_lon=120):
            ego_unit = env.vehicle.heading
            lane = env.vehicle.navigation.reference_trajectory
            spawned_objects = env.engine.traffic_manager.spawned_objects
            obj_pos = []
            obj_vel = []
            for id, value in spawned_objects.items():
                r = np.array([value.position[0] - ego_pos[0], value.position[1] - ego_pos[1]])
                lon = np.dot(r, ego_unit)  # 纵向距离
                lat = np.cross(r, ego_unit)  # 横向距离（2D cross）
                if abs(lon) > max_lon or abs(lat) > max_lat:
                    continue

                # 方向过滤：相对速度在自车航向投影 > −0.5 m/s
                v_rel = value.velocity
                v_rel_ego = v_rel - v_ego
                if np.dot(v_rel_ego, ego_unit) > -0.5:
                    continue

                # 车道横向偏移
                lon_lane, lat_lane = lane.local_coordinates(value.position)
                if abs(lat_lane) > 1.75 * lane.width:
                    continue

                obj_pos.append(value.position)
                obj_vel.append(value.velocity)
            # selected = sorted(selected, key=lambda c: abs(np.dot([car.position[0] - ego_pos[0], car.position[1] - ego_pos[1]], ego_unit)))[:5]
            return obj_pos, obj_vel
        obj_pos, obj_vel = select_relevant_vehicles(ego_pos, ego_vel, env)
        if not obj_pos:
            return 1
        r = np.array([np.array(v) for v in obj_pos[:5]]) - np.array(ego_pos)
        v = np.array([np.array(v) for v in obj_vel[:5]]) - np.array(ego_vel)
        dot = np.sum(v * r, axis=1)
        norm_r = np.linalg.norm(r, axis=1)
        rel_v = dot / (norm_r + 1e-6)
        ttc = np.where(rel_v < 0, -norm_r / rel_v, np.inf)
        ttc_min = np.min(ttc)
        if ttc_min == np.inf:
            return 1.0
        return float(np.clip(ttc_min / ttc_max, 0.0, 1.0))

    def calculate_speed_compliance_score(self):
        """速度限制合规性得分"""
        speed_violation_integral = 0
        for speed, speed_limit in zip(self.speed_data, self.speed_limits):
            if speed > speed_limit:
                speed_violation_integral += (speed - speed_limit) * self.dt
        return max(0,
                   1 - (speed_violation_integral / (self.max_overspeed_value_threshold * self.total_scenario_duration + 1e-5) ))

    def calculate_progress_score(self,env):
        """行驶进度沿专家路径得分"""
        length = env.vehicle.navigation.reference_trajectory.length
        long, _ = env.vehicle.navigation.reference_trajectory.local_coordinates(env.vehicle.position)

        progress = long / length
        self.progress = float(np.clip(progress,0.0,1.0))
        return self.progress

    def calculate_I_comfort_score(self, info, current_yaw_rate):
        """I_score 舒适性得分"""
        velocity = info['velocity']
        acceleration = info['acceleration']
        previous_heading = self.previous_info['steering']
        current_heading = info['steering']
        angular_velocity = compute_angular_velocity(previous_heading, current_heading, self.dt)
        last_acceleration = self.previous_info['acceleration']

        longitudinal_acc = acceleration
        lateral_acc = velocity * angular_velocity
        yaw_acc = (angular_velocity - self.last_angular_velocity) / self.dt
        longitudinal_impact = (acceleration - last_acceleration) / self.dt
        self.last_angular_velocity = angular_velocity

        # 归一化因子（可以根据实际情况调整）
        max_longitudinal_acc = 2.0  # m/s^2
        max_lateral_acc = 1.5  # m/s^2
        max_yaw_acc = 1.0  # rad/s^2
        max_longitudinal_impact = 5.0  # m/s^3

        # 归一化
        norm_longitudinal_acc = abs(longitudinal_acc) / max_longitudinal_acc
        norm_lateral_acc = abs(lateral_acc) / max_lateral_acc
        norm_yaw_acc = abs(yaw_acc) / max_yaw_acc
        norm_longitudinal_impact = abs(longitudinal_impact) / max_longitudinal_impact

        # 加权系数（可以根据实际情况调整）
        w_longitudinal = 0.4
        w_lateral = 0.3
        w_yaw = 0.2
        w_impact = 0.1

        # 计算舒适性得分
        discomfort_score = (w_longitudinal * norm_longitudinal_acc +
                            w_lateral * norm_lateral_acc +
                            w_yaw * norm_yaw_acc +
                            w_impact * norm_longitudinal_impact)

        # 将不适感得分转换为舒适性得分
        comfort_score = max(0, 1 - discomfort_score)
        # print(comfort_score)
        return comfort_score


    def calculate_comfort_score(self, info, current_yaw_rate):
        """舒适性得分"""
        velocity = info['velocity']
        acceleration = info['acceleration']
        previous_heading = self.previous_info['steering']
        current_heading = info['steering']
        angular_velocity = compute_angular_velocity(previous_heading, current_heading, self.dt)
        last_acceleration = self.previous_info['acceleration']

        longitudinal_acc = acceleration
        lateral_acc = velocity * angular_velocity
        yaw_acc = (angular_velocity - self.last_angular_velocity) / self.dt
        longitudinal_impact = (acceleration - last_acceleration) / self.dt
        self.last_angular_velocity = angular_velocity

        if (self.comfort_thresholds['min_longitudinal_acc'] < longitudinal_acc < self.comfort_thresholds[
            'max_longitudinal_acc']
                and self.comfort_thresholds['max_lateral_acc'] > lateral_acc
                and self.comfort_thresholds['max_yaw_acc'] > yaw_acc
                and self.comfort_thresholds['max_yaw_rate'] > current_yaw_rate
                and self.comfort_thresholds['max_longitudinal_impact'] > longitudinal_impact):
            return 1
        return 0

    def reset(self, total_steps, env):
        self.total_scenario_duration = self.dt * total_steps
        self.speed_limits = [self.max_speed_m_s] * total_steps

        scene_score, final_score = self.calculate_scene_score(env)
        mean_speed = sum(self.speed_data) / len(self.speed_data)
        success = int(not self.has_crashed)
        # success = int((not self.has_crashed) and (self.progress > 0.5))

        # 存储每回合的得分到 self.round_scores
        self.round_scores['no_collision'].append(final_score['no_collision'])
        self.round_scores['area_compliance'].append(final_score['area_compliance'])
        self.round_scores['direction_compliance'].append(final_score['direction_compliance'])
        self.round_scores['ttc'].append(final_score['ttc'])
        # print(self.round_scores['ttc'])
        self.round_scores['speed_compliance'].append(final_score['speed_compliance'])
        self.round_scores['progress'].append(final_score['progress'])
        self.round_scores['comfort'].append(final_score['comfort'])
        # 保存每回合的得分

        # 存储每回合的平均得分
        round_average = self.calculate_average(self.score)
        self.round_averages.append(round_average)

        # 每回合清空，但保留总分
        self.total_scores.append(scene_score)  # 保存每回合的得分
        self.total_speed.append(mean_speed)

        self.score = {
            'no_collision': [],
            'area_compliance': [],
            'direction_compliance': [],
            'ttc': [],
            'speed_compliance': [],
            'progress': [],
            'comfort': []
        }
        self.speed_data = []
        self.hist_traj = []
        self.previous_info = []
        self.crash_counter = 0
        self.has_crashed = False

        # 处理I round score
        round_progress = self.progress
        round_comfort =sum( self.I_scores['I_comfort']) / len( self.I_scores['I_comfort']) if len( self.I_scores['I_comfort']) > 0 else 0

        self.I_round_scores['I_progress'].append(round_progress)
        self.I_round_scores['I_comfort'].append(round_comfort)

        return scene_score, final_score, success

    def calculate_average(self, score_dict):
        """计算每个指标的平均得分"""
        averages = {key: sum(value) / len(value) if value else 0 for key, value in score_dict.items()}
        return averages

    def print_metrics(self, info_eval, file_path='./metrics_before' , file_name='eval'):
        """
        打印并保存每个指标的总加权平均值以及总分。
        :param file_path: 输出文件路径，默认为"metrics_output.txt"
        """
        from tabulate import tabulate

        # 计算每个指标的总加权平均值
        mean_reward_all = info_eval['mean_reward_all']
        success_rate = info_eval['success_rate']
        metrics_data = []
        for metric, scores in self.round_scores.items():
            scores_floats = [float(score) for score in scores]  # 将 scores 转换为浮点数列表
            average = sum(scores_floats) / len(scores_floats) if scores_floats else 0  # 计算平均值
            metrics_data.append([metric, average])

        # 计算总分（从 self.total_scores 求平均）
        if hasattr(self, 'total_scores') and self.total_scores:
            total_scores_floats = [float(score) for score in self.total_scores]  # 转换为浮点数列表
            total_average = sum(total_scores_floats) / len(total_scores_floats)  # 计算总分平均值
            metrics_data.append(["Total Score", total_average])  # 添加到表格数据中

        metrics_data.append(["Mean Reward per Step", mean_reward_all])
        metrics_data.append(["Success Rate", success_rate])
        # 定义表格头
        headers = ["Metric", "Score","Mean Reward per Step","Success Rate"]

        # 生成表格（调整表格宽度和格式）
        table1 = tabulate(
            metrics_data,
            headers,
            tablefmt="pretty",  # 使用更美观的格式
            floatfmt=".3f",  # 浮点数显示 3 位小数
            colalign=("left", "right")  # 第一列左对齐，第二列右对齐
        )

        # 打印到控制台
        print("\nMetrics Summary (Weighted Averages):")
        print(table1)
        # final_I_safety = success_rate
        # 生成第二个表格：每个回合的详细得分情况

        # 计算完成得分
        round_details = []
        final_I_progress = sum(self.I_round_scores['I_progress']) / len(self.I_round_scores['I_progress']) if len(
            self.I_round_scores['I_progress']) > 0 else 0


        final_I_comfort = sum(self.I_round_scores['I_comfort']) / len(self.I_round_scores['I_comfort']) if len(
            self.I_round_scores['I_comfort']) > 0 else 0

        # 计算正负偏差
        def calculate_deviation(scores, average):
            positive_deviation = sum(max(score - average, 0) for score in scores) / len(scores) if len(
                scores) > 0 else 0
            negative_deviation = sum(max(average - score, 0) for score in scores) / len(scores) if len(
                scores) > 0 else 0
            return f"{average:.3f} ± {max(positive_deviation, negative_deviation):.3f}"  # 使用 ± 表示偏差范围

        # 计算每个指标的偏差
        safety_dev = f"{success_rate:.3f}"
        progress_dev = calculate_deviation(self.I_round_scores['I_progress'], final_I_progress)
        comfort_dev = calculate_deviation(self.I_round_scores['I_comfort'], final_I_comfort)
        final_mean_speed = sum(self.total_speed) / len(self.total_speed)  # 计算总分平均值
        # 将得分和偏差添加到二维列表中
        round_details.append([safety_dev, progress_dev, comfort_dev, final_mean_speed])

        # 定义第二个表格的表头
        round_headers = ["安全性", "完成效率", "舒适性"," 平均速度"]

        # 生成第二个表格
        table2 = tabulate(
            round_details,
            headers=round_headers,
            tablefmt="pretty",  # 使用更美观的格式
            colalign=("center", "center", "center")  # 所有列居中对齐
        )

        # 打印第二个表格到控制台
        print("\nDetailed Scores per Round (with Deviation):")
        print(table2)

        # 输出到文本文件
        os.makedirs(str(file_path), exist_ok=True)
        with open(str(file_path) + "/" + file_name, "w") as file:
            file.write("Metrics Summary (Weighted Averages):\n")
            file.write(table1)  # 假设 table1 已经定义
            file.write("\n I_Metrics Summary (with Deviation):\n")
            file.write(table2)

        print(f"\nMetrics have been saved to {file_path}")
