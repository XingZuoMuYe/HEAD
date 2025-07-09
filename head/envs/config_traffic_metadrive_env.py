from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.base_env import BaseEnv
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable
from metadrive.utils import clip, Config
import numpy as np
import sys
import os
from metadrive.component.map.base_map import BaseMap


class TerminationState:
    SUCCESS = "arrive_dest"
    OUT_OF_ROAD = "out_of_road"
    MAX_STEP = "max_step"
    CRASH = "crash"
    CRASH_VEHICLE = "crash_vehicle"
    CRASH_HUMAN = "crash_human"
    CRASH_OBJECT = "crash_object"
    CRASH_BUILDING = "crash_building"
    CRASH_SIDEWALK = "crash_sidewalk"
    CURRENT_BLOCK = "current_block"
    ENV_SEED = "env_seed"
    IDLE = "idle"
    TaskFail = 'To_far_from_ego_vehicle'


CONF_TRAFFIC_DEFAULT_CONFIG = dict(
    vehicle_config=dict(
        # Vehicle model. Candidates: "s", "m", "l", "xl", "default". random_agent_model makes this config invalid
        vehicle_model="default",
        # If set to True, the vehicle can go backwards with throttle/brake < -1
        enable_reverse=True,
        # Whether to show the box as navigation points
        show_navi_mark=True,
        # Whether to show a box mark at the destination
        show_dest_mark=False,
        # Whether to draw a line from current vehicle position to the designation point
        show_line_to_dest=False,
        # Whether to draw a line from current vehicle position to the next navigation point
        show_line_to_navi_mark=False,
        spawn_longitude=5.0,
        spawn_lateral=3.5),
    scenario_difficulty=1,
    use_pedestrian=False,
    comfort_reward=1.0
)


class StraightConfTraffic(MetaDriveEnv):

    def default_config(self) -> Config:
        config = super(StraightConfTraffic, self).default_config()
        config.update(
            CONF_TRAFFIC_DEFAULT_CONFIG,
            allow_add_new_key=True,
        )

        return config


    def __init__(self, config=None):
        super(StraightConfTraffic, self).__init__(config)

    # 这部分代码是一个类的初始化方法（`__init__`），其中传入了一个可选参数`config`。在初始化方法中，首先对类中的属性
    # `default_config_copy`进行赋值操作：
    #
    # 1.`self.default_config_copy = Config(self.default_config(), unchangeable=True)`  这行代码创建了一个名为`
    # Config`的对象，并将其赋值给类的`default_config_copy`属性。在创建 `Config`对象时，传入了两个参数：第一个是调用
    # `self.default_config()`方法的返回值，这个方法返回默认的配置信息；第二个是一个布尔值`unchangeable = True`，用于标识这个配置信息是否可以被改变。
    # 2. `super(SparseConfTraffic, self).__init__(config)`这行代码调用了父类（`SparseConfTraffic`的父类）的初始化方法，将传入的参数
    # `config`传递给父类的初始化方法以完成对象的初始化工作。通过调用`super()`函数，可以实现在子类中对父类的方法进行调用。

    def reset(self, *args, **kwargs):
        return super(StraightConfTraffic, self).reset(*args, **kwargs)

    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        from head.manager.config_traffic_manager import ConTrafficManager
        from metadrive.manager.pg_map_manager import PGMapManager
        from metadrive.manager.object_manager import TrafficObjectManager
        self.engine.register_manager("map_manager", PGMapManager())
        self.engine.register_manager("traffic_manager", ConTrafficManager())
        if self.config['use_pedestrian']:
            from head.manager.config_pedestrain_manager import Pedestrian_Manager
            self.engine.register_manager("pedestrian_manager", Pedestrian_Manager())
        if abs(self.config["accident_prob"] - 0) > 1e-2:
            self.engine.register_manager("object_manager", TrafficObjectManager())

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, truncate, terminate, info = super(MetaDriveEnv, self).step(actions)
        # info.update({"position": np.array(self.agents['default_agent'].position),
        #              "global_path": self.agents['default_agent'].global_path,
        #              "speed": self.agents['default_agent'].velocity,
        #              "lat_dis": self.engine.agents['default_agent'].cur_d,
        #              "yaw": self.agents['default_agent'].heading
        # })
        return obs, reward, truncate, terminate, info


    def done_function(self, vehicle_id: str):

        vehicle = self.agents[vehicle_id]
        done = False
        max_step = self.config["horizon"] is not None and self.episode_lengths[vehicle_id] >= self.config["horizon"]
        done_info = {
            TerminationState.CRASH_VEHICLE: vehicle.crash_vehicle,
            TerminationState.CRASH_OBJECT: vehicle.crash_object,
            TerminationState.CRASH_BUILDING: vehicle.crash_building,
            TerminationState.CRASH_HUMAN: vehicle.crash_human,
            TerminationState.CRASH_SIDEWALK: vehicle.crash_sidewalk,
            TerminationState.OUT_OF_ROAD: self._is_out_of_road(vehicle),
            TerminationState.SUCCESS: self._is_arrive_destination(vehicle),
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
            TerminationState.TaskFail: self._is_task_failed(vehicle)
            # TerminationState.CURRENT_BLOCK: self.agent.navigation.current_road.block_ID(),
            # crash_vehicle=False, crash_object=False, crash_building=False, out_of_road=False, arrive_dest=False,
        }

        # for compatibility
        # crash almost equals to crashing with vehicles
        done_info[TerminationState.CRASH] = (
                done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
                or done_info[TerminationState.CRASH_BUILDING] or done_info[TerminationState.CRASH_SIDEWALK]
                or done_info[TerminationState.CRASH_HUMAN]
        )

        # determine env return
        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: arrive_dest.".format(self.current_seed),
                extra={"log_once": True}
            )

        if done_info[TerminationState.TaskFail]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: TaskFail.".format(self.current_seed),
                extra={"log_once": True}
            )

        if done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: out_of_road.".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash vehicle ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash object ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_BUILDING]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash building ".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: crash human".format(self.current_seed),
                extra={"log_once": True}
            )
        if done_info[TerminationState.MAX_STEP]:
            # single agent horizon has the same meaning as max_step_per_agent
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(
                "Episode ended! Scenario Index: {} Reason: max step ".format(self.current_seed),
                extra={"log_once": True}
            )
        return done, done_info

    def _is_task_failed(self, vehicle):
        flag = False
        traffic_vehicle_list = self.engine.traffic_manager.traffic_vehicles
        dis = []
        for v in traffic_vehicle_list:
            dis.append((v.position[0] - vehicle.position[0]) ** 2 + (v.position[1] - vehicle.position[1]) ** 2)

        if min(dis) > 120 ** 2:
            flag = True
        return flag

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road

        """
        根据车辆的最近两次动作（横、纵向控制量）来计算舒适度奖励。
        """
        # 获取最近两次的动作（横、纵向控制量）
        prev_action = self.vehicle.last_current_action[-2]  # 上一时刻的动作
        current_action = self.vehicle.last_current_action[-1]  # 当前的动作

        # 确保动作里面所有元素都转换为float
        def convert_to_float(action):
            # 如果是numpy数组，转换为列表并将其元素转换为float
            if isinstance(action, np.ndarray):
                return float(action.item())  # 使用 item() 获取标量值并转换为 float
            elif isinstance(action, (tuple, list)):  # 如果是元组或列表，递归处理
                return [convert_to_float(i) for i in action]
            else:
                return float(action)  # 如果是标量直接转换为 float

        # 将prev_action和current_action里面的所有元素转换为float
        prev_lat, prev_lon = convert_to_float(prev_action)  # 上一时刻的动作（横纵向）
        cur_lat, cur_lon = convert_to_float(current_action)  # 当前时刻的动作（横纵向

        # 计算控制动作变化量
        lat_diff = abs(cur_lat - prev_lat)
        lon_diff = abs(cur_lon - prev_lon)

        # 计算舒适度惩罚
        reward += (-0.05 * lat_diff * self.config["comfort_reward"])

        # print("舒适度奖励：", -0.1 * (lat_diff + lon_diff) * self.config["comfort_reward"])
        # print("奖励：", reward)
        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object or vehicle.crash_human:
            reward = -self.config["crash_object_penalty"]

        step_info["route_completion"] = vehicle.navigation.route_completion

        return reward, step_info
