from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.envs.base_env import BaseEnv
from head.renderer.head_renderer import HeadTopDownRenderer
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable
from metadrive.utils import clip, Config
import numpy as np
import sys
import os
from metadrive.component.map.base_map import BaseMap

from metadrive.manager.scenario_curriculum_manager import ScenarioCurriculumManager
from metadrive.manager.scenario_data_manager import ScenarioDataManager
from metadrive.manager.scenario_traffic_manager import ScenarioTrafficManager
from metadrive.utils.math import wrap_to_pi

from metadrive.manager.scenario_map_manager import ScenarioMapManager
from metadrive.manager.scenario_light_manager import ScenarioLightManager
from head.component.map.custom_map_manager import CustomMapManager
from head.component.map.custom_light_manager import CustomLightManager
# from head.manager.config_traffic_manager import GenTrafficManager

# from head.component.navigation.trajectory_navigation import OsmTrajectoryNavigation
from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from head.manager.bev_img_manager.bev_img_manager import BEVRenderer
from metadrive.utils import Config


class RealScenarioEnv(ScenarioEnv):
    @classmethod
    def default_config(cls):
        config = super(RealScenarioEnv, cls).default_config()
        config.update(dict(
            dataset_name=None,
            dataset_candidates=None,
            adversarial=None,
            render_bev=True,
            frame_skip=5,
            frame_stack=3,
            post_stack=5,
            norm_pixel=True,
            resolution_size=200,
            distance=45,
            bev_clip_rgb=True,  # 是否裁剪BEV的RGB值
            bev_onscreen=False  # 是否在屏幕上显示BEV
        ))
        return config

    def __init__(self, config: Optional[Union[Dict, Config]] = None):
        super(RealScenarioEnv, self).__init__(config)
        self.head_renderer = None
        self.bev_renderer = None  # BEV渲染器实例
        self._bev_initialized = False  # 标记BEV是否已初始化
        self.dataset_name = config.get("dataset_name", None)
        self.dataset_candidates = config.get("dataset_candidates", {})
        self.official_datasets = self.dataset_candidates.get("official_datasets", [])
        self.custom_datasets = self.dataset_candidates.get("custom_datasets", [])
        self.adv = config.get("adversarial", None)

    def _post_process_config(self, config):
        """根据 dataset_name 动态修改 config"""
        config = super(RealScenarioEnv, self)._post_process_config(config)
        if config.get("dataset_name") in config.get("dataset_candidates", {}).get("custom_datasets", []):
            config.update(dict(
                map_region_size=2048,
                # even_sample_vehicle_class=True,  deprecated
                vehicle_config=dict(
                    navigation_module= TrajectoryNavigation,    # todo：待改为OsmTrajectoryNavigation
                    lidar=dict(num_lasers=240, distance=50),
                    side_detector=dict(num_lasers=0, distance=50),
                    lane_line_detector=dict(num_lasers=0, distance=20),
                ),
                crash_vehicle_done=True,
                crash_object_done=True,
                crash_human_done=True,
            ))
        else:
            pass
        return config

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, truncate, terminate, info = super(ScenarioEnv, self).step(actions)
        # 生成并添加BEV图像
        if self.config["render_bev"] and self._bev_initialized:
            # 获取自我车辆
            ego_vehicle = self.agents.get('default_agent', None)
            current_scenario_id = self.engine.managers['data_manager'].current_scenario_id
            dataset_name = self.engine.managers['data_manager'].current_scenario_summary['dataset']
            if ego_vehicle:
                # 渲染BEV图像
                bev_image = self.bev_renderer.get_bev_render()
                # 使用 Matplotlib 显示图像
                # plt.imshow(bev_image)
                # plt.axis('off')  # 关闭坐标轴显示
                # plt.show()
                info["bev"] = bev_image
                info["current_scenario_id"] = dataset_name + '_v1.2_' + current_scenario_id

        return obs, reward, truncate, terminate, info

    def _initialize_bev_renderer(self):
        """初始化BEV渲染器"""
        if self.bev_renderer is not None:
            self.bev_renderer.destroy()
        # 创建BEV渲染器实例
        self.bev_renderer = BEVRenderer(
            engine = self.engine,
            resolution=(self.config["resolution_size"], self.config["resolution_size"]),
            max_distance=self.config["distance"],
            onscreen=self.config["bev_onscreen"],
            clip_rgb=self.config["bev_clip_rgb"]
        )
        # 使用当前地图的道路网络重置渲染器
        self.bev_renderer.reset(self.current_map.road_network)
        self._bev_initialized = True

    def reset(self, *args, **kwargs):
        obs = super(RealScenarioEnv, self).reset(*args, **kwargs)
        self.head_renderer = HeadTopDownRenderer(self)
        if self.head_renderer is not None:
            self.head_renderer.reset()
        # 初始化BEV渲染器
        if self.config["render_bev"]:
            self._initialize_bev_renderer()
        return obs

    def setup_engine(self):
        super(ScenarioEnv, self).setup_engine()
        self.engine.register_manager("data_manager", ScenarioDataManager())
        # 判断数据集属于官方还是自建
        if self.dataset_name in self.official_datasets:
            # 官方数据集
            self.engine.register_manager("map_manager", ScenarioMapManager())
            if not self.config["no_light"]:
                self.engine.register_manager("light_manager", ScenarioLightManager())
            if not self.config["no_traffic"]:
                self.engine.register_manager("traffic_manager", ScenarioTrafficManager())

        elif self.dataset_name in self.custom_datasets:
            # 自建数据集
            self.engine.register_manager("map_manager", CustomMapManager())
            if not self.config["no_light"]:
                self.engine.register_manager("light_manager", CustomLightManager())
            if not self.config["no_traffic"]:
                if self.adv:
                    self.engine.register_manager("traffic_manager", ScenarioTrafficManager())  # AdvTrafficManager
                else:
                    self.engine.register_manager("traffic_manager", ScenarioTrafficManager())  # NaturalTrafficManager
        else:
            print("No valid dataset!!!")
            raise ValueError(f"❌ Unknown dataset: {self.dataset_name}")

        self.engine.register_manager("curriculum_manager", ScenarioCurriculumManager())

    def reward_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        step_info = dict()

        long_last = vehicle.navigation.last_longitude
        long_now = vehicle.navigation.current_longitude
        lateral_now = vehicle.navigation.current_lateral

        # ===== Geely 修改奖励 =====
        if self.dataset_name == "geely":
            # 改1：用 lateral_factor 而不是线性 penalty
            if vehicle.lane in vehicle.navigation.current_ref_lanes:
                positive_road = 1
            else:
                current_road = vehicle.navigation.current_road
                positive_road = 1 if not current_road.is_negative_road() else -1

            lateral_factor = clip(
                1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(),
                0.0, 1.0
            )

            reward = 0.0
            reward += 1.0 * (long_now - long_last) * lateral_factor * positive_road   # 改2：前进奖励换算
            reward += 0.1 * (vehicle.speed_km_h / vehicle.max_speed_km_h) * positive_road  # 改3：额外速度奖励

            # 改4：Crash 统一为 -10
            if vehicle.crash_vehicle:
                reward = -10.0
                print("Crash vehicle penalty: ", reward)
            if vehicle.crash_object:
                reward = -10.0
                print("Crash object penalty: ", reward)
            if vehicle.crash_human:
                reward = -10.0
                print("Crash human penalty: ", reward)

        # ===== 原版奖励 =====
        else:
            reward = 0
            reward += self.config["driving_reward"] * (long_now - long_last)

            # 侧偏惩罚
            lateral_factor = abs(lateral_now) / self.config["max_lateral_dist"]
            lateral_penalty = -lateral_factor * self.config["lateral_penalty"]
            reward += lateral_penalty
            # 航向惩罚
            ref_line_heading = vehicle.navigation.current_heading_theta_at_long
            heading_diff = wrap_to_pi(abs(vehicle.heading_theta - ref_line_heading)) / np.pi
            heading_penalty = -heading_diff * self.config["heading_penalty"]
            reward += heading_penalty
            # 转向过大惩罚
            steering = abs(vehicle.current_action[0])
            allowed_steering = (1 / max(vehicle.speed, 1e-2))
            overflowed_steering = min((allowed_steering - steering), 0)
            steering_range_penalty = overflowed_steering * self.config["steering_range_penalty"]
            reward += steering_range_penalty

            if self.config["no_negative_reward"]:
                reward = max(reward, 0)
            # crash penalty
            if vehicle.crash_vehicle:
                reward = -self.config["crash_vehicle_penalty"]
            if vehicle.crash_object:
                reward = -self.config["crash_object_penalty"]
            if vehicle.crash_human:
                reward = -self.config["crash_human_penalty"]
            if vehicle.on_yellow_continuous_line or vehicle.crash_sidewalk or vehicle.on_white_continuous_line:
                reward = -self.config["on_lane_line_penalty"]

            step_info["step_reward_lateral"] = lateral_penalty
            step_info["step_reward_heading"] = heading_penalty
            step_info["step_reward_action_smooth"] = steering_range_penalty

        # ===== 公共统计信息 =====
        step_info["step_reward"] = reward
        step_info["track_length"] = vehicle.navigation.reference_trajectory.length
        step_info["carsize"] = [vehicle.WIDTH, vehicle.LENGTH]
        step_info["route_completion"] = vehicle.navigation.route_completion
        step_info["curriculum_level"] = self.engine.current_level
        step_info["scenario_index"] = self.engine.current_seed
        step_info["num_stored_maps"] = self.engine.map_manager.num_stored_maps
        step_info["scenario_difficulty"] = self.engine.data_manager.current_scenario_difficulty
        step_info["data_coverage"] = self.engine.data_manager.data_coverage
        step_info["curriculum_success"] = self.engine.curriculum_manager.current_success_rate
        step_info["curriculum_route_completion"] = self.engine.curriculum_manager.current_route_completion
        step_info["lateral_dist"] = lateral_now

        return reward, step_info