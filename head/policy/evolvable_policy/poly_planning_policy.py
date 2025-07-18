import gymnasium as gym
import numpy as np
from scipy.spatial import cKDTree

from metadrive.component.navigation_module.trajectory_navigation import TrajectoryNavigation
from metadrive.engine.logger import get_logger
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.engine.engine_utils import get_global_config
from metadrive.policy.env_input_policy import EnvInputPolicy
from head.policy.evolvable_policy.common.utils import smooth_curve
from head.policy.evolvable_policy.common.config import cfg, cfg_from_yaml_file
from head.policy.evolvable_policy.common.local_planner.frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from head.policy.evolvable_policy.common.utils import EGO_POSE, calc_cur_s, calc_cur_d, preview_point, match_point, \
    calculate_laterror
import math
import os

logger = get_logger()


class RLPlanningPolicy(EnvInputPolicy):
    def __init__(self, obj, seed):
        # Since control object may change
        super(RLPlanningPolicy, self).__init__(obj, seed)
        self.discrete_action = self.engine.global_config["discrete_action"]
        assert not self.discrete_action, "Must set discrete_action=False for using RL Planning policy"

        self.longitudinal_pid = PIDController(80.0, 0.0, 0.0)
        self.lateral_pid = PIDController(1.0, 0.0, 1.0)
        self.drawer = self.engine.make_point_drawer(scale=1)
        self.preview_distance = 3.0
        current_path = os.path.realpath(__file__)
        root_path = os.path.dirname(os.path.dirname(current_path))
        self.global_csp = None
        self.cfg = cfg_from_yaml_file(root_path + '/evolvable_policy/common/cfgs/config.yaml', cfg)
        self.max_s = int(self.cfg.FRENET.MAX_S)
        self.motionPlanner = MotionPlanner(self.cfg)
        self.ego_pose = None
        self.get_ego_info()
        self.count = 0
        self.f_idx = 0
        self.vehicleController = None
        self.begin_RL_modules()

    def begin_RL_modules(self):

        self.motionPlanner.update_global_route(self.get_global_path())
        self.global_csp = self.motionPlanner.csp

        cur_s, f_idx = calc_cur_s(self.global_csp, self.ego_pose, 0)
        self.f_idx = f_idx
        cur_s_yaw = self.global_csp.calc_yaw(cur_s)
        cur_s_k = self.global_csp.calc_curvature(cur_s)
        cur_d = calc_cur_d(self.ego_pose, self.global_csp, cur_s)
        cur_s_d = self.ego_pose.speed * math.cos(self.ego_pose.yaw - cur_s_yaw)
        cur_d_d = self.ego_pose.speed * math.sin(self.ego_pose.yaw - cur_s_yaw)
        cur_s_dd = self.ego_pose.acc * math.cos(self.ego_pose.yaw - cur_s_yaw) / (1 - cur_d * cur_s_k)

        self.motionPlanner.reset(cur_s, cur_d, cur_s_d, cur_s_dd, cur_d_d, 0, df_n=0, Tf=3, Vf_n=0, optimal_path=False)

    def get_ego_info(self):
        ego_x, ego_y = self.control_object.position
        ego_phi = self.control_object.heading_theta
        ego_v = self.control_object.speed
        ego_a = np.diff(self.control_object.last_velocity)[0] / 0.1
        self.ego_pose = EGO_POSE(ego_x, ego_y, ego_v, ego_phi, ego_a)

    def get_global_path(self):
        road_network = self.control_object.navigation.map.road_network
        checkpoints = self.control_object.navigation.checkpoints
        global_center_points = np.empty((0, 2))
        if isinstance(self.control_object.navigation, TrajectoryNavigation):
            if len(checkpoints) < 4:
                print('checkpoints', len(checkpoints))
                # 如果 checkpoints 少于 3 个点，获取起点和终点
                start_point = checkpoints[0]
                end_point = checkpoints[-1]
                checkpoints = np.linspace(start_point, end_point, num=4)
                print('start_point', start_point, 'end_point',end_point)
            global_center_points = np.array(checkpoints)
            global_path = smooth_curve(global_center_points, dis=0.5)
        else:
            for i in range(len(checkpoints) - 1):
                lane_seg = road_network.graph[checkpoints[i]][checkpoints[i + 1]][1]
                lane_seg_center_points = lane_seg.get_polyline()
                if i != 0:
                    global_center_points = np.vstack([global_center_points, lane_seg_center_points[1:]])
                else:
                    global_center_points = np.vstack([global_center_points, lane_seg_center_points])
            global_path = smooth_curve(global_center_points, dis=4.0)

        return global_path

    def generate_fpath(self, lat_ter, lon_ter, Tf ):

        temp = [self.ego_pose.speed, self.ego_pose.acc]
        ego_state = [self.ego_pose.x, self.ego_pose.y, self.ego_pose.speed, self.ego_pose.acc, self.ego_pose.yaw, temp,
                     self.max_s]
        ego_fstate, f_idx = self.motionPlanner.estimate_frenet_state_new(ego_state, self.f_idx)

        if isinstance(self.control_object.navigation, TrajectoryNavigation):
            lane_num = self.get_current_lane_info()
            print('lane_num',lane_num)
            lane_width = 3.5
        else:
            lane_num = self.engine.global_config.map_config['lane_num']
            lane_width = self.engine.global_config.map_config['lane_width']

        assert -1.0 <= lat_ter <= 1.0, "lat_ter 应该在 [-1, 1] 范围内"
        assert lane_num >= 1
        top_center = lane_width * (lane_num > 1)
        bottom_center = top_center - (lane_num - 1) * lane_width
        t = (1 - lat_ter) / 2
        df_n = (1 - t) * top_center + t * bottom_center

        # 纵向目标速度
        Vf_n = (lon_ter + 1) * 5.0
        # 时间目标
        Tf_n = 0.5 * Tf + 3.0

        fpath_rl = self.motionPlanner.run_step_single_path(ego_fstate, self.f_idx,
                                                           df_n=df_n, Tf=Tf_n,
                                                           Vf_n=Vf_n)
        self.motionPlanner.last_fpath = fpath_rl
        return fpath_rl

    def act(self, agent_id):
        self.f_idx = 0
        action = super(RLPlanningPolicy, self).act(agent_id)
        lat_ter = action[0]
        lon_ter = action[1]
        Tf = action[2]
        self.get_ego_info()

        fpath_rl = self.generate_fpath(lat_ter, lon_ter, Tf)

        self.engine.agents['default_agent'].plan_traj = fpath_rl

        temp = [self.ego_pose.speed, self.ego_pose.acc]
        ego_state = [self.ego_pose.x, self.ego_pose.y, self.ego_pose.speed, self.ego_pose.acc, self.ego_pose.yaw, temp,
                     self.max_s]

        preview_position = preview_point(ego_state, self.preview_distance)
        match_point_control_par = match_point(fpath_rl,
                                              preview_position.update_xvehicle,
                                              preview_position.update_yvehicle)

        self.f_idx = match_point_control_par.find_match_point()

        calculate_laterror_par = calculate_laterror(self.f_idx, fpath_rl,
                                                    preview_position.update_xvehicle,
                                                    preview_position.update_yvehicle)
        control_ed = calculate_laterror_par.laterror

        cmdSpeed = fpath_rl.s_d[min(self.f_idx + 8, len(fpath_rl.s_d)-1)]
        speed_ed = ego_state[2] - cmdSpeed

        steering = max(-1.0, min(self.lateral_pid.get_result(control_ed), 1.0))
        throttle = max(-1.0, min(self.longitudinal_pid.get_result(speed_ed), 1.0))
        # print("steering", steering)
        # print('throttle', throttle)
        action = [steering, throttle]

        self.action_info["action"] = action
        self.count += 1
        return action

    @classmethod
    def get_input_space(cls):
        """
       The Input space is a class attribute
       """
        engine_global_config = get_global_config()

        discrete_action = engine_global_config["discrete_action"]

        if not discrete_action:
            return gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def lane_num_table(self):
        """计算并缓存所有车道的并行车道数量"""
        if hasattr(self, '_lane_num_cache') and getattr(self, '_lane_num_cache_valid', False):
            return self._lane_num_cache

        road_network = self.control_object.navigation.map.road_network
        lanes = road_network.graph
        lane_num_table = {}

        for lane_id in lanes:
            try:
                lane_num_table[lane_id] = self._count_adjacent_lanes(lane_id)
            except Exception as e:
                logger.warning(f"计算车道 {lane_id} 并行数量失败: {e}")
                lane_num_table[lane_id] = 1  # 默认值

        self._lane_num_cache = lane_num_table
        self._lane_num_cache_valid = True
        return lane_num_table

    def find_nearest_lane(self):
        """高效查找最近车道及其中心线"""
        if not hasattr(self, '_lane_kdtree') or self._lane_kdtree is None:
            self._build_lane_centroid_cache()
            if self._lane_kdtree is None:  # 空路网处理
                return None, None, float('inf')

        ego_pos = [self.ego_pose.x, self.ego_pose.y]
        dist, idx = self._lane_kdtree.query(ego_pos)
        lane_id = self._lane_ids[idx]
        return lane_id

    def _build_lane_centroid_cache(self):
        """预计算所有车道的中心点并构建KDTree索引"""
        road_network = self.control_object.navigation.map.road_network
        lanes = road_network.graph
        centroids = []
        lane_ids = []
        self._lane_centroids_map = {}  # 新增：车道ID到中心点的映射

        # 遍历所有车道并计算中心点
        for lane_id, info in lanes.items():
            segments = getattr(info.lane, 'segment_property', [])
            if not segments:
                logger.warning(f"车道 {lane_id} 无分段数据")
                continue

            # 计算几何中心而非简单中点
            points = np.array([seg['start_point'] for seg in segments])
            centroid = np.mean(points, axis=0)

            centroids.append(centroid)
            lane_ids.append(lane_id)
            self._lane_centroids_map[lane_id] = centroid

        if centroids:
            self._lane_kdtree = cKDTree(np.array(centroids))
            self._lane_ids = np.array(lane_ids)
        else:
            self._lane_kdtree = None
            self._lane_ids = np.array([])

        self._lane_cache_valid = True

    def get_current_lane_info(self):
        """获取当前车道信息（带异常处理）"""
        try:
            lane_id = self.find_nearest_lane()  # 修改后只返回lane_id

            # 使用缓存避免全表访问
            if not hasattr(self, '_lane_num_cache'):
                self.lane_num_table()  # 确保缓存存在
            return self._lane_num_cache.get(lane_id, 1)  # 默认1条车道

        except (KeyError, TypeError) as e:  # 更精确的异常捕获
            logger.error(f"获取车道信息失败: {e}")
            return 1  # 默认值
        except Exception as e:
            logger.exception(f"意外错误: {e}")
            return 1

    def _count_adjacent_lanes(self, lane_id):
        """计算当前车道所在道路的并行车道总数（含自身）"""
        if lane_id not in self.control_object.navigation.map.road_network.graph:
            return 1

            # 1. 向左遍历找最左侧车道
        leftmost = lane_id
        left_visited = set()
        while True:
            if leftmost in left_visited:
                break
            left_visited.add(leftmost)

            lane_info = self.control_object.navigation.map.road_network.graph.get(leftmost)
            if not lane_info or not getattr(lane_info, "left_lanes", []):
                break

            # 考虑所有左邻居
            for neighbor in lane_info.left_lanes:
                if neighbor in self.control_object.navigation.map.road_network.graph:
                    leftmost = neighbor
                    break
            else:
                break

        # 2. 向右遍历计数
        count = 0
        current = leftmost
        right_visited = set()
        while True:
            if current in right_visited:
                logger.warning(f"车道 {current} 出现循环引用")
                break
            right_visited.add(current)
            count += 1

            lane_info = self.control_object.navigation.map.road_network.graph.get(current)
            if not lane_info or not getattr(lane_info, "right_lanes", []):
                break

            next_found = False
            for neighbor in lane_info.right_lanes:
                if neighbor in self.control_object.navigation.map.road_network.graph:
                    current = neighbor
                    next_found = True
                    break
            if not next_found:
                break
        return count  # 返回总并行车道数