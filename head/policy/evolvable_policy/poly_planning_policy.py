import gymnasium as gym
import numpy as np

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
        for i in range(len(checkpoints) - 1):
            lane_seg = road_network.graph[checkpoints[i]][checkpoints[i + 1]]
            target_lane_seg = lane_seg[1] if len(lane_seg) > 1 else lane_seg[0]
            lane_seg_center_points = target_lane_seg.get_polyline()
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
