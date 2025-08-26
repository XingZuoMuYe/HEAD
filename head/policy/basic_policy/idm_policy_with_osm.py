import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.utils.math import not_zero, wrap_to_pi, norm


class FrontBackObjects:
    """
    用于处理车辆或物体在车道前后位置的检测和管理。
    检查是否有物体在前方、后方或旁边的车道上，并返回对应的物体和距离信息。
    FrontBackObjects 新增 get_find_front_back_objs_id 和 get_find_front_back_objs_id_with_lane_index，
    适配了 数字编号的自定义道路 或 ScenarioNet 的 reference_lane
    """

    def __init__(self, front_ret, back_ret, front_dist, back_dist):
        self.front_objs = front_ret
        self.back_objs = back_ret
        self.front_dist = front_dist
        self.back_dist = back_dist

    def left_lane_exist(self):
        # 不存在的相邻车道就是None
        return True if self.front_dist[0] is not None else False

    def right_lane_exist(self):
        return True if self.front_dist[-1] is not None else False

    def has_front_object(self):
        return True if self.front_objs[1] is not None else False

    def has_back_object(self):
        return True if self.back_objs[1] is not None else False

    def has_left_front_object(self):
        return True if self.front_objs[0] is not None else False

    def has_left_back_object(self):
        return True if self.back_objs[0] is not None else False

    def has_right_front_object(self):
        return True if self.front_objs[-1] is not None else False

    def has_right_back_object(self):
        return True if self.back_objs[-1] is not None else False

    def front_object(self):
        return self.front_objs[1]

    def left_front_object(self):
        return self.front_objs[0]

    def right_front_object(self):
        return self.front_objs[-1]

    def back_object(self):
        return self.back_objs[1]

    def left_back_object(self):
        return self.back_objs[0]

    def right_back_object(self):
        return self.back_objs[-1]

    def left_front_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.front_dist[0]

    def right_front_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.front_dist[-1]

    def front_min_distance(self):
        return self.front_dist[1]

    def left_back_min_distance(self):
        assert self.left_lane_exist(), "left lane doesn't exist"
        return self.back_dist[0]

    def right_back_min_distance(self):
        assert self.right_lane_exist(), "right lane doesn't exist"
        return self.back_dist[-1]

    def back_min_distance(self):
        return self.back_dist[1]

    @classmethod
    def get_find_front_back_objs(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        if ref_lanes is not None:
            assert lane in ref_lanes
        idx = lane.index[-1] if ref_lanes is not None else None
        left_lane = ref_lanes[idx - 1] if ref_lanes is not None and idx > 0 else None
        right_lane = ref_lanes[idx + 1] if ref_lanes is not None and idx + 1 < len(ref_lanes) else None
        lanes = [left_lane, lane, right_lane]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in enumerate(lanes)]

        # 计算车辆和障碍物之间的距离
        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < min_back_long[i]:
                        min_back_long[i] = abs(long)
                        back_ret[i] = obj
                        find_back_in_current_lane[i] = True

                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]
                    if min_front_long[i] > long > 0:
                        min_front_long[i] = long
                        front_ret[i] = obj
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if min_back_long[i] > long:
                        min_back_long[i] = long
                        back_ret[i] = obj

        return cls(front_ret, back_ret, min_front_long, min_back_long)

    @classmethod
    def get_find_front_back_objs_id(cls, objs, lane, position, max_distance, ref_lanes=None):
        """
        只考虑同一个道路下不同车道的障碍物
        因为原先的lane.index是按照区块设置的，以字母等命名
        但是在地图采用数字命名道路时，原先代码会报错故重写
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        return:front_objs(前方所有同向车道障碍物对象), back_objs(后方障碍物对象), front_dist(距离前方障碍物最近距离), back_dist(距离后方障碍物最近距离)
        """
        left_lane = None
        right_lane = None
        if ref_lanes is not None:
            assert lane in ref_lanes
            for l in ref_lanes:
                if l != lane:
                    if l.index in lane.left_lanes and left_lane == None:
                        left_lane = l
                    elif l.index in lane.right_lanes and right_lane == None:
                        right_lane = l
        lanes = [left_lane, lane, right_lane]
        # 列表用于存储各车道前方和后方物体的最小距离，初始值设为 max_distance（表示未检测到物体）
        front_dist = [max_distance if lane is not None else None for lane in lanes]
        back_dist = [max_distance if lane is not None else None for lane in lanes]
        # 用于存储每个车道检测到的前方和后方物体
        front_objs = [None, None, None]
        back_objs = [None, None, None]
        # 用于标记在当前车道中是否已找到前方或后方物体
        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in
                        lanes]  # 存储车辆在各车道坐标系上的当前纵向位置
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in
                     enumerate(lanes)]  # 计算剩余的车道长度

        for i, lane in enumerate(lanes):
            if lane is None:  # 遍历每个车道，对于存在的车道（不为 None），再遍历所有物体 objs
                continue
            for obj in objs:
                # 障碍物在当前车道
                # <-----o----v----|-----
                if obj.lane is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]  # 物体和车辆在当前车道坐标系下的纵向距离
                    if front_dist[i] > long > 0:  # 只考虑前方障碍物
                        front_dist[i] = long
                        front_objs[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < back_dist[i]:
                        back_dist[i] = abs(long)
                        back_objs[i] = obj
                        find_back_in_current_lane[i] = True
                # 若当前车道的end是障碍物所在车道的start
                # <-----o------------------|---------------v----|
                #       |<--obj.position-->|<--left_long-->|
                # 但是会存在这种情况：障碍物在当前车道的下下个车道，但是下个车道的length很短，所以会检测不到障碍物
                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(obj.lane):
                    long = obj.lane.local_coordinates(obj.position)[0] + left_long[i]  # o-v之间的距离
                    if front_dist[i] > long > 0:
                        front_dist[i] = long
                        front_objs[i] = obj
                # 若当前车道的start是障碍物所在车道的end
                # <-----v---|----o----|
                elif not find_back_in_current_lane[i] and obj.lane.is_previous_lane_of(lane):
                    long = obj.lane.length - obj.lane.local_coordinates(obj.position)[0] + current_long[i]
                    if back_dist[i] > long:
                        back_dist[i] = long
                        back_objs[i] = obj

        return cls(front_objs, back_objs, front_dist, back_dist)

    @classmethod
    def get_find_front_back_objs_id_with_lane_index(cls, network, objs, lane, position, max_distance, ref_lanes=None):
        """
        与get_find_front_back_objs_id不同之处在于将obj.lane替换为network.get_lane(obj.lane_index)
        因为此时自车的lane是给定的reference_lane，是一整条历史轨迹且没有index信息的
        所以通过在IdmReplayTrajectoryNavigation中的update_localization每一步根据车辆位置获取当前车道的index来定位
        """
        left_lane = None
        right_lane = None
        if ref_lanes is not None:
            assert lane in ref_lanes
            for l in ref_lanes:
                if l != lane:
                    if l.index in lane.left_lanes and left_lane == None:
                        left_lane = l
                    elif l.index in lane.right_lanes and right_lane == None:
                        right_lane = l
        lanes = [left_lane, lane, right_lane]
        # 列表用于存储各车道前方和后方物体的最小距离，初始值设为 max_distance（表示未检测到物体）
        front_dist = [max_distance if lane is not None else None for lane in lanes]
        back_dist = [max_distance if lane is not None else None for lane in lanes]
        # 用于存储每个车道检测到的前方和后方物体
        front_objs = [None, None, None]
        back_objs = [None, None, None]
        # 用于标记在当前车道中是否已找到前方或后方物体
        find_front_in_current_lane = [False, False, False]
        find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in
                        lanes]  # 存储车辆在各车道坐标系上的当前纵向位置
        left_long = [lane.length - current_long[idx] if lane is not None else None for idx, lane in
                     enumerate(lanes)]  # 计算剩余的车道长度

        for i, lane in enumerate(lanes):
            if lane is None:  # 遍历每个车道，对于存在的车道（不为 None），再遍历所有物体 objs
                continue
            for obj in objs:
                # 障碍物在当前车道
                # <-----o----v----|-----
                if network.get_lane(obj.lane_index) is lane:
                    long = lane.local_coordinates(obj.position)[0] - current_long[i]  # 物体和车辆在当前车道坐标系下的纵向距离
                    if front_dist[i] > long > 0:  # 只考虑前方障碍物
                        front_dist[i] = long
                        front_objs[i] = obj
                        find_front_in_current_lane[i] = True
                    if long < 0 and abs(long) < back_dist[i]:
                        back_dist[i] = abs(long)
                        back_objs[i] = obj
                        find_back_in_current_lane[i] = True
                # 若当前车道的end是障碍物所在车道的start
                # <-----o------------------|---------------v----|
                #       |<--obj.position-->|<--left_long-->|
                # 但是会存在这种情况：障碍物在当前车道的下下个车道，但是下个车道的length很短，所以会检测不到障碍物
                elif not find_front_in_current_lane[i] and lane.is_previous_lane_of(network.get_lane(obj.lane_index)):
                    long = network.get_lane(obj.lane_index).local_coordinates(obj.position)[0] + left_long[
                        i]  # o-v之间的距离
                    if front_dist[i] > long > 0:
                        front_dist[i] = long
                        front_objs[i] = obj
                # 若当前车道的start是障碍物所在车道的end
                # <-----v---|----o----|
                elif not find_back_in_current_lane[i] and network.get_lane(obj.lane_index).is_previous_lane_of(lane):
                    long = network.get_lane(obj.lane_index).length - \
                           network.get_lane(obj.lane_index).local_coordinates(obj.position)[0] + current_long[i]
                    if back_dist[i] > long:
                        back_dist[i] = long
                        back_objs[i] = obj

        return cls(front_objs, back_objs, front_dist, back_dist)

    @classmethod
    def get_find_front_back_objs_single_lane(cls, objs, lane, position, max_distance):
        """
        Find objects in front of/behind the lane and its left lanes/right lanes, return objs, dist.
        If ref_lanes is None, return filter results of this lane
        """
        lanes = [None, lane, None]

        min_front_long = [max_distance if lane is not None else None for lane in lanes]
        min_back_long = [max_distance if lane is not None else None for lane in lanes]

        front_ret = [None, None, None]
        back_ret = [None, None, None]

        find_front_in_current_lane = [False, False, False]
        # find_back_in_current_lane = [False, False, False]

        current_long = [lane.local_coordinates(position)[0] if lane is not None else None for lane in lanes]

        for i, lane in enumerate(lanes):
            if lane is None:
                continue
            for obj in objs:
                _d = obj.position - position
                if norm(_d[0], _d[1]) > max_distance:
                    continue
                if hasattr(obj, "bounding_box") and all([not lane.point_on_lane(p) for p in obj.bounding_box]):
                    continue
                elif not hasattr(obj, "bounding_box") and not lane.point_on_lane(obj.position):
                    continue

                long, _ = lane.local_coordinates(obj.position)
                # if abs(lat) > lane.width / 2:
                #     continue
                long = long - current_long[i]
                if min_front_long[i] > long > 0:
                    min_front_long[i] = long
                    front_ret[i] = obj
                    find_front_in_current_lane[i] = True

        return cls(front_ret, back_ret, min_front_long, min_back_long)


class MyPolicy(BasePolicy):
    """
    IDMPolicy的修改版本，面向大规模背景车流, 基于 OSM地图/车道拓扑 的 IDM
    """

    DEBUG_MARK_COLOR = (219, 3, 252, 255)

    TAU_ACC = 0.6  # [s]
    TAU_HEADING = 0.3  # [s]
    TAU_LATERAL = 0.8  # [s]

    TAU_PURSUIT = 0.5 * TAU_HEADING  # [s]
    KP_A = 1 / TAU_ACC
    KP_HEADING = 1 / TAU_HEADING
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_SPEED = 5  # [m/s]

    DISTANCE_WANTED = 10.0  # 到前车所希望的间隔距离
    TIME_WANTED = 1.5  # [s] 到前车所希望的间隔时间
    DELTA = 10.0  # [] Exponent of the velocity term
    DELTA_RANGE = [3.5, 4.5]  # Range of delta when chosen randomly

    # 侧向策略参数
    LANE_CHANGE_FREQ = 50  # [step]
    LANE_CHANGE_SPEED_INCREASE = 10
    SAFE_LANE_CHANGE_DISTANCE = 15
    MAX_LONG_DIST = 30
    MAX_SPEED = 100  # km/h

    # 正常速度，目标速度
    NORMAL_SPEED = 20  # km/h

    # Creep Speed
    CREEP_SPEED = 5

    # acc factor
    ACC_FACTOR = 1.0
    DEACC_FACTOR = -5

    SAFE_DISTANCE = 10.0  # 安全距离（米）
    EMERGENCY_DISTANCE = 5.0  # 紧急制动距离（米）

    def __init__(self, control_object, random_seed):
        super(MyPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = self.NORMAL_SPEED + self.np_random.uniform(-5, 5)
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        # self.enable_lane_change = False
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, *args, **kwargs):
        # success的作用是判断车辆是否在正确的车道上，首先确定在初始位置在正确道路上，然后可以执行正确动作；
        # 若是发现执行完动作后车辆不在正确道路上，则一直执行沿着当前道路行驶的动作，self.routing_target_lane始终不会更新了
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            if success and self.enable_lane_change:
                # perform lane change due to routing
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                # 采用get_find_front_back_objs_id_with_lane_index
                surrounding_objects = FrontBackObjects.get_find_front_back_objs_id_with_lane_index(
                    self.engine.map_manager.current_map.road_network,
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except:
            # error fallback
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane
            # logging.warning("IDM bug! fall back")
            # print("IDM bug! fall back")

        # if success and self.enable_lane_change:
        #     # perform lane change due to routing
        #     acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
        # else:
        #     # can not find routing target lane
        #     surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
        #         all_objects,
        #         self.routing_target_lane,
        #         self.control_object.position,
        #         max_distance=self.MAX_LONG_DIST
        #     )
        #     acc_front_obj = surrounding_objects.front_object()
        #     acc_front_dist = surrounding_objects.front_min_distance()
        #     steering_target_lane = self.routing_target_lane

        # 以上并不考虑交叉路口中车道交错的情况，应该针对此作出反应，并实现转弯让直行，在steering较大时才减速

        # ============ control by PID and IDM ============
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)

        # 找到前方最近的障碍物,基于前方障碍物计算加速度,会在路口避让，但是可能造成交通堵塞
        closest_front_obj, closest_distance = self._find_closest_front_object(all_objects)
        acc = self._calculate_acceleration(closest_front_obj, closest_distance)

        action = [steering, acc]
        self.action_info["action"] = action
        # print(action)
        return action

    def move_to_next_road(self):
        # 第一步：如果还没有设置路径目标车道，就将当前车道设为目标车道，并检查是否在合法的参考车道列表中。
        current_lanes = self.control_object.navigation.current_ref_lanes
        if self.routing_target_lane is None:
            self.routing_target_lane = self.control_object.lane
            return True if self.routing_target_lane in current_lanes else False

        # 第二步：如果目标车道不在当前参考车道列表中（即第一步初始时车辆所在车道），
        # 说明车辆所在车道和相邻车道是之前车道的下一个车道或有连接关系
        if self.routing_target_lane not in current_lanes:
            for lane in current_lanes:
                if self.routing_target_lane.is_previous_lane_of(lane):
                    self.routing_target_lane = lane
                    return True
                    # lane change for lane num change
            return False
        # 第三步：如果车辆在当前参考车道列表中，但车辆所在车道不是之前的目标车道，说明车辆进行了左右变道
        elif self.control_object.lane in current_lanes and self.routing_target_lane is not self.control_object.lane:
            # lateral routing lane change
            self.routing_target_lane = self.control_object.lane
            self.overtake_timer = self.np_random.randint(0, int(self.LANE_CHANGE_FREQ / 2))
            return True
        # 第四步：如果车辆在当前参考车道列表中，并且车辆所在车道是之前的目标车道，说明车辆没有变道
        else:
            return True

    def steering_control(self, target_lane) -> float:
        # heading control following a lateral distance contro
        ego_vehicle = self.control_object
        long, lat = target_lane.local_coordinates(ego_vehicle.position)
        lane_heading = target_lane.heading_theta_at(long + 1)
        v_heading = ego_vehicle.heading_theta
        steering = self.heading_pid.get_result(-wrap_to_pi(lane_heading - v_heading))
        steering += self.lateral_pid.get_result(-lat)
        return float(steering)

    def acceleration(self, front_obj, dist_to_front) -> float:
        """
        改进的IDM加速度计算，确保能产生负加速度
        """
        ego_vehicle = self.control_object
        ego_target_speed = not_zero(self.target_speed, 0.1)
        ego_speed = max(ego_vehicle.speed_km_h, 0)

        # 自由流加速度项
        free_flow_term = np.power(ego_speed / ego_target_speed, self.DELTA)

        if front_obj and (not self.disable_idm_deceleration):
            # 确保最小距离
            d = max(dist_to_front, 0.1)
            d_star = self.desired_gap(ego_vehicle, front_obj)

            # 交互项
            interaction_term = (d_star / d) ** 2

            # 完整IDM公式：a = a_max * [1 - (v/v_0)^δ - (d*/d)^2]
            acceleration = self.ACC_FACTOR * (1 - free_flow_term - interaction_term)

            # 紧急制动逻辑
            if dist_to_front < 5.0:
                # 当距离很近时，强制制动
                emergency_factor = max(0, (5.0 - dist_to_front) / 5.0)
                max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
                emergency_decel = max_decel * emergency_factor
                acceleration = min(acceleration, emergency_decel)

            # 静止车辆检测
            if hasattr(front_obj, 'speed_km_h') and front_obj.speed_km_h < 2.0:
                if dist_to_front < 10.0 and ego_speed > 2.0:
                    # 计算停车所需的减速度
                    # v² = u² + 2as => a = (v² - u²) / 2s
                    stop_distance = max(dist_to_front - 1.5, 0.5)  # 保留1.5米安全距离
                    ego_speed_ms = ego_speed / 3.6
                    required_decel = -(ego_speed_ms ** 2) / (2 * stop_distance)
                    acceleration = min(acceleration, required_decel)

                    # print(f"Static object detected: required_decel={required_decel:.3f}")
        else:
            # 无前车，使用自由流加速度
            acceleration = self.ACC_FACTOR * (1 - free_flow_term)

        # 限制加速度范围
        max_decel = -abs(self.ACC_FACTOR * self.DEACC_FACTOR)
        max_accel = self.ACC_FACTOR
        acceleration = np.clip(acceleration, max_decel, max_accel)

        return acceleration

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        """
        改进的期望间距计算
        """
        d0 = max(self.DISTANCE_WANTED, 2.0)  # 最小保持2米距离
        tau = self.TIME_WANTED
        ab = abs(self.ACC_FACTOR * self.DEACC_FACTOR)  # 确保为正值

        # 速度差计算
        if hasattr(front_obj, 'velocity_km_h') and hasattr(front_obj, 'speed_km_h'):
            if projected:
                dv = np.dot(ego_vehicle.velocity_km_h - front_obj.velocity_km_h, ego_vehicle.heading)
            else:
                dv = ego_vehicle.speed_km_h - front_obj.speed_km_h
        else:
            # 假设前方物体静止
            dv = ego_vehicle.speed_km_h

        # IDM期望间距公式
        ego_speed_ms = ego_vehicle.speed_km_h / 3.6  # 转换为m/s
        dv_ms = dv / 3.6  # 转换为m/s

        if ab > 0:
            d_star = d0 + max(0, ego_speed_ms * tau + ego_speed_ms * dv_ms / (2 * np.sqrt(ab)))
        else:
            d_star = d0 + ego_speed_ms * tau

        return max(d_star, d0)  # 确保不小于最小距离

    def reset(self):
        self.heading_pid.reset()
        self.lateral_pid.reset()
        self.target_speed = self.NORMAL_SPEED
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)

    def lane_change_policy(self, all_objects):
        """
        适用于ScenarioMap和PGMap的车道变更策略。
        """
        current_lanes = self.control_object.navigation.current_ref_lanes
        surrounding_objects = FrontBackObjects.get_find_front_back_objs_id(
            all_objects, self.routing_target_lane, self.control_object.position, self.MAX_LONG_DIST, current_lanes
        )
        self.available_routing_index_range = [i for i in range(len(current_lanes))]
        next_lanes = self.control_object.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        from metadrive.component.map.scenario_map import ScenarioMap
        from metadrive.component.map.pg_map import PGMap
        # 获取当前车道在列表中的索引
        current_lane_idx = current_lanes.index(self.routing_target_lane)

        # 根据地图类型进行不同的处理
        if isinstance(self.engine.map_manager.current_map, ScenarioMap):
            # 如果下一个导航车道在当前车道的两侧
            if self.routing_target_lane in next_lanes:
                # 如果下一个车道在当前车道的右边
                if next_lanes[0].index in self.routing_target_lane.right_lanes:
                    if self._safe_to_change_lane(surrounding_objects, "right"):
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                            next_lanes[0]
                # 如果下一个车道在当前车道的左边
                if next_lanes[0].index in self.routing_target_lane.left_lanes:
                    if self._safe_to_change_lane(surrounding_objects, "left"):
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                            next_lanes[0]

            # # 如果下一条道路的车道数比当前道路少，则需要进行换道
            # TODO
            # # 1. 强制换道：当车道数减少时
            # if lane_num_diff > 0:
            #     # 计算目标车道范围
            #     if next_lanes and len(next_lanes) > 0:
            #         # 简化逻辑：目标是保持在有效的车道范围内
            #         target_range = list(range(len(next_lanes)))
            #         if current_lane_idx not in target_range:
            #             # 需要换道到有效范围内
            #             if current_lane_idx >= len(next_lanes):
            #                 # 需要向左换道
            #                 target_idx = len(next_lanes) - 1
            #                 if target_idx >= 0 and self._safe_to_change_lane(surrounding_objects, "left"):
            #                     self.target_speed = self.NORMAL_SPEED
            #                     return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
            #                         current_lanes[target_idx]
            #                 else:
            #                     # 不安全，减速等待
            #                     self.target_speed = self.CREEP_SPEED
            #                     return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), \
            #                         self.routing_target_lane

            # 2. 主动换道：超车或避让慢车
            if self._should_attempt_lane_change(surrounding_objects):
                # 检查左侧换道
                if self._safe_to_change_lane(surrounding_objects, "left"):
                    # if self._can_change_left(surrounding_objects, current_lane_idx, current_lanes):
                    left_benefit = self._calculate_lane_change_benefit(surrounding_objects, "left")
                    if left_benefit > self.LANE_CHANGE_SPEED_INCREASE:
                        expect_lane_idx = current_lane_idx - 1
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                            current_lanes[1]

                # 检查右侧换道
                if self._safe_to_change_lane(surrounding_objects, "right"):
                    # if self._can_change_right(surrounding_objects, current_lane_idx, current_lanes):
                    right_benefit = self._calculate_lane_change_benefit(surrounding_objects, "right")
                    if right_benefit > self.LANE_CHANGE_SPEED_INCREASE:
                        expect_lane_idx = current_lane_idx + 1
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                            current_lanes[2]

        if isinstance(self.engine.map_manager.current_map, PGMap):
            # 如果下一条道路的车道数比当前道路少，则需要进行换道
            if lane_num_diff > 0:
                # lane num decreasing happened in left road or right road
                if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                    index_range = [i for i in range(len(next_lanes))]
                else:
                    index_range = [i for i in range(lane_num_diff, len(current_lanes))]
                self.available_routing_index_range = index_range
                if self.routing_target_lane.index[-1] not in index_range:
                    # not on suitable lane do lane change !!!
                    if self.routing_target_lane.index[-1] > index_range[-1]:
                        # change to left
                        if surrounding_objects.left_back_min_distance(
                        ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.left_front_min_distance() < 5:
                            # creep to wait
                            self.target_speed = self.CREEP_SPEED
                            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                            ), self.routing_target_lane
                        else:
                            # it is time to change lane!
                            self.target_speed = self.NORMAL_SPEED
                            return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                                current_lanes[self.routing_target_lane.index[-1] - 1]
                    else:
                        # change to right
                        if surrounding_objects.right_back_min_distance(
                        ) < self.SAFE_LANE_CHANGE_DISTANCE or surrounding_objects.right_front_min_distance() < 5:
                            # unsafe, creep and wait
                            self.target_speed = self.CREEP_SPEED
                            return surrounding_objects.front_object(), surrounding_objects.front_min_distance(
                            ), self.routing_target_lane,
                        else:
                            # change lane
                            self.target_speed = self.NORMAL_SPEED
                            return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                                current_lanes[self.routing_target_lane.index[-1] + 1]

            # lane follow or active change lane/overtake for high driving speed
            if abs(self.control_object.speed_km_h - self.NORMAL_SPEED) > 3 and surrounding_objects.has_front_object(
            ) and abs(surrounding_objects.front_object().speed_km_h -
                      self.NORMAL_SPEED) > 3 and self.overtake_timer > self.LANE_CHANGE_FREQ:
                # may lane change
                right_front_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED \
                    if surrounding_objects.right_lane_exist() and surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
                front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object(
                ) else self.MAX_SPEED
                left_front_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED \
                    if surrounding_objects.left_lane_exist() and surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE and surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE else None
                if left_front_speed is not None and left_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    # left overtake has a high priority
                    expect_lane_idx = current_lanes.index(self.routing_target_lane) - 1
                    if expect_lane_idx in self.available_routing_index_range:
                        return surrounding_objects.left_front_object(), surrounding_objects.left_front_min_distance(), \
                            current_lanes[expect_lane_idx]
                if right_front_speed is not None and right_front_speed - front_speed > self.LANE_CHANGE_SPEED_INCREASE:
                    expect_lane_idx = current_lanes.index(self.routing_target_lane) + 1
                    if expect_lane_idx in self.available_routing_index_range:
                        return surrounding_objects.right_front_object(), surrounding_objects.right_front_min_distance(), \
                            current_lanes[expect_lane_idx]

        # fall back to lane follow
        # self.target_speed = self.NORMAL_SPEED
        self.overtake_timer += 1
        return surrounding_objects.front_object(), surrounding_objects.front_min_distance(), self.routing_target_lane

    def _should_attempt_lane_change(self, surrounding_objects):
        """
        判断是否应该尝试换道:
        1、当前速度与正常速度差异较大 且 前方有车辆且其速度与正常速度差异较大
        2、前车很近且很慢

        """
        # 放宽换道条件
        speed_condition = abs(self.control_object.speed_km_h - self.NORMAL_SPEED) > 2  # 速度太快或太慢时才考虑换道
        front_obj_condition = surrounding_objects.has_front_object()  # bool

        if front_obj_condition:
            front_speed_condition = abs(
                surrounding_objects.front_object().speed_km_h - self.NORMAL_SPEED) > 2  # 前车速度太快或太慢
            timer_condition = self.overtake_timer > self.LANE_CHANGE_FREQ * 0.8  # 减少等待时间

            # 添加距离条件：如果前车很近且很慢，立即尝试换道
            front_distance = surrounding_objects.front_min_distance()
            close_slow_condition = (front_distance < 15.0 and
                                    surrounding_objects.front_object().speed_km_h < self.control_object.speed_km_h - 5)
            return (speed_condition and front_speed_condition and timer_condition) and close_slow_condition

        return False

    def _safe_to_change_lane(self, surrounding_objects, direction):
        """检查换道是否安全，根据前后车的距离和车道存在性"""
        if direction == "left":
            if not surrounding_objects.left_lane_exist():
                return False
            safe_front = surrounding_objects.left_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE
            safe_back = surrounding_objects.left_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE
            return safe_front and safe_back
        elif direction == "right":
            if not surrounding_objects.right_lane_exist():
                return False
            safe_front = surrounding_objects.right_front_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE
            safe_back = surrounding_objects.right_back_min_distance() > self.SAFE_LANE_CHANGE_DISTANCE
            return safe_front and safe_back
        return False

    def _can_change_left(self, surrounding_objects, current_idx, current_lanes):
        """检查是否可以向左换道"""
        return (current_idx > 0 and
                current_idx - 1 in self.available_routing_index_range and
                self._safe_to_change_lane(surrounding_objects, "left"))

    def _can_change_right(self, surrounding_objects, current_idx, current_lanes):
        """检查是否可以向右换道"""
        return (current_idx < len(current_lanes) - 1 and
                current_idx + 1 in self.available_routing_index_range and
                self._safe_to_change_lane(surrounding_objects, "right"))

    def _calculate_lane_change_benefit(self, surrounding_objects, direction):
        """计算换道收益"""
        front_speed = surrounding_objects.front_object().speed_km_h if surrounding_objects.has_front_object() else self.MAX_SPEED

        if direction == "left":
            target_speed = surrounding_objects.left_front_object().speed_km_h if surrounding_objects.has_left_front_object() else self.MAX_SPEED
        elif direction == "right":
            target_speed = surrounding_objects.right_front_object().speed_km_h if surrounding_objects.has_right_front_object() else self.MAX_SPEED
        else:
            return 0

        return target_speed - front_speed

    def _find_closest_front_object(self, all_objects):
        """
        找到前方最近的障碍物
        """
        ego_pos = self.control_object.position
        ego_heading = self.control_object.heading_theta

        # 前进方向向量
        forward_vec = np.array([np.cos(ego_heading), np.sin(ego_heading)])

        closest_obj = None
        min_distance = float('inf')

        for obj in all_objects:
            if hasattr(obj, 'position'):
                # 计算相对位置
                rel_pos = np.array(obj.position[:2]) - np.array(ego_pos[:2])
                distance = np.linalg.norm(rel_pos)

                # 检查是否在前方（投影到前进方向 > 0）
                forward_distance = np.dot(rel_pos, forward_vec)

                if forward_distance > 0.5 and distance < min_distance:
                    # 检查是否在合理的横向范围内（可选）
                    lateral_distance = abs(np.dot(rel_pos, np.array([-forward_vec[1], forward_vec[0]])))
                    if lateral_distance < 3.0:  # 3米横向范围内
                        closest_obj = obj
                        min_distance = distance

        return closest_obj, min_distance

    # def _calculate_acceleration(self, front_obj, distance):
    #     """
    #     基于前方障碍物距离计算加速度，包含静止车辆检测
    #     """
    #     ego_speed = self.control_object.speed_km_h

    #     # 添加完全停车逻辑
    #     if ego_speed < 1.0 and front_obj is not None and distance < self.SAFE_DISTANCE:
    #         return 0.0  # 强制停车需要设置为0加速度

    #     if front_obj is None:
    #         # 没有前方障碍物，正常加速到目标速度
    #         if ego_speed < self.NORMAL_SPEED:
    #             return self.ACC_FACTOR
    #         else:
    #             return 0.0

    #     # 有前方障碍物的情况
    #     if distance < self.EMERGENCY_DISTANCE:
    #         # 紧急制动 - 增强制动力
    #         return -20.0  # 增加制动力到-20
    #     elif distance < self.SAFE_DISTANCE:
    #         # 渐进减速
    #         if ego_speed < 2.0:  # 降低阈值到2.0
    #             return -15.0   # 增强制动力
    #         else:
    #             decel_factor = (self.SAFE_DISTANCE - distance) / (self.SAFE_DISTANCE - self.EMERGENCY_DISTANCE)
    #             acceleration = self.DEACC_FACTOR  * decel_factor * 2.0  # 增强减速效果
    #             return acceleration
    #     else:
    #         # 距离较远时的处理
    #         if ego_speed < self.NORMAL_SPEED:
    #             return self.ACC_FACTOR * 0.5
    #         else:
    #             return 0.0
    def _calculate_acceleration(self, front_obj, distance):
        """
        基于前方障碍物距离计算加速度，包含静止车辆检测和朝向判断
        """
        ego_speed = self.control_object.speed_km_h

        # 添加完全停车逻辑
        if ego_speed < 1.0 and front_obj is not None and distance < self.SAFE_DISTANCE:
            # 检查朝向和静止状态
            if self._should_ignore_static_object(front_obj, distance):
                # 忽略该静止物体，继续前进
                return self.ACC_FACTOR * 0.3  # 缓慢前进
            return 0.0  # 强制停车
        if self._should_ignore_static_object(front_obj, distance):
            # print(f"忽略静止物体: 距离={distance:.2f}m, 朝向差={self._get_heading_difference(front_obj):.1f}°")
            # 忽略该物体，按无障碍物处理
            if ego_speed < self.target_speed:
                return self.ACC_FACTOR * 0.3  # 稍微保守的加速
            else:
                return 0.0

        # 有前方障碍物的情况
        if distance < self.EMERGENCY_DISTANCE:
            # 紧急制动 - 增强制动力
            return -5.0  # 增加制动力到-20
        elif distance < self.SAFE_DISTANCE:
            # 渐进减速
            if ego_speed < 2.0:  # 降低阈值到2.0
                return -0.5  # 增强制动力
            else:
                decel_factor = (self.SAFE_DISTANCE - distance) / (self.SAFE_DISTANCE - self.EMERGENCY_DISTANCE)
                acceleration = self.DEACC_FACTOR * decel_factor * 2.0  # 增强减速效果

                # 静止车辆检测（原有逻辑保留）
                if hasattr(front_obj, 'speed_km_h') and front_obj.speed_km_h < 2.0:
                    if distance < 10.0 and ego_speed > 1.0:
                        stop_distance = max(distance - 1.0, 0.3)
                        ego_speed_ms = ego_speed / 3.6
                        required_decel = -(ego_speed_ms ** 2) / (2 * stop_distance)
                        required_decel = min(required_decel, -10.0)
                        acceleration = min(acceleration, required_decel)

                        # print(f"Static object detected: distance={distance:.2f}, required_decel={required_decel:.3f}")

                return acceleration
        else:
            # 距离较远时的处理
            if ego_speed < self.target_speed:
                return self.ACC_FACTOR * 0.5
            else:
                return 0.0

    def _should_ignore_static_object(self, front_obj, distance):
        """
        判断是否应该忽略前方的静止物体
        条件：
        1. 前方物体静止（速度 < 2 km/h）
        2. 自车静止（速度 < 2 km/h）
        3. 两车朝向角度差较大（> 60度）
        4. 距离在合理范围内（2-8米）
        """
        if front_obj is None:
            return False

        # 检查速度条件
        ego_speed = self.control_object.speed_km_h
        front_speed = getattr(front_obj, 'speed_km_h', 0)

        # 两车都静止
        if ego_speed < 2.0 and front_speed > 2.0:
            return False

        # # 检查距离范围
        # if distance < 2.0 or distance > 8.0:
        #     return False

        # 检查朝向角度差
        heading_diff = self._get_heading_difference(front_obj)
        if heading_diff < 60.0:  # 60度阈值
            return False

        # 所有条件满足，可以忽略该静止物体
        return True

    def _get_heading_difference(self, front_obj):
        """
        计算自车与前方物体的朝向角度差（度）
        """
        if not hasattr(front_obj, 'heading_theta'):
            return 0.0

        ego_heading = self.control_object.heading_theta
        front_heading = front_obj.heading_theta

        # 计算角度差，转换为度数
        angle_diff = abs(ego_heading - front_heading)

        # 处理角度周期性（确保在0-180度范围内）
        if angle_diff > np.pi:
            angle_diff = 2 * np.pi - angle_diff

        # 转换为度数
        return np.degrees(angle_diff)

    def _is_perpendicular_or_opposite(self, front_obj):
        """
        判断前方物体是否与自车垂直或反向
        """
        heading_diff = self._get_heading_difference(front_obj)

        # 垂直：80-100度
        is_perpendicular = 80.0 <= heading_diff <= 100.0

        # 反向：160-180度
        is_opposite = 160.0 <= heading_diff <= 180.0

        return is_perpendicular or is_opposite

    def _classify_static_object(self, front_obj, distance):
        """
        对静止物体进行分类
        """
        if front_obj is None:
            return "none"

        ego_speed = self.control_object.speed_km_h
        front_speed = getattr(front_obj, 'speed_km_h', 0)

        if ego_speed < 2.0 and front_speed < 2.0:
            heading_diff = self._get_heading_difference(front_obj)

            if heading_diff < 30.0:
                return "same_direction"  # 同向静止
            elif 60.0 <= heading_diff <= 120.0:
                return "perpendicular"  # 垂直静止
            elif heading_diff > 150.0:
                return "opposite"  # 反向静止
            else:
                return "angled"  # 斜向静止

        return "moving"  # 有一方在移动


class MyEgoPolicy(MyPolicy):
    """
    面向自车的 基于感知的 IDM
    根据雷达获得的周围物体信息，计算加速度和转向角度的策略
    其他的IDM策略是根据车辆所在车道索引关系判断的
    """

    def __init__(self, control_object, random_seed):
        super(MyEgoPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.target_speed = 20
        self.routing_target_lane = None
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
        self.enable_lane_change = False
        self.disable_idm_deceleration = self.engine.global_config.get("disable_idm_deceleration", False)
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)

    def act(self, *args, **kwargs):
        success = self.move_to_next_road()
        # 获取周围物体
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)

        # 找到前方最近的障碍物
        closest_front_obj, closest_distance = self._find_closest_front_object(all_objects)

        # 基于前方障碍物计算加速度
        acc = self._calculate_acceleration(closest_front_obj, closest_distance)

        # 不转向，保持直行
        steering = self.steering_control(self.routing_target_lane)
        action = [steering, acc]

        self.action_info["action"] = action
        return action