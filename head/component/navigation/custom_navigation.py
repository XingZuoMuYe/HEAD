from collections import deque
from metadrive.constants import CamMask

import numpy as np
from panda3d.core import NodePath, Material
from metadrive.engine.logger import get_logger
from metadrive.component.navigation_module.base_navigation import BaseNavigation
from metadrive.engine.asset_loader import AssetLoader
from metadrive.utils.coordinates_shift import panda_vector
from metadrive.utils.math import norm, clip
from metadrive.utils.math import panda_vector
from metadrive.utils.math import wrap_to_pi
from metadrive.utils.pg.utils import ray_localization


class OsmTrajectoryNavigation(BaseNavigation):
    """
    此类适用于类似OSM地图格式的地图数据
    需包含车道之间的连接关系(前后左右)和车道的polyline信息
    以生成的全局路径为索引
    """
    DISCRETE_LEN = 2  # m 2
    CHECK_POINT_INFO_DIM = 5  # 2 指定每个检查点所包含的导航信息的数量或维度
    NUM_WAY_POINT = 2  # 10 作为obs的参数，用作导航信息的检查点个数，会影响navi_info的维度，维度=NUM_WAY_POINT*CHECK_POINT_INFO_DIM
    NUM_NAVI_NODE = 5  # 原代码是使用NUM_WAY_POINT作为显示的检查点个数，但是会影响模型参数个数，故使用新变量
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
            self,
            show_navi_mark: bool = False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=None,
            name=None,
            vehicle_config=None
    ):
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in OsmTrajectoryNavigation")
        super(OsmTrajectoryNavigation, self).__init__(
            show_navi_mark=show_navi_mark,
            show_dest_mark=show_dest_mark,  # 需要在这的False修改掉，不然无法传递参数
            show_line_to_dest=show_line_to_dest,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None  # 导航点模型，就是方块模型
        self._ckpt_vis_models = None  # 检查点可视化模型
        if show_navi_mark and self._show_navi_info and self.vehicle_config['vehicle_model'] == 'default':
            # self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_NAVI_NODE)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    # material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                    # self._navi_point_model.setColor(1, 0, 0, 1) # 红色。若设置了材质，通过节点设置颜色需要在材质设置之后
                self._navi_point_model.instanceTo(model)  # 将导航点模型实例化到当前的可视化模型上，就是在检查点处生成方块模型
                model.reparentTo(self.origin)  # 将可视化模型重新添加到场景中的一个父节点

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)
        self.record_lane_index = '0'

    def reset(self, vehicle):
        current_lane = self.get_checkpoints(vehicle)
        # 若没有下一个车道，则当前车道是最后一个车道，下一个导航车道也是当前车道
        if len(self.checkpoints) == 1:
            self.checkpoints.append(self.checkpoints[0])
        # super(OsmTrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        super(OsmTrajectoryNavigation, self).reset(current_lane)
        self.set_route()

    def get_checkpoints(self, vehicle):
        """
        生成车辆行驶路径保存在self.checkpoints中current_lane
        """
        # # 根据车辆生成位置判断的可能位于的lanes
        # # possible_lanes = ray_localization(vehicle.heading, vehicle.spawn_place, self.engine, use_heading_filter=False) # 如果需要每次reset都在同一个生成点就用spawn_place
        # possible_lanes = ray_localization(vehicle.heading, vehicle.position, self.engine, use_heading_filter=True) # 设置车辆行驶角度过滤条件，不然在车辆转弯角度过大行驶到对向车道时导航会反向
        # possible_lane_indexes = [lane_index for lane, lane_index, dist in possible_lanes]

        # # 如果车辆所在的位置根据车辆朝向判断不出可能的车道（ray_localization存在的不足），则使用车辆位置进行判断，找到那个和车辆朝向最接近的车道
        # if len(possible_lanes) == 0:
        #     possible_lanes = ray_localization(vehicle.heading, vehicle.position, self.engine, use_heading_filter=False)
        #     angle = []
        #     for lane in possible_lanes:
        #         lane_vector = (lane[0].start-lane[0].end)
        #         lane_heading = lane_vector / np.linalg.norm(lane_vector)
        #         angle.append(np.arccos(np.clip(np.dot(vehicle.heading, lane_heading) / (np.linalg.norm(lane_heading) * np.linalg.norm(vehicle.heading)), -1.0, 1.0)))
        #     index = angle.index(min(angle))
        #     possible_lanes = [possible_lanes[index]]
        #     possible_lane_indexes = [lane_index for lane_index in possible_lanes]

        # 首先尝试使用朝向过滤
        possible_lanes = ray_localization(vehicle.heading, vehicle.position, self.engine, use_heading_filter=True)

        # 如果找不到车道，尝试不使用朝向过滤
        if len(possible_lanes) == 0:
            possible_lanes = ray_localization(vehicle.heading, vehicle.position, self.engine, use_heading_filter=False)

            # 如果仍然找不到，使用备用方法：最近车道检测
            if len(possible_lanes) == 0:

                # 使用车道包围车辆位置的索引方法
                closest_lane_index = self.map.road_network.get_closest_lane_index(vehicle.position, True)
                if closest_lane_index is None or len(closest_lane_index) == 0:
                    return None
                for lane in closest_lane_index:
                    closest_lane = self.map.road_network.get_lane(lane)
                    possible_lanes.append((closest_lane, lane, 0))  # 填充0是为了格式对齐，表示距离为0

        # 如果使用朝向过滤找到了多个车道，选择朝向最匹配的
        if len(possible_lanes) > 1:
            angles = []
            for lane, lane_index, dist in possible_lanes:
                # 计算车辆位置处的车道朝向
                try:
                    long, _ = lane.local_coordinates(vehicle.position)
                    lane_heading = lane.heading_theta_at(long)
                    # 计算朝向角度差
                    angle_diff = abs(np.arccos(np.clip(
                        np.dot(vehicle.heading, [np.cos(lane_heading), np.sin(lane_heading)]) /
                        (np.linalg.norm(vehicle.heading) * 1.0), -1.0, 1.0)))
                    angles.append(angle_diff)
                except:
                    angles.append(float('inf'))

            # 选择角度差最小的车道
            best_idx = np.argmin(angles)
            possible_lanes = [possible_lanes[best_idx]]

        # possible_lane_indexes = [lane_index for lane, lane_index in possible_lanes]

        assert len(possible_lanes) > 0, f"无法找到车辆位置 {vehicle.position} 对应的车道"

        self.map.road_network.extract_cycles_from_graph()
        current_lane, current_index = possible_lanes[0][:-1]

        # lanes = [lane_info.lane for lane_info in self.map.road_network.graph.values()]
        from metadrive.utils import get_np_random
        rng = get_np_random()
        self.checkpoints = []
        if self.vehicle_config['vehicle_model'] == 'default':
            # 将自车的位置也限制在更新的图中，只选择位于图结构中的lane
            # current_lane, current_index = [lane for lane in possible_lanes if lane[1] in self.map.road_network.no_dead_lane_graph][0][:-1]
            dest_mode = 1
        elif self.vehicle_config['vehicle_model'] == 'adbv':
            dest_mode = 1
        else:
            # 使用原始图结构，因为周车可以respawn，不需要一直循环驾驶，不
            dest_mode = 2

        # ----------不同的导航方式----------
        if dest_mode == 1:
            # 方法一：给定目的地寻找路径，有时候会很慢
            while len(self.checkpoints) == 0:
                # destination = rng.choice(lanes).index
                self.checkpoints = self.map.road_network.shortest_path(current_lane.index, '232435')  # 232435，231876
                self.checkpoints
        elif dest_mode == 2:
            # 每到一个新路径时才选择下一个路径，更合理
            self.checkpoints.append(current_index)
            last_lane = current_lane
            if last_lane.exit_lanes != [] and last_lane.exit_lanes != None:
                current_index = rng.choice(last_lane.exit_lanes)
                # current_index = last_lane.exit_lanes[0]
                current_lane = self.map.road_network.get_lane(current_index)
                if current_index in self.map.road_network.graph and current_index not in self.checkpoints:
                    self.checkpoints.append(current_index)
                    last_lane = current_lane

        elif dest_mode == 3:
            # 一次性添加5条路径到导航路径中
            self.checkpoints.append(current_index)
            last_lane = current_lane
            while last_lane.exit_lanes != [] and len(self.checkpoints) < 10:
                current_index = rng.choice(last_lane.exit_lanes)
                current_lane = self.map.road_network.get_lane(current_index)
                # current_lane = last_lane.exit_lanes[0] # 只选择第一个出口车道
                if current_index in self.map.road_network.graph and current_index not in self.checkpoints:
                    self.checkpoints.append(current_index)
                    last_lane = current_lane
                else:
                    break

        elif dest_mode == 4:
            pass

        elif dest_mode == 5:
            # 始终以自车下一个checkpoint作为对抗车导航终点。存在问题
            self.checkpoints = self.map.road_network.shortest_path(current_lane.index, next(
                iter(self.engine.agent_manager.spawned_objects.values())).navigation.next_checkpoint_lane_index)

        elif dest_mode == 6:
            # 对抗车的导航路径 = 对抗车到自车的导航路径 + 自车的导航路径
            ego_checkpoints = next(iter(self.engine.agent_manager.spawned_objects.values())).navigation.checkpoints

            self.checkpoints = self.map.road_network.shortest_path(current_lane.index, ego_checkpoints[0])
            if self.checkpoints == None:
                self.checkpoints = ego_checkpoints
            else:
                self.checkpoints += ego_checkpoints

        return current_lane

    @property
    def reference_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    # @property
    # def current_ref_lanes(self):
    #     return [self.reference_trajectory]

    def set_route(self):
        """
        Find a shortest path from start road to end road
        设置一些基础变量
        :param current_lane_index: start road node
        :param destination: end road node or end lane index
        :return: None
        """
        self._target_checkpoints_index = [0, 1]
        self._navi_info.fill(0.0)
        # 从给定的车道索引获取相邻的车道信息[当前车道，左边车道，右边车道]
        self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.current_checkpoint_lane_index)
        self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.next_checkpoint_lane_index)

    # def set_route(self):
    #     self.checkpoints = self.discretize_reference_trajectory()
    #     num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)

    #     self._navi_info.fill(0.0) # 将一个数组或类似的数据结构中的所有元素设置为 0.0
    #     self.next_ref_lanes = None
    #     check_point = self.reference_trajectory.end # 获取参考路径的终点作为最终目标点

    #     if self._dest_node_path is not None:
    #         self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))

    def show_dest_node(self, vehicle_mode):
        """
        在路径终点处显示红色方块
        """
        if self._dest_node_path is not None and vehicle_mode == 'default':
            # 设置终点的节点位置
            self.final_lane = self.map.road_network.get_lane(self.checkpoints[-1])
            ref_lane = self.final_lane
            later_middle = (float(self.get_current_lane_num()) / 2 - 0.5) * ref_lane.width
            # check_point = ref_lane.position(ref_lane.length, later_middle)
            check_point = ref_lane.position(0, later_middle)
            self._dest_node_path.setPos(check_point[0], check_point[1],
                                        self.MARK_HEIGHT)  # check_point[0], check_point[1],self.MARK_HEIGHT
            self._dest_node_path.setColor(1, 0, 0, 0.7)

    def show_target_node(self, check_point, vehicle_mode):
        """
        在指定位置处显示蓝色方块
        for debug
        """
        if self._dest_node_path is not None and vehicle_mode == 'default':
            self._goal_node_path.setPos(check_point[0], check_point[1],
                                        3)  # check_point[0], check_point[1],self.MARK_HEIGHT
            self._goal_node_path.setColor(0, 0, 1, 0.7)

    @property
    def current_checkpoint_lane_index(self):
        return self.checkpoints[self._target_checkpoints_index[0]]

    @property
    def next_checkpoint_lane_index(self):
        return self.checkpoints[self._target_checkpoints_index[1]]

    def discretize_reference_trajectory(self):
        """
        将参考路径以self.DISCRETE_LEN为间隔进行划分作为作为检查点
        """
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):

        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        lane, lane_index = self._update_current_lane(ego_vehicle)  # 根据车辆位置更新当前所在车道
        # 在到达一个新的路段时会触发need_update = True
        need_update = self._update_target_checkpoints(lane_index)  # 检查是否存在下一个检查点，是返回True，False表示位于最后的检查点

        # 让导航路径点一直保持在一定数量
        # while self.vehicle_config['vehicle_model'] == 'default' and len(self.checkpoints) <= 30:
        #     self.checkpoints = self.checkpoints[:-1] + self.map.road_network.random_path(self.checkpoints[-1])
        # 表示车辆已经到达了最后一个检查点，需要重新选择新的目的地进行轨迹规划
        # if self._target_checkpoints_index[0] == self._target_checkpoints_index[1] and (self.vehicle_config['vehicle_model'] == 'adbv' or self.vehicle_config['vehicle_model'] == 'default'):
        #     print("ARRIVE!") # 到达目的地

        # ========== 重新选择新的目的地进行轨迹规划 ==========
        # 1、车辆已经到达了最后一个检查点
        if self._target_checkpoints_index[0] == self._target_checkpoints_index[1]:
            _ = self.get_checkpoints(ego_vehicle)  # 生成行驶路径
            # 若没有下一个车道，则当前车道是最后一个车道，下一个导航车道也是当前车道
            if len(self.checkpoints) == 1:
                self.checkpoints.append(self.checkpoints[0])
            self._target_checkpoints_index = [0, 1]  # 重置当前检查点和下一个检查点的序号
            need_update = True
        # 2、车辆驶出了规划路径范围
        if lane_index not in self.checkpoints:
            # 如果车辆驶出了规划路径范围，且找不到当前所在的车道，则删去车辆
            if self.get_checkpoints(ego_vehicle) is None:
                return True
            # 若没有下一个车道，则当前车道是最后一个车道，下一个导航车道也是当前车道
            if len(self.checkpoints) == 1:
                self.checkpoints.append(self.checkpoints[0])
            self._target_checkpoints_index = [0, 1]  # 重置当前检查点和下一个检查点的序号
            need_update = True

        if need_update:
            # 主要是更新临近的车道关系，从给定的车道索引获取相邻的车道信息
            self.current_ref_lanes = self.map.road_network.get_peer_lanes_from_index(
                self.current_checkpoint_lane_index)  # [当前车道，所有左边车道，所有右边车道]
            self.next_ref_lanes = self.map.road_network.get_peer_lanes_from_index(self.next_checkpoint_lane_index)

        self._navi_info.fill(0.0)

        self.show_dest_node(self.vehicle_config['vehicle_model'])  # 原本是在set_route中设置，每次reset显示，对于需要经常变化终点的需求在此更新

        # -------------------显示的是导航路径中每条道路的end_point-------------------
        # 每次显示self.NUM_NAVI_NODE个导航点，如果最后的导航点不足self.NUM_NAVI_NODE个
        # 那么将多出的self._ckpt_vis_models模型重合放置在最后一个检查点
        # next_idx = self._target_checkpoints_index[0]
        # end_idx = min(next_idx + self.NUM_NAVI_NODE, len(self.checkpoints)) # 保证不超出序号
        # ckpts = self.checkpoints[next_idx:end_idx]
        # diff = self.NUM_NAVI_NODE - len(ckpts)
        # assert diff >= 0, "Number of Navigation points error!"
        # if diff > 0:
        #     ckpts += [self.checkpoints[-1] for _ in range(diff)]
        # self._navi_info.fill(0.0) # 重置导航信息

        # for k, ckpt in enumerate(ckpts[:]):
        #     start = k * self.CHECK_POINT_INFO_DIM
        #     end = (k + 1) * self.CHECK_POINT_INFO_DIM
        #     # self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
        #     # 更新检查点的可视化效果
        #     if self._show_navi_info and self._ckpt_vis_models is not None and self.vehicle_config['vehicle_model'] == 'default':
        #         pos_of_goal = self.map.road_network.get_lane(ckpt).end # 检查点所在路径的end端坐标
        #         lane=self.map.road_network.get_lane(ckpt)
        #         self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
        #         # self._ckpt_vis_models[k].setZ(self._ckpt_vis_models[k].getZ() + 3) # 修改高度
        #         # self._ckpt_vis_models[k].setColor(1, 0, 0, 1) # 修改颜色

        # -------------------显示的是导航路径中车辆面前的几个点-------------------
        if ego_vehicle.class_name == 'DefaultVehicle':
            checkpoints = []
            # 将每一条车道按照5m的间隔进行采样添加到checkpoints中
            for checkpoint in self.checkpoints:
                lane = self.map.road_network.get_lane(checkpoint)
                n = int(lane.length / 5) + 1
                for i in range(n):
                    checkpoints.append(lane.position(i * 5, 0))
            checkpoints = np.array(checkpoints)
            # 根据自车位置计算距离自车最近的检查点进行显示
            distances = np.linalg.norm(checkpoints - ego_vehicle.position, axis=1)
            next_idx = np.argmin(distances) + 1
            end_idx = min(next_idx + self.NUM_NAVI_NODE, len(checkpoints))  # 保证不超出序号
            ckpts = checkpoints[next_idx:end_idx]
            # 保证检查点的个数为self.NUM_NAVI_NODE，不足的用最后一个检查点重复
            diff = self.NUM_NAVI_NODE - len(ckpts)
            assert diff >= 0, "Number of Navigation points error!"
            if diff > 0:
                ckpts = np.append(ckpts, [checkpoints[-1] for _ in range(diff)], axis=0)
            self._navi_info.fill(0.0)  # 重置导航信息

            for k, ckpt in enumerate(ckpts[:]):
                #
                start = k * self.CHECK_POINT_INFO_DIM
                end = (k + 1) * self.CHECK_POINT_INFO_DIM
                self._navi_info[start:end] = self._get_info_for_checkpoint(ckpt, ego_vehicle, self.current_lane)
                # 更新检查点的可视化效果
                if self._show_navi_info and self._ckpt_vis_models is not None and self.vehicle_config[
                    'vehicle_model'] == 'default':
                    pos_of_goal = ckpt
                    self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                    # self._ckpt_vis_models[k].setZ(self._ckpt_vis_models[k].getZ() + 3) # 修改高度
                    # self._ckpt_vis_models[k].setColor(1, 0, 0, 1) # 修改颜色
            self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
            self._navi_info[end + 1] = clip(
                (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
            )

    def _get_info_for_checkpoint(self, ref_lane, ego_vehicle, current_lane):
        """
        检查点位置映射到目标车辆的坐标系中，其中 +x 是车辆的航向，+y 是车辆的右侧。
        """
        navi_information = []
        check_point = ref_lane.end
        dir_vec = check_point - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        # 将dir_vec向量投影到目标车辆的坐标系中，沿着行驶方向为y轴，右侧为x轴
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )

        # Dim 1:将检查点在目标车辆的航向方向上的相对位置添加到导航信息中
        navi_information.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2:将检查点在目标车辆的右侧方向上的横向相对位置添加到导航信息中
        navi_information.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        """当检查点维度信息需设为5时才增加以下代码处理"""
        # 尝试将当前车道的信息包含到导航信息中，如果当前车道是环形车道，那么将环形车道的信息包含到导航信息中，否则为0
        bendradius = 0.0
        dir = 0.0
        angle = 0.0
        from metadrive.component.pg_space import Parameter, BlockParameterSpace
        from head.component.map.lane_utils import count_adjacent_lanes
        def get_current_lane_num(self) -> float:
            try:
                lane = current_lane
                if not lane:
                    return 1
                lane_id = lane.lane_index
                road_network = self.engine.map_manager.mao.road_network
                return count_adjacent_lanes(lane_id, road_network)
            except Exception as e:
                get_logger().error(f"Error in counting adjacent lanes: {e}")
                return 1

        if not current_lane.is_straight:
            bendradius = current_lane.get_radius() / (
                    60 + get_current_lane_num(self) * 3.5
            )
            dir = current_lane.get_direction()
            angle = current_lane.get_angle()

        # Dim 3: The bending radius of current lane
        navi_information.append(clip(bendradius, 0.0, 1.0))

        # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))

        # Dim 5: The angular difference between the heading in lane ending position and
        # the heading in lane starting position
        navi_information.append(
            clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        )
        return navi_information

    def _update_current_lane(self, ego_vehicle):
        """
        具体逻辑就是根据车辆的位置，计算车辆当前所在的车道，因为在一个step中车辆的位置会发生变化。
        但如果根据目前位置无法确定车道，那么就根据车辆之前的位置来确定车道。
        ego_vehicle.lane调用的还是self.navigation.current_lane来获得车道信息，
        所以在没有执行self.current_lane = lane这一句时还保留了之前的车道信息。
        不论通过哪种方式获得的lane都在self.current_lane = lane这一句中更新了当前车道。
        """
        lane, lane_index, on_lane = self._get_current_lane(ego_vehicle)
        ego_vehicle.on_lane = on_lane
        if on_lane == False:
            pass
        if lane is None:  # 如果未找到当前车道，则从 ego_vehicle 获取之前的车道和车道索引
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            if self.FORCE_CALCULATE:
                # 若为 True，则根据车辆当前位置计算最近的车道索引，并获取对应的车道。
                lane_index, _ = self.map.road_network.get_closest_lane_index(ego_vehicle.position)
                lane = self.map.road_network.get_lane(lane_index)
        self.current_lane = lane
        assert lane_index == lane.index, "lane index mismatch!"  # 确保车道索引与车道的实际索引匹配
        return lane, lane_index

    def _update_target_checkpoints(self, ego_lane_index) -> bool:
        """
        更新检查点，如果车辆当前车道在之后的导航路径中则返回 True以便更新ref_lanes
        否则表示此时已经在最后一段路径，返回 False
        具体逻辑：
        刚开始时，self._target_checkpoints_index = [0, 1]，即当前检查点索引为 0，下一个检查点索引为 1。
        通过检查车辆当前车道ego_lane_index是否在从self.next_checkpoint_lane_index直到最后的检查点(实际就是规划路径中所有车道的索引)中
        来判断是否到达了下一个检查点，就是是否更换了车道。那么就更新self._target_checkpoints_index为 [idx]，[idx+1]。
        如果idx + 1 == len(self.checkpoints)，因为idx从0开始，表示已经到最后一段路，没有下一个idx了。
        """
        if self.current_checkpoint_lane_index == self.next_checkpoint_lane_index:  # on last road
            return False

        # arrive to second checkpoint
        new_index = ego_lane_index
        if new_index in self.checkpoints[self._target_checkpoints_index[1]:]:
            idx = self.checkpoints.index(new_index, self._target_checkpoints_index[
                1])  # index() 方法用于查找 new_index 在 self.checkpoints 列表中的位置（索引）。self._target_checkpoints_index[1] 是 index() 方法的起始查找位置，这意味着搜索将从这个索引开始
            self._target_checkpoints_index = [idx]
            if idx + 1 == len(self.checkpoints):
                self._target_checkpoints_index.append(idx)
            else:
                self._target_checkpoints_index.append(idx + 1)
            return True
        return False

    def _get_current_lane(self, ego_vehicle):
        """
        Called in update_localization to find current lane information
        """
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading,
            ego_vehicle.position,
            ego_vehicle.engine,
            use_heading_filter=False,
            return_on_lane=True
        )
        for lane, index, l_1_dist in possible_lanes:
            if lane in self.current_ref_lanes:
                return lane, index, on_lane
        nx_ckpt = self._target_checkpoints_index[-1]
        if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
            return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

        next_ref_lanes = self.next_ref_lanes
        for lane, index, l_1_dist in possible_lanes:
            if lane in next_ref_lanes:
                return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * len(self.current_ref_lanes)
        # return self.current_lane.width * 2
        # return self.current_lane.width * len(self.current_ref_lanes)

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(OsmTrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        # return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM  # 修改观察： 2*5

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]


class IdmReplayTrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    """
    DISCRETE_LEN = 2  # 2,mj1231  # m
    CHECK_POINT_INFO_DIM = 2  # 2,mj1231
    NUM_WAY_POINT = 10  # 10,mj1231
    # DISCRETE_LEN = 20 #2,mj1231  # m
    # CHECK_POINT_INFO_DIM = 5 #2,mj1231
    # NUM_WAY_POINT = 2  #10,mj1231
    NAVI_POINT_DIST = 50  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
            self,
            show_navi_mark: bool = False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=None,
            name=None,
            vehicle_config=None
    ):
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in TrajectoryNavigation")
        super(IdmReplayTrajectoryNavigation, self).__init__(
            show_navi_mark=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None
        self._ckpt_vis_models = None
        if show_navi_mark and self._show_navi_info:
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                self._navi_point_model.instanceTo(model)
                model.reparentTo(self.origin)

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)

    def reset(self, vehicle):
        super(IdmReplayTrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        self.set_route()

    @property
    def reference_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    def set_route(self):
        self.checkpoints = self.discretize_reference_trajectory()
        num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)

        self._navi_info.fill(0.0)
        self.next_ref_lanes = None
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        if self.reference_trajectory is None:
            return

        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        possible_lane = ray_localization(
            ego_vehicle.heading,
            ego_vehicle.position,
            ego_vehicle.engine,
            use_heading_filter=False)

        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)

        next_idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINT, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINT - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:
            ckpts += [self.checkpoints[-1] for _ in
                      range(diff)]  # 如果最后的检查点个数小于self.NUM_WAY_POINT，那么就让diff个检查点位置重合以达到消失的目的

        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        for k, ckpt in enumerate(ckpts[:]):
            start = k * self.CHECK_POINT_INFO_DIM
            end = (k + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end], lanes_heading = self._get_info_for_checkpoint(ckpt, ego_vehicle)
            # self._navi_info[start:end] = self._get_info_for_checkpoint(ckpt, ego_vehicle, self.current_lane)

            if self._show_navi_info and self._ckpt_vis_models is not None:
                pos_of_goal = ckpt
                self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                self._ckpt_vis_models[k].setH(self._goal_node_path.getH() + 3)

        self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = clip(
            (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
        )
        # print('_navi_info', self._navi_info)
        # Use RC as the only criterion to determine arrival in Scenario env.
        self._route_completion = long / self.reference_trajectory.length
        if len(possible_lane) != 0:
            current_lane = possible_lane[0][0]  # 获取当前车辆所在车道
            self.reference_trajectory.index = current_lane.index  # 更新参考路径的索引为当前车道的索引
        else:
            return True

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    @classmethod
    def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle):
        # def _get_info_for_checkpoint(cls, checkpoint, ego_vehicle, current_lane):
        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = checkpoint - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > cls.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * cls.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / cls.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))
        # print('navi_information', navi_information)

        # # 新增三维度，仿node_navi
        # # Try to include the current lane's information into the navigation information
        # bendradius = 0.0
        # dir = 0.0
        # angle = 0.0
        # if not current_lane.is_straight:
        #     bendradius = current_lane.get_radius() / (
        #         60 + 1 * current_lane.width
        #     )
        #     dir = current_lane.get_direction()
        #     angle = current_lane.get_angle()
        #
        #
        # # Dim 3: The bending radius of current lane
        # navi_information.append(clip(bendradius, 0.0, 1.0))
        #
        # # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        # navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))
        #
        # # Dim 5: The angular difference between the heading in lane ending position and
        # # the heading in lane starting position
        # navi_information.append(
        #     clip((np.rad2deg(angle) / 135 + 1) / 2, 0.0, 1.0)
        # )
        return navi_information

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(IdmReplayTrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2
        # return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM  # mj1231

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]


class TrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    """
    DISCRETE_LEN = 2  # m
    CHECK_POINT_INFO_DIM = 2
    NUM_WAY_POINT = 5
    NAVI_POINT_DIST = 30  # m, used to clip value, should be greater than DISCRETE_LEN * MAX_NUM_WAY_POINT

    def __init__(
            self,
            show_navi_mark: bool = False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=None,
            name=None,
            vehicle_config=None
    ):
        self.current_scenariolane = None
        if show_dest_mark or show_line_to_dest:
            get_logger().warning("show_dest_mark and show_line_to_dest are not supported in TrajectoryNavigation")
        super(TrajectoryNavigation, self).__init__(
            show_navi_mark=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=panda_color,
            name=name,
            vehicle_config=vehicle_config
        )
        if self.origin is not None:
            self.origin.hide(CamMask.RgbCam | CamMask.Shadow | CamMask.DepthCam | CamMask.SemanticCam)

        self._route_completion = 0
        self.checkpoints = None  # All check points

        # for compatibility
        self.next_ref_lanes = None

        # override the show navi mark function here
        self._navi_point_model = None
        self._ckpt_vis_models = None
        if show_navi_mark and self._show_navi_info:
            self._ckpt_vis_models = [NodePath(str(i)) for i in range(self.NUM_WAY_POINT)]
            for model in self._ckpt_vis_models:
                if self._navi_point_model is None:
                    self._navi_point_model = AssetLoader.loader.loadModel(AssetLoader.file_path("models", "box.bam"))
                    self._navi_point_model.setScale(0.5)
                    # if self.engine.use_render_pipeline:
                    material = Material()
                    material.setBaseColor((19 / 255, 212 / 255, 237 / 255, 1))
                    material.setShininess(16)
                    material.setEmission((0.2, 0.2, 0.2, 0.2))
                    self._navi_point_model.setMaterial(material, True)
                self._navi_point_model.instanceTo(model)
                model.reparentTo(self.origin)

        # should be updated every step after calling update_localization
        self.last_current_long = deque([0.0, 0.0], maxlen=2)
        self.last_current_lat = deque([0.0, 0.0], maxlen=2)
        self.last_current_heading_theta_at_long = deque([0.0, 0.0], maxlen=2)

    def reset(self, vehicle):
        super(TrajectoryNavigation, self).reset(current_lane=self.reference_trajectory)
        self.set_route()

    @property
    def reference_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    @property
    def current_ref_lanes(self):
        return [self.reference_trajectory]

    def set_route(self):
        self.checkpoints = self.discretize_reference_trajectory()
        num_way_point = min(len(self.checkpoints), self.NUM_WAY_POINT)
        self._target_checkpoints_index = [0, 1]
        self._navi_info.fill(0.0)
        self.next_ref_lanes = None
        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(panda_vector(check_point[0], check_point[1], 1))

    def discretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DISCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.position(i * self.DISCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        if self.reference_trajectory is None:
            return

        # Update ckpt index
        long, lat = self.reference_trajectory.local_coordinates(ego_vehicle.position)
        ego_lane, lane_index = self._update_current_lane(ego_vehicle)  # 根据车辆位置更新当前所在车道
        self.current_scenariolane = ego_lane
        heading_theta_at_long = self.reference_trajectory.heading_theta_at(long)
        self.last_current_heading_theta_at_long.append(heading_theta_at_long)
        self.last_current_long.append(long)
        self.last_current_lat.append(lat)

        next_idx = max(int(long / self.DISCRETE_LEN) + 1, 0)
        next_idx = min(next_idx, len(self.checkpoints) - 1)
        end_idx = min(next_idx + self.NUM_WAY_POINT, len(self.checkpoints))
        ckpts = self.checkpoints[next_idx:end_idx]
        diff = self.NUM_WAY_POINT - len(ckpts)
        assert diff >= 0, "Number of Navigation points error!"
        if diff > 0:
            ckpts += [self.checkpoints[-1] for _ in range(diff)]

        # target_road_1 is the road segment the vehicle is driving on.
        self._navi_info.fill(0.0)
        for k, ckpt in enumerate(ckpts[1:]):
            start = k * self.CHECK_POINT_INFO_DIM
            end = (k + 1) * self.CHECK_POINT_INFO_DIM
            self._navi_info[start:end] = self._get_info_for_checkpoint(ckpt, ego_vehicle, ego_lane)
            if self._show_navi_info and self._ckpt_vis_models is not None:
                pos_of_goal = ckpt
                self._ckpt_vis_models[k].setPos(panda_vector(pos_of_goal[0], pos_of_goal[1], self.MARK_HEIGHT))
                self._ckpt_vis_models[k].setH(self._goal_node_path.getH() + 3)

        self._navi_info[end] = clip((lat / self.engine.global_config["max_lateral_dist"] + 1) / 2, 0.0, 1.0)
        self._navi_info[end + 1] = clip(
            (wrap_to_pi(heading_theta_at_long - ego_vehicle.heading_theta) / np.pi + 1) / 2, 0.0, 1.0
        )

        # Use RC as the only criterion to determine arrival in Scenario env.
        self._route_completion = long / self.reference_trajectory.length

    def _update_current_lane(self, ego_vehicle):
        """
        具体逻辑就是根据车辆的位置，计算车辆当前所在的车道，因为在一个step中车辆的位置会发生变化。
        但如果根据目前位置无法确定车道，那么就根据车辆之前的位置来确定车道。
        ego_vehicle.lane调用的还是self.navigation.current_lane来获得车道信息，
        所以在没有执行self.current_lane = lane这一句时还保留了之前的车道信息。
        不论通过哪种方式获得的lane都在self.current_lane = lane这一句中更新了当前车道。
        """
        lane, lane_index, on_lane = self._get_current_lane(ego_vehicle)
        ego_vehicle.on_lane = on_lane
        if on_lane == False:
            pass
        if lane is None:  # 如果未找到当前车道，则从 ego_vehicle 获取之前的车道和车道索引
            lane, lane_index = ego_vehicle.lane, ego_vehicle.lane_index
            if self.FORCE_CALCULATE:
                # 若为 True，则根据车辆当前位置计算最近的车道索引，并获取对应的车道。
                lane_index, _ = self.map.road_network.get_closest_lane_index(ego_vehicle.position)
                lane = self.map.road_network.get_lane(lane_index)
        assert lane_index == lane.index, "lane index mismatch!"  # 确保车道索引与车道的实际索引匹配
        # print('lane_index', lane_index)
        return lane, lane_index

    def _get_current_lane(self, ego_vehicle):
        """
        Called in update_localization to find current lane information
        """
        possible_lanes, on_lane = ray_localization(
            ego_vehicle.heading,
            ego_vehicle.position,
            ego_vehicle.engine,
            use_heading_filter=False,
            return_on_lane=True
        )
        # print('ego_position',ego_vehicle.position)
        for lane, index, l_1_dist in possible_lanes:
            if lane in self.current_ref_lanes:
                return lane, index, on_lane
        # nx_ckpt = self._target_checkpoints_index[-1]
        # if nx_ckpt == self.checkpoints[-1] or self.next_ref_lanes is None:
        #     return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

        # next_ref_lanes = self.next_ref_lanes
        # for lane, index, l_1_dist in possible_lanes:
        #     if lane in next_ref_lanes:
        #         return lane, index, on_lane
        return (*possible_lanes[0][:-1], on_lane) if len(possible_lanes) > 0 else (None, None, on_lane)

    def get_current_lateral_range(self, current_position, engine) -> float:
        return self.current_lane.width * 2

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        from head.component.map.lane_utils import count_adjacent_lanes
        try:
            lane = self.current_scenariolane
            if not lane:
                return 1
            lane_id = lane.index
            road_network = self.engine.map_manager.current_map.road_network
            return count_adjacent_lanes(lane_id, road_network)
        except Exception as e:
            get_logger().error(f"Error in counting adjacent lanes: {e}")
        return 1

    # @classmethod
    def _get_info_for_checkpoint(self, checkpoint, ego_vehicle, current_lane):
        navi_information = []
        # Project the checkpoint position into the target vehicle's coordination, where
        # +x is the heading and +y is the right hand side.
        dir_vec = checkpoint - ego_vehicle.position  # get the vector from center of vehicle to checkpoint
        dir_norm = norm(dir_vec[0], dir_vec[1])
        if dir_norm > self.NAVI_POINT_DIST:  # if the checkpoint is too far then crop the direction vector
            dir_vec = dir_vec / dir_norm * self.NAVI_POINT_DIST
        ckpt_in_heading, ckpt_in_rhs = ego_vehicle.convert_to_local_coordinates(
            dir_vec, 0.0
        )  # project to ego vehicle's coordination

        # Dim 1: the relative position of the checkpoint in the target vehicle's heading direction.
        navi_information.append(clip((ckpt_in_heading / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        # Dim 2: the relative position of the checkpoint in the target vehicle's right hand side direction.
        navi_information.append(clip((ckpt_in_rhs / self.NAVI_POINT_DIST + 1) / 2, 0.0, 1.0))

        """当检查点维度信息需设为5时才增加以下代码处理"""
        # # 尝试将当前车道的信息包含到导航信息中，如果当前车道是环形车道，那么将环形车道的信息包含到导航信息中，否则为0
        # bendradius = 0.0
        # dir = 0.0
        # angle = 0.0
        #
        # from metadrive.component.pg_space import Parameter, BlockParameterSpace
        # if not current_lane.is_straight:
        #     current_lane_num = self.get_current_lane_num()
        #     # print("current_lane_num current_lane_num current_lane_num", current_lane_num)
        #     bendradius = current_lane.get_radius() / (
        #             60 + current_lane_num * 3.5
        #     )
        #     dir = current_lane.get_direction()
        #     angle = current_lane.get_angle()
        #
        # # Dim 3: The bending radius of current lane
        # navi_information.append(clip(bendradius, 0.0, 1.0))
        #
        # # Dim 4: The bending direction of current lane (+1 for clockwise, -1 for counterclockwise)
        # navi_information.append(clip((dir + 1) / 2, 0.0, 1.0))
        #
        # # Dim 5: The angular difference between the heading in lane ending position and
        # # the heading in lane starting position
        # navi_information.append(
        #     clip((np.rad2deg(angle) / BlockParameterSpace.CURVE[Parameter.angle].max + 1) / 2, 0.0, 1.0)
        # )
        return navi_information

    def destroy(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None
        super(TrajectoryNavigation, self).destroy()

    def before_reset(self):
        self.checkpoints = None
        # self.current_ref_lanes = None
        self.next_ref_lanes = None
        self.final_lane = None
        self._current_lane = None
        # self.reference_trajectory = None

    @property
    def route_completion(self):
        return self._route_completion

    @classmethod
    def get_navigation_info_dim(cls):
        return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM + 2
        # return cls.NUM_WAY_POINT * cls.CHECK_POINT_INFO_DIM # 2*5

    @property
    def last_longitude(self):
        return self.last_current_long[0]

    @property
    def current_longitude(self):
        return self.last_current_long[1]

    @property
    def last_lateral(self):
        return self.last_current_lat[0]

    @property
    def current_lateral(self):
        return self.last_current_lat[1]

    @property
    def last_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[0]

    @property
    def current_heading_theta_at_long(self):
        return self.last_current_heading_theta_at_long[1]
