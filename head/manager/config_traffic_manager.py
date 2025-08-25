import copy
import logging
from collections import namedtuple
from typing import Dict

import math
import numpy as np

from metadrive.component.lane.abs_lane import AbstractLane
from metadrive.component.map.base_map import BaseMap
from metadrive.component.road_network import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.constants import TARGET_VEHICLES, TRAFFIC_VEHICLES, OBJECT_TO_AGENT, AGENT_TO_OBJECT
from metadrive.manager.base_manager import BaseManager
from metadrive.utils import merge_dicts

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class TrafficMode:
    # Traffic vehicles will be respawned, once they arrive at the destinations
    Respawn = "respawn"

    # Traffic vehicles will be triggered only once
    Trigger = "trigger"

    # Hybrid, some vehicles are triggered once on map and disappear when arriving at destination, others exist all time
    Hybrid = "hybrid"


class ConTrafficManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Control the whole traffic flow
        """
        super(ConTrafficManager, self).__init__()

        self._traffic_vehicles = []

        self.block_triggered_vehicles = []

        # traffic property
        self.mode = self.engine.global_config["traffic_mode"]
        self.random_traffic = self.engine.global_config["random_traffic"]
        self.density = self.engine.global_config["traffic_density"]
        self.respawn_lanes = None

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        logging.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))

        # update vehicle list
        self.block_triggered_vehicles = []

        traffic_density = self.density
        if abs(traffic_density) < 1e-2:
            return
        self.respawn_lanes = self._get_available_respawn_lanes(map)
        if self.mode == TrafficMode.Respawn:
            # add respawn vehicle
            self._create_respawn_vehicles(map, traffic_density)
        elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
            self._create_vehicles_once(map, traffic_density)
        else:
            raise ValueError("No such mode named {}".format(self.mode))

    # 这段代码是一个Python类的方法，方法名为reset，其目的是根据给定的模式和密度在地图上生成交通车辆。代码中首先判断是否需要生成随机交通车辆，然后对交通车辆列表进行更新操作。
    #
    # 然后根据给定的交通密度来决定接下来的动作。如果密度很小，则直接返回，否则根据模式选择具体生成车辆的策略：如果模式是Respawn，则调用_create_respawn_vehicles方法生成车辆；如果模式是Trigger或Hybrid，则调用_create_vehicles_once方法生成车辆。
    #
    # 在方法的结尾，如果模式不是上述提到的模式，则抛出一个ValueError异常，提示该模式未定义。

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine
        if self.mode != TrafficMode.Respawn:
            for v in engine.agent_manager.active_agents.values():
                ego_lane_idx = v.lane_index[:-1]
                ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
                if len(self.block_triggered_vehicles) > 0 and \
                        ego_road == self.block_triggered_vehicles[-1].trigger_road:
                    block_vehicles = self.block_triggered_vehicles.pop()
                    self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())
        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    # 这段代码是一个方法`before_step`的实现，用于处理交通车辆在每一步之前的一系列操作。具体解释如下：
    # 1.获取当前引擎对象`renderer`，以及当前模式`mode`。
    # 2.如果当前模式不是`TrafficMode.Respawn`（不是重生模式），则遍历活动代理车辆。
    # 3.获取每个代理车辆的车道索引和对应的道路对象。
    # 4.如果有阻塞触发的车辆（`block_triggered_vehicles`列表不为空），则将最后一个阻塞车辆的车辆列表中的车辆加入到整体交通车辆列表中。
    # 5.遍历整体交通车辆列表，获取每辆车的策略对象，并调用其`act()`方法作出驾驶决策。
    # 6.返回空的字典对象。
    # 总的来说，这段代码实现了交通车辆在每一步之前的决策过程，包括获取每辆车的策略和作出相应的驾驶决策。

    def after_step(self, *args, **kwargs):
        """
        Update all traffic vehicles' states,
        """
        v_to_remove = []
        for v in self._traffic_vehicles:
            v.after_step()
            if not v.on_lane:
                if self.mode == TrafficMode.Trigger:
                    v_to_remove.append(v)
                elif self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                    v_to_remove.append(v)
                else:
                    raise ValueError("Traffic mode error: {}".format(self.mode))
        for v in v_to_remove:
            vehicle_type = type(v)
            self.clear_objects([v.id])
            self._traffic_vehicles.remove(v)
            if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
                lane = self.respawn_lanes[self.np_random.randint(0, len(self.respawn_lanes))]
                lane_idx = lane.index
                long = self.np_random.rand() * lane.length / 2
                traffic_v_config = {"spawn_lane_index": lane_idx, "spawn_longitude": long}
                new_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
                from metadrive.policy.idm_policy import IDMPolicy
                self.add_policy(new_v.id, IDMPolicy, new_v, self.generate_seed())
                self._traffic_vehicles.append(new_v)

        return dict()

    # 这段代码是一个类方法`after_step`，在每一步模拟后被调用。它的主要目的是更新所有的交通车辆的状态。
    # 首先，它创建了一个空列表`v_to_remove`，用于存放需要移除的车辆。
    # 然后，它通过一个循环遍历所有的交通车辆，并对每辆车调用`after_step`方法。如果车辆不在车道上（`on_lane`为False），则根据当前的交通模式（`mode`）做出不同的处理。
    # 如果交通模式是`Trigger`、`Respawn`或`Hybrid`，则将该车辆加入`v_to_remove`列表中。
    # 如果交通模式不是这三种情况，则抛出一个值错误异常。接下来，再次遍历`v_to_remove`列表中的车辆，对每一辆车执行以下操作：
    # 1.获取车辆类型
    # 2.清除该车辆
    # 3.从交通车辆列表中移除该车辆
    # 4.如果交通模式是`Respawn`或`Hybrid`，则在随机选择的复活车道上生成一个新车辆，并加入到交通车辆列表中
    # 最后，方法返回一个空字典。
    # 总体来说，该方法的作用是更新每辆交通车辆的状态，并根据不同的交通模式处理需要移除的车辆或生成新的车辆。

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(ConTrafficManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self.block_triggered_vehicles = []
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        if self.mode == TrafficMode.Respawn:
            return len(self._traffic_vehicles)
        return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)

    def get_global_states(self) -> Dict:
        """
        Return all traffic vehicles' states
        :return: States of all vehicles
        """
        states = dict()
        traffic_states = dict()
        for vehicle in self._traffic_vehicles:
            traffic_states[vehicle.index] = vehicle.get_state()

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    traffic_states[vehicle.index] = vehicle.get_state()
        states[TRAFFIC_VEHICLES] = traffic_states
        active_obj = copy.copy(self.engine.agent_manager._active_objects)
        pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
        dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
        states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v.get_state()
                                      for k, v in pending_obj.items()}, allow_new_keys=True
        )
        states[TARGET_VEHICLES] = merge_dicts(
            states[TARGET_VEHICLES], {k: v_count[0].get_state()
                                      for k, v_count in dying_obj.items()},
            allow_new_keys=True
        )

        states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
        states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
        return states

    def get_global_init_states(self) -> Dict:
        """
        Special handling for first states of traffic vehicles
        :return: States of all vehicles
        """
        vehicles = dict()
        for vehicle in self._traffic_vehicles:
            init_state = vehicle.get_state()
            init_state["index"] = vehicle.index
            init_state["type"] = vehicle.class_name
            init_state["enable_respawn"] = vehicle.enable_respawn
            vehicles[vehicle.index] = init_state

        # collect other vehicles
        if self.mode != TrafficMode.Respawn:
            for v_b in self.block_triggered_vehicles:
                for vehicle in v_b.vehicles:
                    init_state = vehicle.get_state()
                    init_state["type"] = vehicle.class_name
                    init_state["index"] = vehicle.index
                    init_state["enable_respawn"] = vehicle.enable_respawn
                    vehicles[vehicle.index] = init_state
        return vehicles

    def _propose_vehicle_configs(self, lane: AbstractLane):
        potential_vehicle_configs = []
        total_num = int(lane.length / self.VEHICLE_GAP)
        vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
        # Only choose given number of vehicles
        for long in vehicle_longs:
            random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
            potential_vehicle_configs.append(random_vehicle_config)
        return potential_vehicle_configs

    def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float):
        total_num = len(self.respawn_lanes)

        # row   0   1  2  ... 19      <== column
        # ----------------
        # -1 |  0   4  8  ... 76
        # -2 |  1   5  9  ... 77
        # -3 |  2   6 10  ... 78
        # -4 |  3   7 11  ... 79
        # ----------------
        # grid width = Lane
        # width = 3.5(m)
        # grid length = 10(m)
        np.random.seed(self.generate_seed())
        # for lane in self.respawn_lanes:
        _traffic_vehicles = []
        if self.global_config['scenario_difficulty'] == 2:
            grid_choices = np.arange(20, 60, 4)
        if self.global_config['scenario_difficulty'] == 1:
            grid_choices = np.arange(20, 60, 8)
        else:
            grid_choices = np.arange(20, 40, 8)

        LIMIT_SPEED = 40.0
        if self.engine.global_config.map_config['lane_num'] == 3:
            grid_choices = np.arange(20, 40, 8)
            SPEED_DIFFERENCE = [15.0, 10.0, 18.75]
            long = self.VEHICLE_GAP * grid_choices / 3
        else:
            SPEED_DIFFERENCE = [5.0, 20.0, 30.0, 40.0, 45.0]
            long = self.VEHICLE_GAP * grid_choices / 4

        for long in long.tolist():
            lane = self.respawn_lanes[np.random.choice(self.engine.global_config.map_config['lane_num'])]
            vehicle_type = self.random_vehicle_type()
            traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "spawn_lateral": 0}

            # print(lane.index)
            traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
            from head.policy.basic_policy.idm_policy_include_pedestrian import IDMPolicyIncluedPedestrain
            difference = float(np.random.choice(SPEED_DIFFERENCE, 1))
            # print(difference)
            IDMPolicyIncluedPedestrain.NORMAL_SPEED = LIMIT_SPEED*difference/100

            self.add_policy(random_v.id, IDMPolicyIncluedPedestrain, random_v, self.generate_seed())

            self._traffic_vehicles.append(random_v)

    # 这段代码是用于在程序中生成基于交通密度的车辆，并将其添加到模拟环境中的指定车道中。具体步骤如下：
    # 1.对于每个重生车道`respawn_lanes`中的车道`lane`：
    # 2.初始化一个空列表来存储生成的交通车辆`_traffic_vehicles`。
    # 3.计算这条车道可容纳的总车辆数`total_num`，即车道长度除以车辆间距`VEHICLE_GAP`取整。
    # 4.生成一组车辆的经过位置`vehicle_longs`，该位置按间距`VEHICLE_GAP`递增，范围是0到车道长度。
    # 5.对经过位置进行随机打乱。
    # 6.对于经过位置中的一部分位置，数量为`traffic_density`乘以总位置数(向上取整)，生成车辆在该位置：
    # -  这里使用`np.ceil()`对生成车辆数量进行向上取整。
    # -  调用 `random_vehicle_type()`来随机选择车辆类型。
    # -  构建车辆的配置信息`traffic_v_config`，包括生成的车道索引和位置。
    # -  更新车辆配置信息，将全局配置中的交通车辆配置合并到车辆配置中。
    # -  使用 `spawn_object()`方法生成具有指定配置信息的车辆对象。
    # -  导入`IDMPolicy`类，并为生成的车辆对象添加该策略，并传递种子值。
    # -  将生成的交通车辆对象添加到列表`_traffic_vehicles`中。
    # 通过以上步骤，程序会为每个重生车道生成对应数量和类型的车辆，并在模拟环境中添加这些交通车辆。

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
            # print("We have {} candidates! We are spawning {} vehicles!".format(total_vehicles, len(selected)))

            from metadrive.policy.idm_policy import IDMPolicy
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

    def _get_available_respawn_lanes(self, map: BaseMap) -> list:
        """
        Used to find some respawn lanes
        :param map: select spawn lanes from this map
        :return: respawn_lanes
        """
        respawn_lanes = []
        respawn_roads = []
        for block in map.blocks:
            roads = block.get_respawn_roads()
            for road in roads:
                if road in respawn_roads:
                    respawn_roads.remove(road)
                else:
                    respawn_roads.append(road)
        for road in respawn_roads:
            respawn_lanes += road.get_lanes(map.road_network)
        return respawn_lanes

    def random_vehicle_type(self):
        from metadrive.component.vehicle.vehicle_type import random_vehicle_type
        vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self._traffic_vehicles = None
        self.block_triggered_vehicles = None

        # traffic property
        self.mode = None
        self.random_traffic = None
        self.density = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self._traffic_vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    def seed(self, random_seed):
        if not self.random_traffic:
            super(ConTrafficManager, self).seed(random_seed)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map

    def get_state(self):
        ret = super(ConTrafficManager, self).get_state()
        ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
        flat = []
        for b_v in self.block_triggered_vehicles:
            flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
        ret["block_triggered_vehicles"] = flat
        return ret

    def set_state(self, state: dict, old_name_to_current=None):
        super(ConTrafficManager, self).set_state(state, old_name_to_current)
        self._traffic_vehicles = list(
            self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
        )
        self.block_triggered_vehicles = [
            BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
            for s, e, v in state["block_triggered_vehicles"]
        ]


# For compatibility check
TrafficManager = ConTrafficManager


class MixedPGTrafficManager(ConTrafficManager):
    def _create_respawn_vehicles(self, *args, **kwargs):
        raise NotImplementedError()

    def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
        vehicle_num = 0
        for block in map.blocks[1:]:

            # Propose candidate locations for spawning new vehicles
            trigger_lanes = block.get_intermediate_spawn_lanes()
            if self.engine.global_config["need_inverse_traffic"] and block.ID in ["S", "C", "r", "R"]:
                neg_lanes = block.block_network.get_negative_lanes()
                self.np_random.shuffle(neg_lanes)
                trigger_lanes += neg_lanes
            potential_vehicle_configs = []
            for lanes in trigger_lanes:
                for l in lanes:
                    if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
                        continue
                    potential_vehicle_configs += self._propose_vehicle_configs(l)

            # How many vehicles should we spawn in this block?
            total_length = sum([lane.length for lanes in trigger_lanes for lane in lanes])
            total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
            total_vehicles = int(math.floor(total_spawn_points * traffic_density))

            # Generate vehicles!
            vehicles_on_block = []
            self.np_random.shuffle(potential_vehicle_configs)
            selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]

            from metadrive.policy.idm_policy import IDMPolicy
            from metadrive.policy.expert_policy import ExpertPolicy
            # print("===== We are initializing {} vehicles =====".format(len(selected)))
            # print("Current seed: ", self.renderer.global_random_seed)
            for v_config in selected:
                vehicle_type = self.random_vehicle_type()
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
                if self.np_random.random() < self.engine.global_config["rl_agent_ratio"]:
                    # print("Vehicle {} is assigned with RL policy!".format(random_v.id))
                    self.add_policy(random_v.id, ExpertPolicy, random_v, self.generate_seed())
                else:
                    self.add_policy(random_v.id, IDMPolicy, random_v, self.generate_seed())
                vehicles_on_block.append(random_v.name)

            trigger_road = block.pre_block_socket.positive_road
            block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)

            self.block_triggered_vehicles.append(block_vehicles)
            vehicle_num += len(vehicles_on_block)
        self.block_triggered_vehicles.reverse()

# from metadrive.engine.logger import get_logger
# from metadrive.policy.idm_policy import IDMPolicy, MyPolicy
#
#
# logger = get_logger()
#
# class GenTrafficManager(BaseManager):
#     VEHICLE_GAP = 10  # m
#     """
#     GenTrafficManager: 基于规则的交通流管理器，不依赖轨迹数据，在地图上动态生成、移除车辆，主要用于合成交通流仿真
#     1. 数据来源：GenTrafficManager基于地图规则生成车辆；ScenarioTrafficManager基于真实轨迹回放。
#     2. 控制策略：GenTrafficManager用自定义Policy；ScenarioTrafficManager用Replay/IDM。
#     3. 动态管理：GenTrafficManager支持respawn/trigger模式保持车流；ScenarioTrafficManager严格跟随数据。
#     4. 支持对象：GenTrafficManager仅生成车辆；ScenarioTrafficManager还包含行人、自行车、障碍物。
#     5. 适用场景：GenTrafficManager适合规则化交通仿真；ScenarioTrafficManager适合真实数据复现。
#     """
#     def __init__(self):
#         """
#         控制整个交通流
#         """
#         super(GenTrafficManager, self).__init__()
#
#         self._traffic_vehicles = []
#
#         # triggered by the event. TODO(lqy) In the future, event trigger can be introduced
#         self.block_triggered_vehicles = []
#
#         # traffic property
#         self.mode = self.engine.global_config["traffic_mode"]
#         self.random_traffic = self.engine.global_config["random_traffic"]
#         # self.density = self.engine.global_config["traffic_density"]
#         self.respawn_lanes = None
#         self.cv = None
#         self.v_to_remove = None
#
#     def reset(self):
#         """
#         根据生成模式和生成密度，在地图上生成交通
#         :return: List of Traffic vehicles
#         """
#         map = self.current_map
#         logger.debug("load scene {}".format("Use random traffic" if self.random_traffic else ""))
#
#         # update vehicle list
#         self.block_triggered_vehicles = []
#
#         traffic_density = 0.1
#         if abs(traffic_density) < 1e-2:
#             return
#         self.respawn_lanes = self.engine.current_map.road_network.get_all_lanes()
#         if self.mode == TrafficMode.Respawn:
#             self._create_respawn_vehicles(map, traffic_density)
#             # self._create_vehicles_once(map, traffic_density)
#         elif self.mode == TrafficMode.Trigger or self.mode == TrafficMode.Hybrid:
#             self._create_vehicles_once(map, traffic_density)
#         else:
#             raise ValueError("No such mode named {}".format(self.mode))
#
#     def before_step(self):
#         """
#         All traffic vehicles make driving decision here
#         :return: None
#         """
#         # trigger vehicles
#         engine = self.engine
#         if self.mode != TrafficMode.Respawn:
#             for v in engine.agent_manager.active_agents.values():
#                 ego_lane_idx = v.lane_index[:-1]
#                 ego_road = Road(ego_lane_idx[0], ego_lane_idx[1])
#                 if len(self.block_triggered_vehicles) > 0 and \
#                         ego_road == self.block_triggered_vehicles[-1].trigger_road:
#                     block_vehicles = self.block_triggered_vehicles.pop()
#                     self._traffic_vehicles += list(self.get_objects(block_vehicles.vehicles).values())
#         for v in self._traffic_vehicles:
#             p = self.engine.get_policy(v.name)
#             self.cv = self.closest_vehicle
#             # if v == self.cv and hasattr(self,'trans_adbv') and self.engine.global_config.trans_adbv == True and self.engine.top_down_renderer is not None:
#             #     p = MyEnvInputPolicy(v,self.generate_seed())
#             #     self.engine.top_down_renderer.closet_vehicle = v.name
#             #     v.before_step(p.act("bv_agent"))
#             # else:
#             v.before_step(p.act(v.id))
#
#         return dict()
#
#     def step(self):
#         if self.mode == TrafficMode.Respawn or self.mode == TrafficMode.Hybrid:
#             self._create_respawn_vehicles(self.current_map, self.density, False)
#
#     @property
#     def closest_vehicle(self):
#         vehicles = self.vehicles
#         distance = dict()
#         for vehicle in vehicles:
#             if vehicle.class_name == "DefaultVehicle":  # "DefaultVehicle" is the ego vehicle
#                 ego_position = vehicle.position
#         for vehicle in vehicles:  # calculate the minimal distance
#             if vehicle.class_name != "AdbvVehicle" and vehicle.class_name != "DefaultVehicle":  # if
#                 distance[vehicle.id] = np.linalg.norm(ego_position - vehicle.position)
#
#         if (self.cv != None) and (self.cv.name in distance) and (
#                 distance[self.cv.name] - min(distance.values()) <= 10):  # and (self.cv.name in distance): # 判断是不是初始化
#             return self.cv
#         else:
#             return list(self.engine.get_object(min(distance, key=distance.get)).values())[0]
#
#     def after_step(self, *args, **kwargs):
#         """
#         方法主要用于更新交通系统中每辆车的状态，并根据车辆*是否在车道上*以及当前的交通模式决定是否移除车辆。
#         如果车辆被移除且当前模式允许重生，则会在随机选择的车道上生成新的车辆。
#         该方法确保交通流的动态管理，涉及车辆状态更新和重生机制。
#         """
#         v_to_remove = []
#         for v in self._traffic_vehicles:
#             v.after_step()
#             if not v.on_lane or v.crash_vehicle or v.crash_object or v.crash_sidewalk or v.crash_building:
#                 v_to_remove.append(v)
#         self.v_to_remove = v_to_remove
#
#         for v in v_to_remove:
#             vehicle_type = type(v)  # 保证生成的车辆类型与原车辆类型一致
#             self.clear_objects([v.id])
#             self._traffic_vehicles.remove(v)
#
#         return dict()
#
#     def before_reset(self) -> None:
#         """
#         Clear the scene and then reset the scene to empty
#         :return: None
#         """
#         super(GenTrafficManager, self).before_reset()
#         self.density = 0.1
#         self.block_triggered_vehicles = []
#         self._traffic_vehicles = []
#
#     def get_vehicle_num(self):
#         """
#         Get the vehicles on road
#         :return:
#         """
#         if self.mode == TrafficMode.Respawn:
#             return len(self._traffic_vehicles)
#         return sum(len(block_vehicle_set.vehicles) for block_vehicle_set in self.block_triggered_vehicles)
#
#     def get_global_states(self) -> Dict:
#         """
#         有问题
#         Return all traffic vehicles' states
#         :return: States of all vehicles
#         """
#         states = dict()
#         traffic_states = dict()
#         # for vehicle in self._traffic_vehicles:
#         #     traffic_states[vehicle.id] = vehicle.get_state()
#         vehicles = self.vehicles
#
#         # # collect other vehicles
#         # if self.mode != TrafficMode.Respawn:
#         #     for v_b in self.block_triggered_vehicles:
#         #         for vehicle in v_b.vehicles:
#         #             # traffic_states[vehicle.id] = vehicle.get.get_state()
#         #             traffic_states[vehicle] = (self.get_object(vehicle).values()).get_state()
#         states[TRAFFIC_VEHICLES] = traffic_states
#         active_obj = copy.copy(self.engine.agent_manager._active_objects)
#         pending_obj = copy.copy(self.engine.agent_manager._pending_objects)
#         dying_obj = copy.copy(self.engine.agent_manager._dying_objects)
#         states[TARGET_VEHICLES] = {k: v.get_state() for k, v in active_obj.items()}
#         states[TARGET_VEHICLES] = merge_dicts(
#             states[TARGET_VEHICLES], {k: v.get_state()
#                                       for k, v in pending_obj.items()}, allow_new_keys=True
#         )
#         states[TARGET_VEHICLES] = merge_dicts(
#             states[TARGET_VEHICLES], {k: v_count[0].get_state()
#                                       for k, v_count in dying_obj.items()}, allow_new_keys=True
#         )
#
#         states[OBJECT_TO_AGENT] = copy.deepcopy(self.engine.agent_manager._object_to_agent)
#         states[AGENT_TO_OBJECT] = copy.deepcopy(self.engine.agent_manager._agent_to_object)
#         return states
#
#     def get_global_init_states(self) -> Dict:
#         """
#         Special handling for first states of traffic vehicles
#         :return: States of all vehicles
#         """
#         vehicles = dict()
#         for vehicle in self._traffic_vehicles:
#             init_state = vehicle.get_state()
#             init_state["index"] = vehicle.index
#             init_state["type"] = vehicle.class_name
#             init_state["enable_respawn"] = vehicle.enable_respawn
#             vehicles[vehicle.index] = init_state
#
#         # collect other vehicles
#         if self.mode != TrafficMode.Respawn:
#             for v_b in self.block_triggered_vehicles:
#                 for vehicle in v_b.vehicles:
#                     init_state = vehicle.get_state()
#                     init_state["type"] = vehicle.class_name
#                     init_state["index"] = vehicle.index
#                     init_state["enable_respawn"] = vehicle.enable_respawn
#                     vehicles[vehicle.index] = init_state
#         return vehicles
#
#     def _propose_vehicle_configs(self, lane: AbstractLane):
#         potential_vehicle_configs = []
#         total_num = int(lane.length / self.VEHICLE_GAP)
#         vehicle_longs = [i * self.VEHICLE_GAP for i in range(total_num)]
#         vehicle_longs = [lane.length / 2]
#         # Only choose given number of vehicles
#         for long in vehicle_longs:
#             random_vehicle_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "enable_reverse": False}
#             potential_vehicle_configs.append(random_vehicle_config)
#         return potential_vehicle_configs
#
#
#     def _create_respawn_vehicles(self, map: BaseMap, traffic_density: float, near_ego_lane: bool = False):
#         '''
#         适用于地图包括车道中心线连接的情况，若是只有车道边界的情况则不适用，此时respawn_lanes为空
#         '''
#         all_lanes = map.road_network.get_all_lanes()  # 获取所有车道
#         ego_vehicle = self.vehicles[0]
#         from metadrive.utils.pg.utils import ray_localization
#         ego_lane = ray_localization(ego_vehicle.heading, ego_vehicle.position, self.engine, use_heading_filter=True)
#         if near_ego_lane and len(ego_lane) != 0:  # 如果需要生成在自车附近
#             near_lane_depth_2 = self.get_near_lanes(ego_lane[0][0], 2)
#             near_lane_depth_3 = self.get_near_lanes(ego_lane[0][0], 3)
#             near_lanes = [lane for lane in near_lane_depth_3 if lane not in near_lane_depth_2]
#             # near_lanes = [lane for lane in near_lane_depth_2 if lane not in near_lane_depth_1]
#         else:
#             # 使用所有车道
#             near_lanes = []
#
#         # 从道路开始端进行生成
#         spawn_from_entry = True
#         if spawn_from_entry:
#             near_lanes = [lane for lane in all_lanes if len(lane.entry_lanes) == 0]
#             forward_lanes = []
#             for lane in near_lanes:
#                 if lane.index in ['232001', '231817', '231884', '232066', '231910', '231824', '231960', '232064',
#                                   '231832', '232015']:
#                     forward_lanes.append(lane)
#             near_lanes = forward_lanes
#
#         potential_vehicle_configs = []
#         near_potential_vehicle_configs = []
#         occupied_lanes = set([vehicle.lane for vehicle in self.vehicles])
#         for l in all_lanes:
#             if l not in occupied_lanes:
#                 if l in near_lanes:
#                     near_potential_vehicle_configs += self._propose_vehicle_configs(l)
#         for l in all_lanes:
#             if l.junction == False and l not in occupied_lanes:  # 不在交叉口生成
#                 if l not in near_lanes:
#                     potential_vehicle_configs += self._propose_vehicle_configs(l)
#
#                     # 以密度设置生成数量
#         total_length = sum([lane.length for lane in all_lanes])
#         total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
#         total_vehicles = int(math.floor(total_spawn_points * traffic_density))
#         total_vehicles = 80
#         diff_vehicles = total_vehicles - len(self._traffic_vehicles)  # 计算需要生成的车辆数目，保持总数基本不变
#         # Generate vehicles!
#         self.np_random.shuffle(potential_vehicle_configs)
#         self.np_random.shuffle(near_potential_vehicle_configs)
#         potential_vehicle_configs = near_potential_vehicle_configs[
#                                     :min(len(near_potential_vehicle_configs), len(near_potential_vehicle_configs))] \
#                                     + potential_vehicle_configs  # 合并两个列表,取前几个近邻车道的候选位置
#         selected = potential_vehicle_configs[:min(diff_vehicles, len(potential_vehicle_configs))]
#
#         for traffic_v_config in selected:
#             vehicle_type = self.random_vehicle_type()
#             traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
#             random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
#             self.add_policy(random_v.id, MyPolicy, random_v, self.generate_seed())
#             self._traffic_vehicles.append(random_v)
#
#         logger.debug("Total {} vehicles are created!".format(len(self._traffic_vehicles)))
#
#     def get_near_lanes(self, ego_lane, max_depth):
#         ego_lane_index = ego_lane.index
#         processed_lanes = set()  # 已处理的车道
#         lanes_to_process = set([ego_lane_index])  # 待处理的车道
#         near_lanes = set([ego_lane_index])  # 结果集：所有相邻车道
#
#         current_depth = 0
#
#         while lanes_to_process and current_depth < max_depth:
#             next_lanes = set()  # 下一轮要处理的车道
#
#             for lane_index in lanes_to_process:
#                 if lane_index in processed_lanes:
#                     continue
#
#                 processed_lanes.add(lane_index)
#                 lane = self.engine.map_manager.current_map.road_network.get_lane(lane_index)
#
#                 # 添加左右车道
#                 next_lanes.update(lane.left_lanes)
#                 next_lanes.update(lane.right_lanes)
#
#                 # 添加入口和出口车道
#                 next_lanes.update(lane.entry_lanes)
#                 next_lanes.update(lane.exit_lanes)
#
#             # 更新近邻车道集合
#             near_lanes.update(next_lanes)
#             lanes_to_process = next_lanes - processed_lanes
#             current_depth += 1
#
#         near_lanes = [self.engine.map_manager.current_map.road_network.get_lane(idx) for idx in near_lanes]
#         return near_lanes
#
#     def calculate_centerline_using_medial_axis(self, polygon):
#         """
#         使用多边形中轴线算法计算车道中心线，并计算每个点的航向角
#
#         参数:
#         - polygon: 车道多边形点列表，按顺时针或逆时针顺序排列
#
#         返回:
#         - centerline: 中心线点列表，每个点为 [x, y]
#         - headings: 对应每个中心线点的航向角列表（弧度）
#         """
#         import numpy as np
#         from skimage.morphology import medial_axis
#         import cv2
#         import matplotlib.pyplot as plt
#         import threading
#
#         # 确定多边形边界框
#         min_x = min(p[0] for p in polygon)
#         max_x = max(p[0] for p in polygon)
#         min_y = min(p[1] for p in polygon)
#         max_y = max(p[1] for p in polygon)
#
#         # 调整分辨率以获得更准确的中轴线
#         resolution = 1.0  # 每米的像素数
#         width = int((max_x - min_x) * resolution) + 10
#         height = int((max_y - min_y) * resolution) + 10
#
#         # 创建二值图像
#         img = np.zeros((height, width), dtype=np.uint8)
#
#         # 将多边形转换为图像坐标
#         polygon_img = []
#         for p in polygon:
#             x = int((p[0] - min_x) * resolution) + 5
#             y = int((p[1] - min_y) * resolution) + 5
#             polygon_img.append([x, y])
#
#         # 填充多边形
#         cv2.fillPoly(img, [np.array(polygon_img, dtype=np.int32)], 255)
#
#         # 计算中轴线
#         skel, dist = medial_axis(img, return_distance=True)
#
#         # 提取中轴线点
#         centerline_img = np.argwhere(skel)
#
#         # 将图像坐标转换回实际坐标
#         centerline_raw = []
#         for p in centerline_img:
#             y, x = p
#             real_x = (x - 5) / resolution + min_x
#             real_y = (y - 5) / resolution + min_y
#             centerline_raw.append([real_x, real_y])
#
#         # 对中轴线点进行排序，使其按照路径的连续性排列
#         if len(centerline_raw) > 1:
#             centerline = self._sort_centerline_points(centerline_raw)
#         else:
#             centerline = centerline_raw
#
#         # 计算每个点的航向角
#         headings = self._calculate_headings(centerline)
#
#         return np.array(centerline), np.array(headings)
#
#     def _sort_centerline_points(self, points):
#         """
#         对中轴线点进行排序，使其按照路径的连续性排列
#
#         参数:
#         - points: 未排序的中轴线点列表
#
#         返回:
#         - sorted_points: 排序后的点列表
#         """
#         if len(points) <= 1:
#             return points
#
#         # 使用贪心算法，从一个端点开始，依次选择最近的点
#         sorted_points = [points[0]]
#         remaining_points = points[1:]
#
#         while remaining_points:
#             current_point = sorted_points[-1]
#             # 找到距离当前点最近的点
#             distances = [np.linalg.norm(np.array(p) - np.array(current_point)) for p in remaining_points]
#             nearest_idx = np.argmin(distances)
#
#             # 将最近的点添加到排序列表中
#             sorted_points.append(remaining_points[nearest_idx])
#             remaining_points.pop(nearest_idx)
#
#         return sorted_points
#
#     def _calculate_headings(self, centerline):
#         """
#         计算中心线上每个点的航向角
#
#         参数:
#         - centerline: 排序后的中心线点列表
#
#         返回:
#         - headings: 每个点的航向角列表（弧度）
#         """
#         if len(centerline) <= 1:
#             return [0.0] * len(centerline)
#
#         headings = []
#
#         for i in range(len(centerline)):
#             if i == 0:
#                 # 第一个点：使用当前点到下一个点的方向
#                 dx = centerline[i + 1][0] - centerline[i][0]
#                 dy = centerline[i + 1][1] - centerline[i][1]
#             elif i == len(centerline) - 1:
#                 # 最后一个点：使用前一个点到当前点的方向
#                 dx = centerline[i][0] - centerline[i - 1][0]
#                 dy = centerline[i][1] - centerline[i - 1][1]
#             else:
#                 # 中间点：使用前一个点到下一个点的方向（更平滑）
#                 dx = centerline[i + 1][0] - centerline[i - 1][0]
#                 dy = centerline[i + 1][1] - centerline[i - 1][1]
#
#             # 计算航向角（弧度）
#             heading = np.arctan2(dy, dx)
#             headings.append(heading)
#
#         return headings
#
#     def _create_vehicles_once(self, map: BaseMap, traffic_density: float) -> None:
#         vehicle_num = 0
#
#         # 选择合适的道路作为新车辆生成的候选位置
#         trigger_lanes = self.respawn_lanes
#         potential_vehicle_configs = []
#         for l in trigger_lanes:
#             # 如果该车道在事故车道中，则跳过
#             if hasattr(self.engine, "object_manager") and l in self.engine.object_manager.accident_lanes:
#                 continue
#             potential_vehicle_configs += self._propose_vehicle_configs(l)
#
#         # 生成数量
#         total_length = sum([lane.length for lane in trigger_lanes])
#         total_spawn_points = int(math.floor(total_length / self.VEHICLE_GAP))
#         total_vehicles = int(math.floor(total_spawn_points * traffic_density))
#
#         # Generate vehicles!
#         self.np_random.shuffle(potential_vehicle_configs)
#         selected = potential_vehicle_configs[:min(total_vehicles, len(potential_vehicle_configs))]
#         selected = [{'spawn_lane_index': '232091', 'spawn_longitude': 0, 'enable_reverse': False}]
#         from metadrive.policy.idm_policy import IDMPolicy, MyPolicy, MyEgoPolicy
#         from metadrive.policy.expert_policy import ExpertPolicy
#         # print("===== We are initializing {} vehicles =====".format(len(selected)))
#         # print("Current seed: ", self.engine.global_random_seed)
#         for v_config in selected:
#             vehicle_type = self.random_vehicle_type()
#             v_config.update(self.engine.global_config["traffic_vehicle_config"])
#             random_v = self.spawn_object(vehicle_type, vehicle_config=v_config)
#             self.add_policy(random_v.id, MyEgoPolicy, random_v, self.generate_seed())
#             self._traffic_vehicles.append(random_v)
#
#         # 因为原本的地图是结构化的，是block组成的，所以trigger模式可以实现，当车辆到trigger车辆所在block的前一个block时，触发该block的车辆生成
#         # 但是真实的地图没有block的概念，所以没法简单实现。若是直接使用respawn_lanes作为触发道路，那么并列车道的还是不会被触发
#         # TODO
#         # trigger_road = block.pre_block_socket.positive_road
#         # block_vehicles = BlockVehicles(trigger_road=trigger_road, vehicles=vehicles_on_block)
#
#         # self.block_triggered_vehicles.append(block_vehicles)
#         # vehicle_num += len(vehicles_on_block)
#         # self.block_triggered_vehicles.reverse()
#
#     def _get_available_respawn_lanes(self, map: BaseMap) -> list:
#         """
#         Used to find some respawn lanes
#         :param map: select spawn lanes from this map
#         :return: respawn_lanes
#         """
#         respawn_lanes = []
#         respawn_roads = []
#         for block in map.blocks:
#             roads = block.get_respawn_roads()
#             for road in roads:
#                 if road in respawn_roads:  # 如果存在，则将其移除（这意味着该道路在多个区块中可能被重复引用）。如果不存在，则将其添加。
#                     respawn_roads.remove(road)
#                 else:
#                     respawn_roads.append(road)
#         for road in respawn_roads:
#             respawn_lanes += road.get_lanes(map.road_network)
#         return respawn_lanes
#
#     def random_vehicle_type(self):
#         from metadrive.component.vehicle.vehicle_type import random_vehicle_type
#         vehicle_type = random_vehicle_type(self.np_random, [0.2, 0.3, 0.3, 0.2, 0.0])
#         return vehicle_type
#
#     def destroy(self) -> None:
#         """
#         Destory func, release resource
#         :return: None
#         """
#         self.clear_objects([v.id for v in self._traffic_vehicles])
#         self._traffic_vehicles = []
#         # current map
#
#         # traffic vehicle list
#         self._traffic_vehicles = None
#         self.block_triggered_vehicles = None
#
#         # traffic property
#         self.mode = None
#         self.random_traffic = None
#         self.density = None
#
#     def __del__(self):
#         logger.debug("{} is destroyed".format(self.__class__.__name__))
#
#     def __repr__(self):
#         return self._traffic_vehicles.__repr__()
#
#     @property
#     def vehicles(self):
#         return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())
#
#     @property
#     def traffic_vehicles(self):
#         return list(self._traffic_vehicles)
#
#     def seed(self, random_seed):
#         if not self.random_traffic:
#             super(GenTrafficManager, self).seed(random_seed)
#
#     @property
#     def current_map(self):
#         return self.engine.map_manager.current_map
#
#     def get_state(self):
#         ret = super(GenTrafficManager, self).get_state()
#         ret["_traffic_vehicles"] = [v.name for v in self._traffic_vehicles]
#         flat = []
#         for b_v in self.block_triggered_vehicles:
#             flat.append((b_v.trigger_road.start_node, b_v.trigger_road.end_node, b_v.vehicles))
#         ret["block_triggered_vehicles"] = flat
#         return ret
#
#     def set_state(self, state: dict, old_name_to_current=None):
#         super(GenTrafficManager, self).set_state(state, old_name_to_current)
#         self._traffic_vehicles = list(
#             self.get_objects([old_name_to_current[name] for name in state["_traffic_vehicles"]]).values()
#         )
#         self.block_triggered_vehicles = [
#             BlockVehicles(trigger_road=Road(s, e), vehicles=[old_name_to_current[name] for name in v])
#             for s, e, v in state["block_triggered_vehicles"]
#         ]
#
