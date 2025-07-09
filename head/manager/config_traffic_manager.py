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
    # 1.获取当前引擎对象`engine`，以及当前模式`mode`。
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
        if self.global_config['lane_num'] == 3:
            grid_choices = np.arange(20, 40, 8)
            SPEED_DIFFERENCE = [15.0, 10.0, 18.75]
            long = self.VEHICLE_GAP * grid_choices / 3
        else:
            SPEED_DIFFERENCE = [5.0, 20.0, 30.0, 40.0, 45.0]
            long = self.VEHICLE_GAP * grid_choices / 4

        for long in long.tolist():
            lane = self.respawn_lanes[np.random.choice(self.global_config['lane_num'])]
            vehicle_type = self.random_vehicle_type()
            traffic_v_config = {"spawn_lane_index": lane.index, "spawn_longitude": long, "spawn_lateral": 0}

            # print(lane.index)
            traffic_v_config.update(self.engine.global_config["traffic_vehicle_config"])
            random_v = self.spawn_object(vehicle_type, vehicle_config=traffic_v_config)
            from head.policy.idm_policy_include_pedestrian import IDMPolicyIncluedPedestrain
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
            # print("Current seed: ", self.engine.global_random_seed)
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
