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
import matplotlib.pyplot as plt
from metadrive.constants import TerminationState

class RealScenarioEnv(ScenarioEnv):
    def __init__(self, config: Optional[Union[Dict, Config]] = None):
        super(RealScenarioEnv, self).__init__(config)
        self.head_renderer = None
        self.head_renderer = None

    @classmethod
    def default_config(cls) -> Config:
        config = ScenarioEnv.default_config()
        config.update(
            {
                "frame_skip": 5,
                "frame_stack": 3,
                "post_stack": 5,
                "norm_pixel": True,
                "resolution_size": 200,
                "distance": 45,
            }
        )
        return config

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, truncate, terminate, info = super(ScenarioEnv, self).step(actions)
        # 添加位置信息
        agent = self.agents.get('default_agent', None)
        if agent:
            info["position"] = np.array(agent.position)

        return obs, reward, truncate, terminate, info


    def reset(self, *args, **kwargs):
        obs = super(RealScenarioEnv, self).reset(*args, **kwargs)
        self.head_renderer = HeadTopDownRenderer(self)
        if self.head_renderer is not None:
            self.head_renderer.reset()

        return obs

    def done_function(self, vehicle_id: str):
        vehicle = self.agents[vehicle_id]
        done = False
        max_step = False
        if self.config["horizon"] is not None:
            horizon =  self.config["horizon"]
            max_scenario_length = self.engine.data_manager.current_scenario_length
            max_step_num = min(horizon,max_scenario_length )
            max_step = self.episode_lengths[vehicle_id] >= max_step_num

        done_info = {
            TerminationState.CRASH_VEHICLE: False,
            TerminationState.CRASH_OBJECT: False,
            TerminationState.CRASH_BUILDING: False,
            TerminationState.CRASH_HUMAN: False,
            TerminationState.CRASH_SIDEWALK: False,
            TerminationState.OUT_OF_ROAD: False,
            TerminationState.SUCCESS: False,
            TerminationState.MAX_STEP: max_step,
            TerminationState.ENV_SEED: self.current_seed,
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

        def msg(reason):
            return "Episode ended! Scenario Index: {} Scenario id: {} Reason: {}.".format(
                self.current_seed, self.engine.data_manager.current_scenario_id, reason
            )

        if done_info[TerminationState.SUCCESS]:
            done = True
            self.logger.info(msg("arrive_dest"), extra={"log_once": True})
        elif done_info[TerminationState.OUT_OF_ROAD]:
            done = True
            self.logger.info(msg("out_of_road"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_HUMAN] and self.config["crash_human_done"]:
            done = True
            self.logger.info(msg("crash human"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_VEHICLE] and self.config["crash_vehicle_done"]:
            done = True
            self.logger.info(msg("crash vehicle"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_OBJECT] and self.config["crash_object_done"]:
            done = True
            self.logger.info(msg("crash object"), extra={"log_once": True})
        elif done_info[TerminationState.CRASH_BUILDING] and self.config["crash_object_done"]:
            done = True
            self.logger.info(msg("crash building"), extra={"log_once": True})
        elif done_info[TerminationState.MAX_STEP]:
            if self.config["truncate_as_terminate"]:
                done = True
            self.logger.info(msg("max step"), extra={"log_once": True})
        elif self.config["allowed_more_steps"] and self.episode_lengths[vehicle_id] >= \
            self.engine.data_manager.current_scenario_length + self.config["allowed_more_steps"]:
            if self.config["truncate_as_terminate"]:
                done = True
            done_info[TerminationState.MAX_STEP] = True
            self.logger.info(msg("more step than original episode"), extra={"log_once": True})

        # log data to curriculum manager
        self.engine.curriculum_manager.log_episode(
            done_info[TerminationState.SUCCESS], vehicle.navigation.route_completion
        )

        return done, done_info