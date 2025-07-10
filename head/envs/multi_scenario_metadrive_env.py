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


class MultiScenario(MetaDriveEnv):
    def __init__(self, config: Optional[Union[Dict, Config]] = None):
        super(MultiScenario, self).__init__(config)
        self.scenario = None
        self.step_collision_list = []
        self.eps_collision_list = []

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, truncate, terminate, info = super(MetaDriveEnv, self).step(actions)
        info.update({"position": np.array(self.agents['default_agent'].position),
                     "global_path": self.agents['default_agent'].global_path,
                     "speed": self.agents['default_agent'].velocity,
                    "lat_dis": self.engine.agents['default_agent'].cur_d,
                    "yaw": self.agents['default_agent'].heading,
                    "safety_intervene": self.agents['default_agent'].safety_intervene}
                    )
        # info.update({"position": np.array(self.agents['default_agent'].position),
        #              "global_path": self.agents['default_agent'].global_path,
        #              "speed": self.agents['default_agent'].velocity,
        #             "lat_dis": self.renderer.agents['default_agent'].cur_d,
        #             "yaw": self.agents['default_agent'].heading}
        #             )
        if info['crash_object'] or info['crash_vehicle'] or info['crash_human'] or info['crash_sidewalk']:
            self.step_collision_list.append(1)
        else:
            self.step_collision_list.append(0)

        return obs, reward, truncate, terminate, info

    def reset(self, *args, **kwargs):

        if 1 in self.step_collision_list:
            self.eps_collision_list.append(1)
        else:
            self.eps_collision_list.append(0)
        collision_rate = sum(self.eps_collision_list) / len(self.eps_collision_list)
        print('collision_rate = ', collision_rate)
        self.step_collision_list = []
        return super(MetaDriveEnv, self).reset(*args, **kwargs)