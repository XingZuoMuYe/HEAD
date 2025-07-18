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


class RealScenarioEnv(ScenarioEnv):
    def __init__(self, config: Optional[Union[Dict, Config]] = None):
        super(RealScenarioEnv, self).__init__(config)
        self.head_renderer = None

    def step(self, actions: Union[Union[np.ndarray, list], Dict[AnyStr, Union[list, np.ndarray]], int]):
        obs, reward, truncate, terminate, info = super(ScenarioEnv, self).step(actions)

        return obs, reward, truncate, terminate, info

    def reset(self, *args, **kwargs):
        obs = super(RealScenarioEnv, self).reset(*args, **kwargs)
        self.head_renderer = HeadTopDownRenderer(self)
        if self.head_renderer is not None:
            self.head_renderer.reset()
        return obs