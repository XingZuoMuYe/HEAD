import random

import numpy as np

from metadrive.manager import BaseManager

from metadrive.component.traffic_participants.pedestrian import Pedestrian
import math


class Pedestrian_Manager(BaseManager):
    def __init__(self):
        super(Pedestrian_Manager, self).__init__()
        self.generated_p = []
        self.recycle_ts = 1000
        if self.engine.global_config['lane_num'] == 3:
            self.pedestrian_num = 2
        else:
            self.pedestrian_num = 5
        self.generate_ts = np.zeros(self.pedestrian_num)
        self.id_x = 0

    def before_step(self):
        self.generate_ts = np.random.choice(np.arange(0, 5), self.pedestrian_num, replace=False)  # 0到5s随机

        if self.engine.global_config['lane_num'] == 3:
            pos_x = [60, 80]
            speed_y = [0.8, 1.4]
        else:
            pos_x = [100, 120, 140, 160, 180]
            speed_y = [0.8, 1.4, 0.6, 1.4, 0.7]

        if self.episode_step in self.generate_ts:
            self.generated_p.append(
                self.spawn_object(Pedestrian, position=[pos_x[self.id_x], -1.75],
                                  heading_theta=math.pi / 2, random_seed=1))
            target_speed = speed_y[self.id_x]
            self.generated_p[-1].set_velocity([1, 0], target_speed, in_local_frame=True)
            self.generated_p[-1].target_speed = target_speed
            self.id_x += 1
            if self.id_x == self.pedestrian_num:
                self.id_x = 0

    def after_step(self):

        keys = list(self.engine.get_objects().keys())
        pedestrian_actor = []
        vehicle_actor = []
        for i in range(len(keys)):
            if self.engine.get_objects()[keys[i]].metadrive_type == "PEDESTRIAN":
                pedestrian_actor.append(self.engine.get_objects()[keys[i]])
            if self.engine.get_objects()[keys[i]].metadrive_type == "VEHICLE":
                vehicle_actor.append(self.engine.get_objects()[keys[i]])

        for p in pedestrian_actor:
            p_x = p.position[0]
            p_y = p.position[1]
            if p_y > 12.25:
                self.clear_objects([p.id])
            for v in vehicle_actor:
                ve_x = v.position[0]
                ve_y = v.position[1]

                walking_cross = ((ve_x - 10.5) <= p_x <= (ve_x + 10.5)) and (0 <= (ve_y - p_y) <= 5.0)

                if walking_cross:
                    p.set_velocity([0, 0], 0.0, in_local_frame=True)
                else:
                    p.set_heading_theta(math.pi / 2)
                    p.set_velocity([1, 0], p.target_speed, in_local_frame=True)

    # def reset(self):
    #     self.clear_objects([p.id for p in self.generated_p])
    #     self.generated_p = []
    #     self.id_x = 0
    #     self.generate_ts = np.zeros(self.pedestrian_num)
