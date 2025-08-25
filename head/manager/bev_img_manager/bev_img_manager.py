import os
import sys
import numpy as np
import pygame
from pygame import gfxdraw
from typing import Tuple, List, Optional, Dict, Any
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.component.map.base_map import BaseMap
from metadrive.constants import Decoration, DEFAULT_AGENT, EDITION
from metadrive.engine.engine_utils import get_engine
from metadrive.utils.math import norm
from metadrive.utils import import_pygame
from metadrive.obs.top_down_obs_impl import WorldSurface, ObservationWindow, COLOR_BLACK, \
    ObjectGraphics, LaneGraphics
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.utils import is_map_related_instance
from metadrive.constants import TopDownSemanticColor, MetaDriveType, PGDrivableAreaProperty
from metadrive.obs.top_down_obs_impl import WorldSurface, ObservationWindow, COLOR_BLACK, \
    ObjectGraphics, LaneGraphics
from metadrive.obs.top_down_obs_impl import WorldSurface, ObjectGraphics, LaneGraphics, history_object


pygame, gfxdraw = import_pygame()

# 常量定义

COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)



class BEVRenderer:
    """鸟瞰图(BEV)渲染器，基于原始TopDownObservation代码重构"""
    # 渲染分辨率配置
    RESOLUTION = (200, 200)  # 默认渲染分辨率 (宽, 高)
    MAP_RESOLUTION = (2000, 2000)  # 地图画布分辨率

    def __init__(
            self,
            engine=None,
            clip_rgb: bool = True,
            resolution: Tuple[int, int] = None,
            max_distance: float = 50.0,
            onscreen: bool = False,
    ):
        """
        初始化BEV渲染器

        :param clip_rgb: 是否将RGB值裁剪到[0,1]范围
        :param resolution: 渲染分辨率 (宽, 高)
        :param max_distance: 最大渲染距离 (米)
        :param onscreen: 是否在屏幕上显示渲染结果
        """
        self.engine = engine
        self.resolution = resolution or self.RESOLUTION
        self.norm_pixel = clip_rgb
        self.num_stacks = 3

        # self.obs_shape = (64, 64)
        self.obs_shape = self.resolution

        self.pygame, _ = import_pygame()

        self.onscreen = onscreen
        main_window_position = (0, 0)

        self._center_pos = None  # automatically change, don't set the value
        self._should_draw_map = True
        self._canvas_to_display_scaling = 0.0
        self.max_distance = max_distance

        # scene
        self.road_network = None
        # self.engine = None

        # initialize
        pygame.init()
        pygame.display.set_caption(EDITION + " (Top-down)")
        # main_window_position means the left upper location.
        os.environ['SDL_VIDEO_WINDOW_POS'] = '{},{}' \
            .format(main_window_position[0] - self.resolution[0], main_window_position[1])
        # Used for display only!
        self.screen = pygame.display.set_mode(
            (self.resolution[0] * 2, self.resolution[1] * 2)
        ) if self.onscreen else None
        self.obs_window = None
        self.canvas_runtime = None
        self.canvas_background = None

        # canvas
        self.init_canvas()
        self.init_obs_window()


    def init_obs_window(self):
        self.obs_window = ObservationWindow((self.max_distance, self.max_distance), self.resolution)


    def init_canvas(self):
        self.canvas_runtime = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))
        self.canvas_background = WorldSurface(self.MAP_RESOLUTION, 0, pygame.Surface(self.MAP_RESOLUTION))

    def get_screenshot(self, name="screenshot.png"):
        pygame.image.save(self.screen, name)

    def draw_map(self) -> pygame.Surface:
        """
        :return: a big map surface, clip  and rotate to use a piece of it
        """

        # Setup the maximize size of the canvas
        # scaling and center can be easily found by bounding box

        # TODO(pzh) We can reuse function draw_top_down_map here!

        b_box = self.road_network.get_bounding_box()
        self.canvas_background.fill(COLOR_WHITE)
        self.canvas_runtime.fill(COLOR_WHITE)
        self.canvas_background.set_colorkey(self.canvas_background.BLACK)
        x_len = b_box[1] - b_box[0]
        y_len = b_box[3] - b_box[2]
        max_len = max(x_len, y_len) + 20  # Add more 20 meters
        scaling = self.MAP_RESOLUTION[1] / max_len - 0.1
        assert scaling > 0

        # real-world distance * scaling = pixel in canvas
        self.canvas_background.scaling = scaling
        self.canvas_runtime.scaling = scaling
        # self._scaling = scaling

        centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
        # self._center_pos = centering_pos
        self.canvas_runtime.move_display_window_to(centering_pos)

        self.canvas_background.move_display_window_to(centering_pos)
        current_map = self.engine.map_manager.current_map
        if isinstance(current_map, ScenarioMap):
            line_sample_interval = 2
            all_lanes = current_map.get_map_features(line_sample_interval)
            for id, data in all_lanes.items():
                if ScenarioDescription.POLYLINE not in data:
                    continue
                LaneGraphics.display_scenario_line(
                    data["polyline"], data["type"],self.canvas_background, line_sample_interval=line_sample_interval
                )
        else:

            for _from in self.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in self.road_network.graph[_from].keys():
                    for l in self.road_network.graph[_from][_to]:
                        two_side = True if l is self.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.LANE_LINE_WIDTH = 2.0
                        LaneGraphics.display(l, self.canvas_background, two_side)

        self.obs_window.reset(self.canvas_runtime)

        self._should_draw_map = False

    def draw_scene(self):
        # Set the active area that can be modify to accelerate
        assert len(self.engine.agents) == 1, "Don't support multi-agent top-down observation yet!"
        vehicle = self.engine.agents[DEFAULT_AGENT]
        pos = self.canvas_runtime.pos2pix(*vehicle.position)
        clip_size = (int(self.obs_window.get_size()[0] * 1.1), int(self.obs_window.get_size()[0] * 1.1))
        self.canvas_runtime.set_clip((pos[0] - clip_size[0] / 2, pos[1] - clip_size[1] / 2, clip_size[0], clip_size[1]))
        self.canvas_runtime.fill(COLOR_BLACK)
        self.canvas_runtime.blit(self.canvas_background, (0, 0))

        # Draw vehicles
        # TODO PZH: I hate computing these in pygame-related code!!!
        ego_heading = vehicle.heading_theta
        ego_heading = ego_heading if abs(ego_heading) > 2 * np.pi / 180 else 0

        ObjectGraphics.display(
            object=vehicle, surface=self.canvas_runtime, heading=ego_heading, color=ObjectGraphics.GREEN
        )
        objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))
        frame_objects = []
        for name, obj in objects.items():
            if obj.class_name == 'DefaultVehicle':
                obj_type =DEFAULT_AGENT
            else:
                obj_type = obj.metadrive_type if hasattr(obj, "metadrive_type") else MetaDriveType.OTHER

            frame_objects.append(
                history_object(
                    name=name,
                    type=obj_type,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=obj.top_down_color,
                    done=False
                )
            )
        for v in frame_objects:

            h = v.heading_theta
            c = v.color
            if (v.type == MetaDriveType.VEHICLE or v.type == MetaDriveType.PEDESTRIAN or v.type == MetaDriveType.CYCLIST
                    or v.type == MetaDriveType.OTHER):
                c = COLOR_BLUE
            elif v.type == DEFAULT_AGENT:
                c = COLOR_GREEN
            h = h if abs(h) > 2 * np.pi / 180 else 0
            ObjectGraphics.display(object=v, surface=self.canvas_runtime, heading=h, color=c)

        # Prepare a runtime canvas for rotation
        return self.obs_window.render(canvas=self.canvas_runtime, position=pos, heading=-vehicle.heading_theta)

    @staticmethod
    def blit_rotate(
        surf: pygame.SurfaceType,
        image: pygame.SurfaceType,
        pos,
        angle: float,
    ) -> Tuple:
        """Many thanks to https://stackoverflow.com/a/54714144."""
        # calculate the axis aligned bounding box of the rotated image
        w, h = image.get_size()
        box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
        box_rotate = [p.rotate(angle) for p in box]
        min_box = (min(box_rotate, key=lambda p: p[0])[0], min(box_rotate, key=lambda p: p[1])[1])
        max_box = (max(box_rotate, key=lambda p: p[0])[0], max(box_rotate, key=lambda p: p[1])[1])

        # calculate the translation of the pivot
        origin_pos = w / 2, h / 2
        pivot = pygame.math.Vector2(origin_pos[0], -origin_pos[1])
        pivot_rotate = pivot.rotate(angle)
        pivot_move = pivot_rotate - pivot

        # calculate the upper left origin of the rotated image
        origin = (
            pos[0] - origin_pos[0] + min_box[0] - pivot_move[0], pos[1] - origin_pos[1] - max_box[1] + pivot_move[1]
        )
        # get a rotated image
        # rotated_image = pygame.transform.rotate(image, angle)
        rotated_image = pygame.transform.rotozoom(image, angle, 1.0)
        # rotate and blit the image
        surf.blit(rotated_image, origin)
        return origin

    def get_observation_window(self):
        return self.obs_window.get_observation_window()

    def get_screen_window(self):
        return self.obs_window.get_screen_window()

    def render(self) -> np.ndarray:
        if self.onscreen:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        sys.exit()

        if self._should_draw_map:
            self.draw_map()

        self.draw_scene()

        if self.onscreen:
            self.screen.fill(COLOR_BLACK)
            screen = self.obs_window.get_screen_window()
            if screen.get_size() == self.screen.get_size():
                self.screen.blit(screen, (0, 0))
            else:
                pygame.transform.smoothscale(self.obs_window.get_screen_window(), self.screen.get_size(), self.screen)
            pygame.display.flip()


    def get_bev_render(self):
        self.render()
        surface = self.get_observation_window()
        img = self.pygame.surfarray.array3d(surface)
        if self.norm_pixel:
            img = img.astype(np.float32) / 255
        else:
            img = img.astype(np.uint8)
        return np.transpose(img, (1, 0, 2))


    @staticmethod
    def destroy():
        """清理资源"""
        pygame.quit()
        pygame.display.quit()

    def reset(self, road_network):
        self.road_network = road_network
        self._should_draw_map = True