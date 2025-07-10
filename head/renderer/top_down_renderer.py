import copy
from metadrive.engine.logger import get_logger

from metadrive.utils import generate_gif
import math
from collections import deque
from typing import Optional, Union, Iterable
import pygame.gfxdraw
import colorsys  # 用于 HSL 颜色空间转换
import numpy as np

from metadrive.component.map.scenario_map import ScenarioMap
from metadrive.constants import Decoration, TARGET_VEHICLES
from metadrive.constants import TopDownSemanticColor, MetaDriveType, PGDrivableAreaProperty
from metadrive.obs.top_down_obs_impl import WorldSurface, ObjectGraphics, LaneGraphics, history_object
from metadrive.scenario.scenario_description import ScenarioDescription
from metadrive.utils.utils import import_pygame
from metadrive.utils.utils import is_map_related_instance

pygame, gfxdraw = import_pygame()

color_white = (255, 255, 255)

font_path = "/usr/share/fonts/wryhBold.ttf"  # 替换为你的字体文件路径



def draw_thick_line(surface, start_pos, end_pos, color, width):
    """
    绘制带有透明度和加粗的线条，模拟加粗效果。

    参数:
    - surface: 要绘制的surface
    - start_pos: 线条起点
    - end_pos: 线条终点
    - color: RGBA颜色
    - width: 线条宽度
    """
    # 计算线条的单位向量，方向从 start_pos 指向 end_pos
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    length = (dx**2 + dy**2) ** 0.5

    # 单位向量的方向
    if length != 0:
        dx /= length
        dy /= length

    # 计算垂直方向的偏移量
        offset_x = dy * (width // 2) # 垂直偏移
        offset_y = -dx * (width // 2) # 垂直偏移

    # 绘制多个线条，模拟加粗效果
    for i in range(-width // 2, width // 2 + 1):
    # 偏移后的起点和终点
        offset_start_pos = (start_pos[0] + offset_x * i, start_pos[1] + offset_y * i)
        offset_end_pos = (end_pos[0] + offset_x * i, end_pos[1] + offset_y * i)

    # 绘制线条
        pygame.gfxdraw.line(surface, int(offset_start_pos[0]), int(offset_start_pos[1]),
        int(offset_end_pos[0]), int(offset_end_pos[1]), color)

def draw_top_down_map_native(
        map,
        semantic_map=True,
        return_surface=False,
        film_size=(2000, 2000),
        scaling=None,
        semantic_broken_line=True
) -> Optional[Union[np.ndarray, pygame.Surface]]:
    """
    Draw the top_down map on a pygame surface
    Args:
        map: MetaDrive.BaseMap instance
        semantic_map: return semantic map
        return_surface: Return the pygame.Surface in fime_size instead of cv2.image
        film_size: The size of the film to draw the map
        scaling: the scaling factor, how many pixels per meter
        semantic_broken_line: Draw broken line on semantic map

    Returns: cv2.image or pygame.Surface

    """
    surface = WorldSurface(film_size, 0, pygame.Surface(film_size))
    if map is None:
        surface.move_display_window_to([0, 0])
        surface.fill([230, 230, 230])
        return surface if return_surface else WorldSurface.to_cv2_image(surface)

    b_box = map.road_network.get_bounding_box()
    x_len = b_box[1] - b_box[0]
    y_len = b_box[3] - b_box[2]
    max_len = max(x_len, y_len)
    # scaling and center can be easily found by bounding box
    scaling = scaling if scaling is not None else (film_size[1] / max_len - 0.1)
    surface.scaling = scaling
    centering_pos = ((b_box[0] + b_box[1]) / 2, (b_box[2] + b_box[3]) / 2)
    surface.move_display_window_to(centering_pos)
    line_sample_interval = 2

    if semantic_map:
        all_lanes = map.get_map_features(line_sample_interval)

        for obj in all_lanes.values():
            if MetaDriveType.is_lane(obj["type"]):
                pygame.draw.polygon(
                    surface, TopDownSemanticColor.get_color(obj["type"]),
                    [surface.pos2pix(p[0], p[1]) for p in obj["polygon"]]
                )

            elif MetaDriveType.is_road_line(obj["type"]) or MetaDriveType.is_road_boundary_line(obj["type"]):
                if semantic_broken_line and MetaDriveType.is_broken_line(obj["type"]):
                    points_to_skip = math.floor(PGDrivableAreaProperty.STRIPE_LENGTH * 2 / line_sample_interval) * 2
                else:
                    points_to_skip = 1
                for index in range(0, len(obj["polyline"]) - 1, points_to_skip):
                    if index + 1 < len(obj["polyline"]):
                        s_p = obj["polyline"][index]
                        e_p = obj["polyline"][index + 1]
                        pygame.draw.line(
                            surface,
                            TopDownSemanticColor.get_color(obj["type"]),
                            surface.vec2pix([s_p[0], s_p[1]]),
                            surface.vec2pix([e_p[0], e_p[1]]),
                            # max(surface.pix(LaneGraphics.STRIPE_WIDTH),
                            surface.pix(PGDrivableAreaProperty.LANE_LINE_WIDTH) * 2
                        )
    else:
        if isinstance(map, ScenarioMap):
            line_sample_interval = 2
            all_lanes = map.get_map_features(line_sample_interval)
            for id, data in all_lanes.items():
                if ScenarioDescription.POLYLINE not in data:
                    continue
                LaneGraphics.display_scenario_line(
                    data["polyline"], data["type"], surface, line_sample_interval=line_sample_interval
                )
        else:
            for _from in map.road_network.graph.keys():
                decoration = True if _from == Decoration.start else False
                for _to in map.road_network.graph[_from].keys():
                    for l in map.road_network.graph[_from][_to]:
                        two_side = True if l is map.road_network.graph[_from][_to][-1] or decoration else False
                        LaneGraphics.display(l, surface, two_side, use_line_color=True)

    return surface if return_surface else WorldSurface.to_cv2_image(surface)


def draw_top_down_trajectory(
        surface: WorldSurface, episode_data: dict, entry_differ_color=False, exit_differ_color=False, color_list=None
):
    if entry_differ_color or exit_differ_color:
        assert color_list is not None
    color_map = {}
    if not exit_differ_color and not entry_differ_color:
        color_type = 0
    elif exit_differ_color ^ entry_differ_color:
        color_type = 1
    else:
        color_type = 2

    if entry_differ_color:
        # init only once
        if "spawn_roads" in episode_data:
            spawn_roads = episode_data["spawn_roads"]
        else:
            spawn_roads = set()
            first_frame = episode_data["frame"][0]
            for state in first_frame[TARGET_VEHICLES].values():
                spawn_roads.add((state["spawn_road"][0], state["spawn_road"][1]))
        keys = [road[0] for road in list(spawn_roads)]
        keys.sort()
        color_map = {key: color for key, color in zip(keys, color_list)}

    for frame in episode_data["frame"]:
        for k, state, in frame[TARGET_VEHICLES].items():
            if color_type == 0:
                color = color_white
            elif color_type == 1:
                if exit_differ_color:
                    key = state["destination"][1]
                    if key not in color_map:
                        color_map[key] = color_list.pop()
                    color = color_map[key]
                else:
                    color = color_map[state["spawn_road"][0]]
            else:
                key_1 = state["spawn_road"][0]
                key_2 = state["destination"][1]
                if key_1 not in color_map:
                    color_map[key_1] = dict()
                if key_2 not in color_map[key_1]:
                    color_map[key_1][key_2] = color_list.pop()
                color = color_map[key_1][key_2]
            start = state["position"]
            pygame.draw.circle(surface, color, surface.pos2pix(start[0], start[1]), 1)
    for step, frame in enumerate(episode_data["frame"]):
        for k, state in frame[TARGET_VEHICLES].items():
            if not state["done"]:
                continue
            start = state["position"]
            if state["done"]:
                pygame.draw.circle(surface, (0, 0, 0), surface.pos2pix(start[0], start[1]), 5)
    return surface

class TopDownRenderer:
    def __init__(
            self,
            film_size=(2000, 2000),  # draw map in size = film_size/scaling. By default, it is set to 400m
            scaling=5,  # None for auto-scale
            screen_size=(800, 800),
            num_stack=15,
            history_smooth=0,
            show_agent_name=False,
            show_sqt_result = True,
            show_plan_traj=False,
            camera_position=None,
            target_agent_heading_up=False,
            target_vehicle_heading_up=None,
            draw_target_vehicle_trajectory=False,
            semantic_map=False,
            semantic_broken_line=True,
            draw_contour=True,
            window=True,
            screen_record=False,
    ):
        """
        Launch a top-down renderer for current episode. Usually, it is launched by env.render(mode="topdown") and will
        be closed when next env.reset() is called or next episode starts.
        Args:
            film_size: The size of the film used to draw the map. The unit is pixel. It should cover the whole map to
            ensure it is complete in the rendered result. It works with the argument scaling to select the region
            to draw. For example, (2000, 2000) film size with scaling=5 can draw any maps whose width and height
            less than 2000/5 = 400 meters.

            scaling: The scaling determines how many pixels are used to draw one meter.

            screen_size: The size of the window popped up. It shows a region with width and length = screen_size/scaling

            num_stack: How many history steps to keep. History trajectory will show in faded color. It should be > 1

            history_smooth: Smoothing the trajectory by drawing less history positions. This value determines the sample
            rate. By default, this value is 0, meaning positions in previous num_stack steps will be shown.

            show_agent_name: Draw the name of the agent.

            camera_position: Set the (x,y) position of the top_down camera. If it is not specified, the camera will move
            with the ego car.

            target_agent_heading_up: Whether to rotate the camera according to the ego agent's heading. When enabled,
            The ego car always faces upwards.

            target_vehicle_heading_up: Deprecated, use target_agent_heading_up instead!

            draw_target_vehicle_trajectory: Whether to draw the ego car's whole trajectory without faded color

            semantic_map: Whether to draw semantic color for each object. The color scheme is in TopDownSemanticColor.

            semantic_broken_line: Whether to draw broken line for semantic map

            draw_contour: Whether to draw a counter for objects

            window: Whether to pop up the window. Setting it to 'False' enables off-screen rendering

            screen_record: Whether to record the episode. The recorded result can be accessed by
            env.top_down_renderer.screen_frames or env.top_down_renderer.generate_gif(file_name, fps)
        """
        # doc-end
        # LQY: do not delete the above line !!!!!

        # Setup some useful flags
        self.pygame_font_small = None
        self.text_srq_value = None
        self.logger = get_logger()
        if num_stack < 1:
            self.logger.warning("num_stack should be greater than 0. Current value: {}. Set to 1".format(num_stack))
            num_stack = 1

        if target_vehicle_heading_up is not None:
            self.logger.warning("target_vehicle_heading_up is deprecated! Use target_agent_heading_up instead!")
            assert target_agent_heading_up is False
            target_agent_heading_up = target_vehicle_heading_up

        self.position = camera_position
        self.target_agent_heading_up = target_agent_heading_up
        self.show_agent_name = show_agent_name
        self.show_plan_traj = show_plan_traj
        self.show_sqt_result =    show_sqt_result
        self.draw_target_vehicle_trajectory = draw_target_vehicle_trajectory
        self.contour = draw_contour
        self.semantic_broken_line = semantic_broken_line
        self.no_window = not window

        if self.show_agent_name or self.show_sqt_result:
            pygame.init()

        self.screen_record = screen_record
        self._screen_frames = []
        self.pygame_font = None
        self.map = self.engine.current_map
        self.stack_frames = deque(maxlen=num_stack)
        self.history_objects = deque(maxlen=num_stack)
        self.history_target_vehicle = []
        self.history_smooth = history_smooth
        # self.current_track_agent = current_track_agent
        if self.target_agent_heading_up:
            assert self.current_track_agent is not None, "Specify which vehicle to track"
        self._text_render_pos = [50, 50]
        self._font_size = 25
        self._text_render_interval = 20
        self.semantic_map = semantic_map
        self.scaling = scaling
        self.film_size = film_size
        self._screen_size = screen_size

        # Setup the canvas
        # (1) background is the underlying layer that draws map.
        # It is fixed and will never change unless the map changes.
        self._background_canvas = draw_top_down_map_native(
            self.map,
            scaling=self.scaling,
            semantic_map=self.semantic_map,
            return_surface=True,
            film_size=self.film_size,
            semantic_broken_line=self.semantic_broken_line
        )

        # (2) frame is a copy of the background so you can draw movable things on it.
        # It is super large as the background.
        self._frame_canvas = self._background_canvas.copy()

        # (3) canvas_rotate is only used when target_vehicle_heading_up=True and is use to center the tracked agent.
        if self.target_agent_heading_up:
            max_screen_size = max(self._screen_size[0], self._screen_size[1])
            self.canvas_rotate = pygame.Surface((max_screen_size * 2, max_screen_size * 2))

        # (4) screen_canvas is a regional surface where only part of the super large background will draw.
        # This will be used to as the final image shown in the screen & saved.
        self._screen_canvas = pygame.Surface(self._screen_size
                                             ) if self.no_window else pygame.display.set_mode(self._screen_size)
        self._screen_canvas.set_alpha(None)
        self._screen_canvas.fill(color_white)

        # Draw
        self.blit()

        # key accept
        self.need_reset = False

    @property
    def screen_canvas(self):
        return self._screen_canvas

    def refresh(self):
        self._frame_canvas.blit(self._background_canvas, (0, 0))
        self.screen_canvas.fill(color_white)

    def render(self, text, to_image=True, *args, **kwargs):
        if "semantic_map" in kwargs:
            self.semantic_map = kwargs["semantic_map"]

        self.need_reset = False
        if not self.no_window:
            key_press = pygame.key.get_pressed()
            if key_press[pygame.K_r]:
                self.need_reset = True

        # Record current target vehicle
        objects = self.engine.get_objects(lambda obj: not is_map_related_instance(obj))
        this_frame_objects = self._append_frame_objects(objects)
        self.history_objects.append(this_frame_objects)

        if self.draw_target_vehicle_trajectory:
            self.history_target_vehicle.append(
                history_object(
                    type=MetaDriveType.VEHICLE,
                    name=self.current_track_agent.name,
                    heading_theta=self.current_track_agent.heading_theta,
                    WIDTH=self.current_track_agent.top_down_width,
                    LENGTH=self.current_track_agent.top_down_length,
                    position=self.current_track_agent.position,
                    color=self.current_track_agent.top_down_color,
                    done=False
                )
            )

        self._handle_event()
        self.refresh()
        self._draw(*args, **kwargs)
        self._add_text(text)
        self.blit()
        ret = self.screen_canvas
        if not self.no_window:
            ret = ret.convert(24)
        ret = WorldSurface.to_cv2_image(ret) if to_image else ret
        if self.screen_record:
            self._screen_frames.append(ret)
        return ret

    def generate_gif(self, gif_name="demo.gif", duration=30):
        return generate_gif(self._screen_frames, gif_name, is_pygame_surface=False, duration=duration)

    def _add_text(self, text: dict):
        if not text:
            return
        if not pygame.get_init():
            pygame.init()
        font2 = pygame.font.SysFont('didot.ttc', 25)
        # pygame do not support multiline text render
        count = 0
        for key, value in text.items():
            line = str(key) + ":" + str(value)
            img2 = font2.render(line, True, (0, 0, 0))
            this_line_pos = copy.copy(self._text_render_pos)
            this_line_pos[-1] += count * self._text_render_interval
            self._screen_canvas.blit(img2, this_line_pos)
            count += 1

    def blit(self):
        if not self.no_window:
            pygame.display.update()

    def close(self):
        self.clear()
        pygame.quit()

    def clear(self):
        # # Reset the super large background
        self._background_canvas = None

        # Reset several useful variables.
        self._frame_canvas = None
        self.canvas_rotate = None

        self.history_objects.clear()
        self.stack_frames.clear()
        self.history_target_vehicle.clear()
        self.screen_frames.clear()

    @property
    def current_track_agent(self):
        return self.engine.current_track_agent

    @staticmethod
    def _append_frame_objects(objects):
        """
        Extract information for drawing objects
        Args:
            objects: list of BaseObject

        Returns: list of history_object

        """
        frame_objects = []
        for name, obj in objects.items():
            frame_objects.append(
                history_object(
                    name=name,
                    type=obj.metadrive_type if hasattr(obj, "metadrive_type") else MetaDriveType.OTHER,
                    heading_theta=obj.heading_theta,
                    WIDTH=obj.top_down_width,
                    LENGTH=obj.top_down_length,
                    position=obj.position,
                    color=obj.top_down_color,
                    done=False
                )
            )
        return frame_objects


    def draw_plan_traj(self, traj, color_set='rainbow'):
        """
        绘制传入的轨迹。
        """
        plan_traj = np.array([traj.x, traj.y])
        for i in range(plan_traj.shape[1]):
            pos = self._frame_canvas.pos2pix(plan_traj[0, i], plan_traj[1, i])
            radius = 4  # 调整圆的半径
            # 每个颜色组件增加亮度
            brightness_increase = 30
            color = (0, 0, 255)
            if color_set == 'rainbow':
                color = (
                    min(255, 255 - i * 5 + brightness_increase),
                    min(255, 50 + i * 5 + brightness_increase),
                    50 + brightness_increase
                )
            elif color == 'blue':
                color = (255, 0, 0)

            pygame.draw.circle(
                surface=self._frame_canvas,
                color=color,
                center=pos,
                radius=radius
            )

    def draw_pos_buffer(self, agent):
        """
        Draw the position buffer of the agent with a softer gradient, using hollow rectangles.
        Rectangles are oriented based on the agent's heading.
        """
        length = 4.0  # Length of the rectangle in world coordinates
        width = 2.8  # Width of the rectangle in world coordinates

        Sid = agent.Sid_info[-1]
        T_s = agent.Sid_info[Sid][0]
        T_e = agent.Sid_info[Sid][1]

        num_positions = T_e - T_s

        for i, pos in enumerate(agent.pos_buffer[1:], start=1):
            # Calculate the gradient color based on the position in the buffer
            # Use HSL color space for smoother transitions
            hue = i / num_positions  # Hue ranges from 0 to 1
            saturation = 0.8  # Keep saturation high for vibrant colors
            lightness = 0.7  # Adjust lightness to make colors softer

            # Convert HSL to RGB
            rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
            color = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))

            # Extract position and heading
            x, y, heading_theta = pos[0], pos[1], pos[2]
            heading_theta = -heading_theta
            # Convert the position to pixel coordinates
            pos_pix = self._frame_canvas.pos2pix(x, y)
            x_pix, y_pix = pos_pix[0], pos_pix[1]

            # Convert length and width from world coordinates to pixel coordinates
            # Assuming pos2pix scales both x and y equally
            length_pix = length * (self._frame_canvas.pos2pix(1, 0)[0] - self._frame_canvas.pos2pix(0, 0)[0])
            width_pix = width * (self._frame_canvas.pos2pix(0, 1)[1] - self._frame_canvas.pos2pix(0, 0)[1])

            # Calculate the half-length and half-width in pixel coordinates
            half_length_pix = length_pix / 2
            half_width_pix = width_pix / 2

            # Define the rectangle's corners relative to the center in pixel coordinates
            corners = [
                (-half_length_pix, -half_width_pix),
                (half_length_pix, -half_width_pix),
                (half_length_pix, half_width_pix),
                (-half_length_pix, half_width_pix)
            ]

            # Rotate the corners based on the heading
            rotated_corners = []
            for corner in corners:
                # Apply rotation
                rotated_x = corner[0] * math.cos(heading_theta) - corner[1] * math.sin(heading_theta)
                rotated_y = corner[0] * math.sin(heading_theta) + corner[1] * math.cos(heading_theta)
                # Translate to the correct position in pixel coordinates
                rotated_corners.append((rotated_x + x_pix, rotated_y + y_pix))

            # Draw the hollow rectangle
            if i % 10 == 0:
                pygame.draw.polygon(
                    self._frame_canvas,
                    color,
                    rotated_corners,
                    width=3  # Set width to 1 for hollow rectangle
                )



    def draw_key_points(self, agent):
        """
        Draw the key points of the agent.
        """
        color = (0, 150, 255)
        radius = 4.0
        for point in agent.key_points:
            pos = self._frame_canvas.pos2pix(point[0], point[1])
            pygame.draw.circle(
                surface=self._frame_canvas,
                color=color,
                center=pos,
                radius=radius
            )


    def draw_time_info(self, agent):
        count = agent.count
        text = 'T = ' + str(round(count * 0.1, 2 ))
        color = (0, 99, 186)
        off_text_t=[-280, 290.0]
        position = self._frame_canvas.pos2pix(*agent.position)
        position_text_t = (position[0] - off_text_t[0], position[1] - off_text_t[1])
        if self.pygame_font is None:
            self.pygame_font = pygame.font.Font(font_path, 10)
        img_text_t = self.pygame_font.render(text, True, color)
        self._frame_canvas.blit(
            source=img_text_t,
            dest=(position_text_t[0], position_text_t[1]),
            # special_flags=pygame.BLEND_RGBA_MULT
        )


    def draw_srq_result(self, ranked_indices_info):

        # 提取前240个点的分数

        v = self.engine.agents['default_agent']
        position = self._frame_canvas.pos2pix(*v.position)
        if self.pygame_font or self.pygame_font_small is None:
            self.pygame_font = pygame.font.Font(font_path, 28)
            self.pygame_font_small = pygame.font.Font(font_path, 20)
        # 获取文本内容

        # 设置抗锯齿和文本颜色
        antialias = True

        # 文字
        text_name = "安全风险量化值："
        color_name = (0, 0, 0)  # 黑色
        off_name=[100.0, -260.0]
        position_name = (position[0] - off_name[0], position[1] - off_name[1])
        img_name = self.pygame_font.render(text_name, antialias, color_name)

        # SRQ值
        if ranked_indices_info['count'] % 10 == 0:
            self.text_srq_value = str(round(ranked_indices_info['SRQ_value'], 2)) + '%'
        if ranked_indices_info['SRQ_value'] > 65:
            color_srq_value = (219, 83, 83)  # 黑色
        else:
            color_srq_value = (0, 0, 0)
        off_srq_value=[-125.0, -260.0]
        position_srq_value = (position[0] - off_srq_value[0], position[1] - off_srq_value[1])
        img_srq_value = self.pygame_font.render(self.text_srq_value, antialias, color_srq_value)

        # T(s)
        text_t = 'T = ' + str(round(ranked_indices_info['count'] * 0.1, 2 ))
        color_text_t = (0, 99, 186)
        off_text_t=[320, 260.0]
        position_text_t = (position[0] - off_text_t[0], position[1] - off_text_t[1])
        img_text_t = self.pygame_font.render(text_t, antialias, color_text_t)


        # img.set_alpha(None)
        self._frame_canvas.blit(
            source=img_name,
            dest=(position_name[0], position_name[1]),
            # special_flags=pygame.BLEND_RGBA_MULT
        )

        self._frame_canvas.blit(
            source=img_srq_value,
            dest=(position_srq_value[0], position_srq_value[1]),
            # special_flags=pygame.BLEND_RGBA_MULT
        )

        self._frame_canvas.blit(
            source=img_text_t,
            dest=(position_text_t[0], position_text_t[1]),
            # special_flags=pygame.BLEND_RGBA_MULT
        )

        if 'collision_risk_flag' in ranked_indices_info:
        # 初始状态（True 表示介入，False 表示未介入）
            safety_constraint_intervened = ranked_indices_info['collision_risk_flag']
            off_sci=[300.0, -261.0]
            position_sci = (position[0] - off_sci[0], position[1] - off_sci[1])
            if safety_constraint_intervened:
                color_rgb = (255, 0, 0)
                pygame.draw.circle(
                    surface=self._frame_canvas,
                    color=color_rgb,
                    center=position_sci,
                    radius=25.0
                )
            else:
                color_rgb = (0, 255, 0)


            if safety_constraint_intervened:
                text_sci = "动态安全约束介入"
                off_text_sci = [374.0, -290.0]
                color_text_sci = (255, 0, 0)
            else:
                text_sci = "正常行驶"
                off_text_sci = [340.0, -290.0]
                color_text_sci = (0, 0, 0)

            position_text_sci = (position[0] - off_text_sci[0], position[1] - off_text_sci[1])

            img_text_sci = self.pygame_font_small.render(text_sci, antialias, color_text_sci)

            # img.set_alpha(None)
            self._frame_canvas.blit(
                source=img_text_sci,
                dest=(position_text_sci[0], position_text_sci[1]),
                # special_flags=pygame.BLEND_RGBA_MULT
            )


        ranked_index = np.array(ranked_indices_info['ranked_indices'])
        ranked_scores = np.array(ranked_indices_info['ranked_scores'])

        ranked_indices_scores = np.vstack((ranked_index, ranked_scores))
        ranked_indices_scores = ranked_indices_scores[ranked_indices_scores[:, 0] <= 239]

        lidar_info = self.engine.managers['agent_manager'].observations
        cloud_points = list(lidar_info.values())[0].cloud_points

        scores = ranked_indices_scores[1, :]
        scores = np.array(scores, dtype=np.float32)
        min_score = np.min(scores)
        max_score = np.max(scores)

        # 归一化分数
        if max_score == min_score:
            normalized_scores = np.zeros_like(scores)
        else:
            normalized_scores = (scores - min_score) / (max_score - min_score)

        def get_color(normal_score):
            # 定义颜色区间和对应的颜色
            color_ranges = [
                (0.0, (0, 255, 255)),  # 蓝色
                (0.5, (0, 255, 0)),  # 绿色
                (0.6, (255, 165, 0)),  # 橙色
                (0.7, (255, 255, 0)),  # 黄色
                (0.8, (255, 0, 0)),  # 红色
                (1.0, (255, 0, 0))  # 红色
            ]

            # 遍历颜色区间，找到当前 normal_score 所在的区间
            for i in range(len(color_ranges) - 1):
                start_score, start_color = color_ranges[i]
                end_score, end_color = color_ranges[i + 1]

                if start_score <= normal_score < end_score:
                    # 计算插值比例
                    ratio = (normal_score - start_score) / (end_score - start_score)
                    # 插值计算 RGB 分量
                    r = int(start_color[0] + ratio * (end_color[0] - start_color[0]))
                    g = int(start_color[1] + ratio * (end_color[1] - start_color[1]))
                    b = int(start_color[2] + ratio * (end_color[2] - start_color[2]))
                    return (r, g, b)

            # 如果超出范围，返回默认颜色（蓝色）
            return (0, 0, 255)


        for i in range(240):
            # 计算半径（0~50米）

            # 计算角度（转换为数学坐标系中的角度）
            theta_deg = math.degrees(v.heading_theta) + i * 1.5  # 转换为以x轴右侧为0度，逆时针为正的坐标系
            theta_rad = math.radians(theta_deg)

            # 计算笛卡尔坐标
            x = cloud_points[i]  *  50.0 *  math.cos(theta_rad) + v.position[0]
            y = cloud_points[i] *  50.0 *  math.sin(theta_rad)  + v.position[1]

            # 获取颜色
            norm_score = normalized_scores[i]
            color_rgb = get_color(norm_score)

            # 绘制雷达点
            pos = self._frame_canvas.pos2pix(x, y)
            # if cloud_points[i] != 1:
            if True:
                if cloud_points[i] != 1:
                    rad = 8
                else:
                    rad = 2
                pygame.draw.circle(
                    surface=self._frame_canvas,
                    color=color_rgb,
                    center=pos,
                    radius=rad
                )

                target_pos = self._frame_canvas.pos2pix(v.position[0], v.position[1])

                # 设置线条颜色和透明度
                alpha = 20 # 透明度（0-255，0为完全透明，255为完全不透明）
                color_rgba = color_rgb + (alpha,)  # 添加透明度通道
                # 绘制带有透明度的线条

                draw_thick_line(self._frame_canvas, target_pos, pos, color_rgba, 10)  # 设置线条宽度为 5


    def _draw(self, *args, **kwargs):
        """
        This is the core function to process the
        """
        if len(self.history_objects) == 0:
            return
        #
        v = self.engine.agents['default_agent']

        if hasattr(v, 'ranked_indices_info'):
            self.draw_srq_result(v.ranked_indices_info)

        if self.show_plan_traj and hasattr(v, 'plan_traj'):
            self.draw_plan_traj(v.plan_traj)

        if self.show_plan_traj and hasattr(v, 'pre_plan_traj'):
            self.draw_plan_traj(v.pre_plan_traj, 'blue')

        if self.show_plan_traj and hasattr(v, 'safe_plan_traj'):
            self.draw_plan_traj(v.safe_plan_traj)

        if hasattr(v, 'pos_buffer') and v.pos_buffer is not None:
            self.draw_pos_buffer(v)

        if hasattr(v, 'key_points'):
            self.draw_key_points(v)

        if hasattr(v,'count'):
            self.draw_time_info(v)


        for i, objects in enumerate(self.history_objects):
            if i == len(self.history_objects) - 1:
                continue
            i = len(self.history_objects) - i
            if self.history_smooth != 0 and (i % self.history_smooth != 0):
                continue
            for v in objects:
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = x / len(self.history_objects)
                if self.semantic_map:
                    c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
                else:
                    c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
                ObjectGraphics.display(object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=False)

        # Draw the whole trajectory of ego vehicle with no gradient colors:
        if self.draw_target_vehicle_trajectory:
            for i, v in enumerate(self.history_target_vehicle):
                i = len(self.history_target_vehicle) - i
                if self.history_smooth != 0 and (i % self.history_smooth != 0):
                    continue
                c = v.color
                h = v.heading_theta
                h = h if abs(h) > 2 * np.pi / 180 else 0
                x = abs(int(i))
                alpha_f = min(x / len(self.history_target_vehicle), 0.5)
                # alpha_f = 0
                ObjectGraphics.display(
                    object=v,
                    surface=self._frame_canvas,
                    heading=h,
                    color=(c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2])),
                    draw_contour=False
                )

        # Draw current vehicle with black contour
        # Use this line if you wish to draw "future" trajectory.
        # i is the index of vehicle that we will render a black box for it.
        # i = int(len(self.history_vehicles) / 2)
        i = -1
        for v in self.history_objects[i]:
            h = v.heading_theta
            c = v.color
            h = h if abs(h) > 2 * np.pi / 180 else 0
            alpha_f = 0
            if self.semantic_map:
                c = TopDownSemanticColor.get_color(v.type) * (1 - alpha_f) + alpha_f * 255
            else:
                c = (c[0] + alpha_f * (255 - c[0]), c[1] + alpha_f * (255 - c[1]), c[2] + alpha_f * (255 - c[2]))
            ObjectGraphics.display(
                object=v, surface=self._frame_canvas, heading=h, color=c, draw_contour=self.contour, contour_width=2
            )

        if not hasattr(self, "_deads"):
            self._deads = []

        for v in self._deads:
            pygame.draw.circle(
                surface=self._frame_canvas,
                color=(255, 0, 0),
                center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                radius=5
            )

        for v in self.history_objects[i]:
            if v.done:
                pygame.draw.circle(
                    surface=self._frame_canvas,
                    color=(255, 0, 0),
                    center=self._frame_canvas.pos2pix(v.position[0], v.position[1]),
                    radius=5
                )
                self._deads.append(v)

        v = self.current_track_agent
        canvas = self._frame_canvas
        field = self._screen_canvas.get_size()
        if not self.target_agent_heading_up:
            if self.position is not None or v is not None:
                cam_pos = (self.position or v.position)
                position = self._frame_canvas.pos2pix(*cam_pos)
            else:
                position = (field[0] / 2, field[1] / 2)
            off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
            self.screen_canvas.blit(source=canvas, dest=(0, 0), area=(off[0], off[1], field[0], field[1]))
        else:
            position = self._frame_canvas.pos2pix(*v.position)
            area = (
                position[0] - self.canvas_rotate.get_size()[0] / 2, position[1] - self.canvas_rotate.get_size()[1] / 2,
                self.canvas_rotate.get_size()[0], self.canvas_rotate.get_size()[1]
            )
            self.canvas_rotate.fill(color_white)
            self.canvas_rotate.blit(source=canvas, dest=(0, 0), area=area)

            rotation = -np.rad2deg(v.heading_theta) + 90
            new_canvas = pygame.transform.rotozoom(self.canvas_rotate, rotation, 1)

            size = self._screen_canvas.get_size()
            self._screen_canvas.blit(
                new_canvas,
                (0, 0),
                (
                    new_canvas.get_size()[0] / 2 - size[0] / 2,  # Left
                    new_canvas.get_size()[1] / 2 - size[1] / 2,  # Top
                    size[0],  # Width
                    size[1]  # Height
                )
            )

        if self.show_agent_name:
            raise ValueError("This function is broken")
            # FIXME check this later
            if self.pygame_font is None:
                self.pygame_font = pygame.font.SysFont("Arial.ttf", 30)
            agents = [agent.name for agent in list(self.engine.agents.values())]
            for v in self.history_objects[i]:
                if v.name in agents:
                    position = self._frame_canvas.pos2pix(*v.position)
                    new_position = (position[0] - off[0], position[1] - off[1])
                    img = self.pygame_font.render(
                        text=self.engine.object_to_agent(v.name),
                        antialias=True,
                        color=(0, 0, 0, 128),
                    )
                    # img.set_alpha(None)
                    self.screen_canvas.blit(
                        source=img,
                        dest=(new_position[0] - img.get_width() / 2, new_position[1] - img.get_height() / 2),
                        # special_flags=pygame.BLEND_RGBA_MULT
                    )

    def _handle_event(self) -> None:
        """
        Handle pygame events for moving and zooming in the displayed area.
        """
        if self.no_window:
            return
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    import sys
                    sys.exit()

    @property
    def engine(self):
        from metadrive.engine.engine_utils import get_engine
        return get_engine()

    @property
    def screen_frames(self):
        return copy.deepcopy(self._screen_frames)

    def get_map(self):
        """
        Convert the map pygame surface to array

        Returns: map in array

        """
        return pygame.surfarray.array3d(self._background_canvas)
