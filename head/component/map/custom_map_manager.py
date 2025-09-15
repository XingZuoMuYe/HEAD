from metadrive.constants import DEFAULT_AGENT
from metadrive.scenario.parse_object_state import parse_full_trajectory, parse_object_state, get_idm_route
import copy

from metadrive.manager.scenario_map_manager import ScenarioMapManager


class CustomMapManager(ScenarioMapManager):
    """
    Modified ScenarioMapManager

    改动说明：
    1. 去掉 include_z_position=True，仅保留 2D 坐标，避免无效的 z 轴处理。
    2. 去掉 NuScenes 专用的 width/length 交换逻辑，保持原始尺寸。
    3. agent_config 更新精简，仅设置位置、朝向和速度，几何参数交由全局 vehicle_config 管理。

    目的：提升跨数据集通用性（Waymo、NuScenes、Geely....），减少不必要的数据修正与冲突。
    """

    def __init__(self):
        super(CustomMapManager, self).__init__()

    def update_route(self):
        """
        重写 update_route：去掉了原版 width/length swap，且不包含 include_z_position=True
        """
        data = self.engine.data_manager.current_scenario
        sdc_track = data.get_sdc_track()

        # full trajectory
        sdc_traj = parse_full_trajectory(sdc_track)

        # init and last state
        init_state = parse_object_state(sdc_track, 0, check_last_state=False)   # 没有 include_z_position=True
        last_state = parse_object_state(sdc_track, -1, check_last_state=True)

        init_position = init_state["position"]
        init_yaw = init_state["heading"]
        last_position = last_state["position"]
        last_yaw = last_state["heading"]

        # route
        self.current_sdc_route = get_idm_route(sdc_traj)
        self.sdc_start_point = copy.deepcopy(init_position)
        self.sdc_dest_point = copy.deepcopy(last_position)

        # update agent configs
        self.engine.global_config.update(
            copy.deepcopy(
                dict(
                    agent_configs={
                        DEFAULT_AGENT: dict(
                            spawn_position_heading=(init_position, init_yaw),
                            spawn_velocity=init_state["velocity"]
                        )
                    }
                )
            )
        )
