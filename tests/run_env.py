import time
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map
from stable_baselines3.common.monitor import Monitor
from head.envs.config_traffic_metadrive_env import StraightConfTraffic
from head.renderer.head_renderer import HeadTopDownRenderer
from head.policy.evolvable_policy.rL_planning_policy import RLPlanningPolicy
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
def create_env(need_monitor=False):
    env = StraightConfTraffic(dict(
                                   # This policy setting simplifies the task
                                   discrete_action=False,
                                   horizon=400,
                                   use_render=False,
                                   agent_policy=RLPlanningPolicy,
                                   # scenario setting
                                   traffic_mode="respawn",
                                   random_spawn_lane_index=False,
                                   num_scenarios=1,
                                   start_seed=5,
                                   accident_prob=0,
                                   use_lateral_reward=True,
                                   log_level=50,
                                   crash_vehicle_penalty=30.0,
                                   crash_object_penalty=30.0,
                                   out_of_road_penalty=30.0,
                                   scenario_difficulty=0,
                                   use_pedestrian=True,
                                   map_config={
                                       "type": 'block_sequence',
                                       "exit_length": 50,
                                       'lane_num': 4,
                                       'config': 'SSSSSSSSSSSSSSSS',
                                       "start_position": [0, 0],
                                   },
                                   ))
    if need_monitor:
        env = Monitor(env)
    _ , _ = env.reset()
    return env

# 创建多场景环境
def create_multi_scenario_env(need_monitor=False):
    """
    创建并返回一个多场景环境，支持渲染和监控。
    """
    env = MetaDriveEnv(dict(
        # map=3,
        discrete_action=False,  # 使用连续动作空间
        horizon=2800,  # 设定最大时间步数
        use_render=False,  # 是否渲染环境
        agent_policy=RLPlanningPolicy,  # 使用IDM策略
        random_spawn_lane_index=True,  # 随机生成车道索引
        num_scenarios=1,  # 场景数量
        start_seed=5,  # 随机种子
        accident_prob=0,  # 事故概率
        use_lateral_reward=True,  # 使用横向奖励
        log_level=50,  # 日志级别
        crash_vehicle_penalty=30.0,  # 撞车惩罚
        crash_object_penalty=30.0,  # 撞物惩罚
        out_of_road_penalty=30.0,  # 离开道路惩罚
        traffic_density=0.15 , # 交通密度
        map_config={
            "type":'block_sequence',
            "exit_length": 50,
            'lane_num': 3,
            'config': 'CCO',
            "start_position": [0, 0],
        },
    ))

    # 如果需要监控，添加 Monitor
    if need_monitor:
        env = Monitor(env)
    return env


# 主函数
if __name__ == "__main__":
    total_reward = 0

    # multi_scenario_env
    # env = create_multi_scenario_env(need_monitor=False)  # 创建环境

    # straight_env
    env = create_env()

    env.reset()
    env.head_renderer = HeadTopDownRenderer(env)
    # 设置绘图
    fig, ax = plt.subplots()
    # 重置环境并绘制地图
    m = draw_top_down_map(env.current_map)

    # 绘制地图
    ax.imshow(m, cmap="bone")
    # 高级网格设置
    ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.7, alpha=0.7, color='#666666')
    # 设置坐标轴标签
    ax.set_xlabel('X (m)', fontsize=10, labelpad=5)
    ax.set_ylabel('Y (m)', fontsize=10, labelpad=5)
    # 设置坐标轴刻度
    ax.tick_params(axis='both', which='major', labelsize=8)

    # 保存图像
    plt.savefig("map.jpg", dpi=1000, bbox_inches='tight')

    # 运行1000个时间步，执行随机动作并记录总奖励
    for i in range(1000):
        while True:
            t1 = time.time()
            action = env.action_space.sample()  # 随机选择一个动作
            obs, reward, truncate, terminate, info = env.step(action)  # 执行动作并获取反馈
            total_reward += reward  # 累加奖励

            # multi_scenario_env
            # env.head_renderer.render(
            #     screen_record=False,
            #     show_plan_traj=True,
            #     mode="topdown")

            # straight_env
            env.head_renderer.render(
                screen_record=False,
                scaling=6,
                film_size=(6000, 400),
                show_plan_traj=True,
                mode="topdown")

            t2 = time.time()
            # 打印每一步的时间消耗（可选）
            # print('t=', t2 - t1)

            # 如果达到终止条件，则重置环境
            if truncate or terminate:
                env.reset()
                break

