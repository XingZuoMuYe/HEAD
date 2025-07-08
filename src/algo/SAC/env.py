import warnings

from metadrive.envs import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from head.policy.rL_planning_policy import RLPlanningPolicy
from head.envs import StraightConfTraffic, MultiScenario
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial

class SeedGenerator:
    def __init__(self):
        self.seed = 42

    def next_seed(self):
        self.seed = int(time.time())
        return self.seed


def create_env(cfg, seed):
    # seed = 15
    env = StraightConfTraffic(dict(map=cfg.args.map_name,
                                   # This policy setting simplifies the task
                                   discrete_action=False,
                                   horizon=400,
                                   use_render=False,
                                   agent_policy=RLPlanningPolicy,
                                   # scenario setting
                                   traffic_mode="respawn",
                                   random_spawn_lane_index=False,
                                   num_scenarios=1,
                                   driving_reward=3.5,
                                   speed_reward=0.8,
                                   start_seed=seed,
                                   accident_prob=0,
                                   use_lateral_reward=True,
                                   log_level=50,
                                   crash_vehicle_penalty=10.0,
                                   crash_object_penalty=10.0,
                                   out_of_road_penalty=10.0,
                                   scenario_difficulty=cfg.args.scenario_difficulty,
                                   use_pedestrian=cfg.args.use_pedestrian,
                                   lane_num=cfg.args.training.lane_num,
                                   comfort_reward=2.0
                                   ))
    print("seed", seed)
    # env.seed(seed)
    return env


def create_metadrive_env(cfg, seed):
    """创建MetaDrive多场景环境"""
    common_config = dict(
        map=cfg.args.map_name,
        discrete_action=False,
        horizon=800,
        use_render=False,
        random_spawn_lane_index=False,
        num_scenarios=1,
        start_seed=5,
        accident_prob=0,
        random_traffic=True,
        use_lateral_reward=True,
        crash_vehicle_penalty=10.0,
        crash_object_penalty=10.0,
        out_of_road_penalty=10.0,
        log_level=50,
        traffic_density=0.15
    )


    env = MetaDriveEnv(common_config)
    # env.seed(seed)  # 注释掉的种子设置
    return env


def make_env_sac(cfg):
    print('Env is starting')
    seed_generator = SeedGenerator()

    if cfg.args.task == 'straight_config_traffic-v0':
        if cfg.args.training.use_vec_env:
            env = SubprocVecEnv([partial(create_env, cfg, seed_generator.next_seed()) for _ in range(cfg.args.training.env_num)])
        else:
            env = create_env(cfg, seed_generator.next_seed())

    elif cfg.args.task == 'muti_scenario-v0' or cfg.args.task == 'single_scenario-v0':
        if cfg.args.training.use_vec_env:
            env = SubprocVecEnv([partial(create_metadrive_env, cfg, seed_generator.next_seed()) for _ in range(cfg.args.training.env_num)])
        else:
            env = create_metadrive_env(cfg, seed_generator.next_seed())

    else:
        env = None
        print('No task')
    return env
