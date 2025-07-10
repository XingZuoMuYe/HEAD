import warnings
from metadrive.envs import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

from head.policy.evolvable_policy.rL_planning_policy import RLPlanningPolicy
from head.envs import StraightConfTraffic
import time
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from functools import partial
from head.renderer.head_renderer import HeadTopDownRenderer

# Disable deprecation warnings for now
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 映射关系（可扩展）

EVOLUTIONARY_POLICY_MAPPING = {
    'Poly': RLPlanningPolicy,
    # 更多可扩展项...
}
DEPLOYMENT_POLICY_MAPPING = {
    'IDM': IDMPolicy,
}

def resolve_agent_policy(cfg):
    """
    根据配置 cfg 中的 algorithm_type 字段，解析出对应的 agent_policy 类。
    """
    use_evolutionary = cfg.args.algorithm.evolutionary['use_evolutionary']
    use_deployment = cfg.args.algorithm.deployment['use_deployment']

    if use_evolutionary and use_deployment:
        print("[Warning] Both evolutionary and deployment algorithms are enabled.")
        print("→ Evolutionary algorithm takes priority; deployment algorithm will be ignored.")

    if use_evolutionary:
        algo_type = cfg.args.algorithm.evolutionary['algorithm_type']
        if algo_type not in EVOLUTIONARY_POLICY_MAPPING:
            raise ValueError(f"Unknown algorithm_type '{algo_type}'. Please check POLICY_MAPPING.")
        return EVOLUTIONARY_POLICY_MAPPING[algo_type]
    else:
        algo_type = cfg.args.algorithm.deployment['deployment_method']
        if algo_type not in DEPLOYMENT_POLICY_MAPPING:
            raise ValueError(f"Unknown algorithm_type '{algo_type}'. Please check POLICY_MAPPING.")
        return DEPLOYMENT_POLICY_MAPPING[algo_type]



class SeedGenerator:
    def __init__(self):
        self.seed = 42

    def next_seed(self):
        self.seed = int(time.time())
        return self.seed


class EnvConfig:
    def __init__(self, cfg):
        self.cfg = cfg
        self.common_config = {
            'discrete_action': False,
            'horizon': 400,  # Default horizon, can be overridden
            'use_render': False,
            'random_spawn_lane_index': False,
            'num_scenarios': 1,
            'accident_prob': 0,
            'use_lateral_reward': True,
            'crash_vehicle_penalty': 10.0,
            'crash_object_penalty': 10.0,
            'out_of_road_penalty': 10.0,
            'log_level': 50,
            'map_config': {
                "type": 'block_sequence',
                "exit_length": 50,
                'lane_num': cfg.args.training.lane_num,
                'config': cfg.args.map_name,
                "start_position": [0, 0],
            },
        }

        self.agent_policy = resolve_agent_policy(cfg)
        self._apply_custom_config(cfg)

    def _apply_custom_config(self, cfg):
        """Apply the custom settings provided in the config."""
        if 'straight_config_traffic-v0' in cfg.args.task:
            self.common_config.update({
                'agent_policy': self.agent_policy,  # Use RLPlanningPolicy for this task
                'driving_reward': 3.5,
                'speed_reward': 0.8,
                'start_seed': None,  # Will be set later
                'scenario_difficulty': cfg.args.scenario_difficulty,
                'use_pedestrian': cfg.args.use_pedestrian,
                'comfort_reward': 2.0,
                'traffic_mode': "respawn",
            })
            self.common_config['horizon'] = 400

        # Override the horizon for MetaDrive or multi-scenario tasks
        if 'muti_scenario' in cfg.args.task or 'single_scenario' in cfg.args.task:
            self.common_config.update({
                'agent_policy': RLPlanningPolicy,  # Use RLPlanningPolicy for this task
            })
            self.common_config['horizon'] = 1200
            self.common_config['start_seed'] = 5
            self.common_config['random_traffic'] = True  # MetaDrive specific
            self.common_config['traffic_density'] = 0.1  # MetaDrive specific

    def create_env(self, seed):
        """Create the environment based on task type."""
        config = self.common_config.copy()

        if self.cfg.args.task == 'straight_config_traffic-v0':
            config['start_seed'] = seed
            env = StraightConfTraffic(config)

        elif self.cfg.args.task in ['muti_scenario-v0', 'single_scenario-v0']:
            env =  MetaDriveEnv(config)

        else:
            print('No task configured.')
            return None
        env.reset()
        env.head_renderer = HeadTopDownRenderer(env)
        return env



def make_env_sac(cfg):
    print('Env is starting')
    seed_generator = SeedGenerator()

    # Create a single environment or a vectorized environment
    env_config = EnvConfig(cfg)
    if cfg.args.training.use_vec_env:
        env = SubprocVecEnv(
            [partial(env_config.create_env, seed_generator.next_seed()) for _ in range(cfg.args.training.env_num)])
    else:
        env = env_config.create_env(seed_generator.next_seed())

    return env
