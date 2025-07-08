import sys
sys.path.append('/home/test/git_shuo/SPI')

from metadrive.envs import MetaDriveEnv
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from functools import partial
from IPython.display import clear_output
import os


def create_env(need_monitor=False):
    # map = "SSSSSS"
    # map = "XCO"
    env = MetaDriveEnv(dict(map="XCO",
                            # This policy setting simplifies the task
                            discrete_action=False,
                            horizon=500,
                            use_render=False,
                            random_traffic=True,
                            # scenario setting
                            random_spawn_lane_index=False,

                            num_scenarios=1,
                            start_seed=5,
                            traffic_density=0.2,
                            accident_prob=0,
                            use_lateral_reward=True,
                            log_level=50))

    if need_monitor:
        env = Monitor(env)
    return env


if __name__ == "__main__":
    train = True

    log_path = os.path.join('Results/sb3_RL/', 'logs')
    model_path = os.path.join('Results/sb3_RL/', 'Saved Models', 'sb3_SAC_model_XCO_4')

    if train:
        set_random_seed(0)
        # 6 subprocess to roll out
        train_env = SubprocVecEnv([partial(create_env, True) for _ in range(1)])
        model = SAC("MlpPolicy",
                    train_env,
                    verbose=1,
                    tensorboard_log=log_path)

        model.learn(total_timesteps=3000000,
                    log_interval=1)

        model.save(model_path)
        clear_output()
        print("Training is finished!")
    else:
        # evaluation
        total_reward = 0
        env = create_env()
        obs, _ = env.reset()
        ep_len = 0
        for i in range(1000):
            while True:
                model = SAC.load(model_path)
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                ret = env.render(mode="topdown",
                                 screen_record=True)
                ep_len+=1
                if done:
                    print("episode_reward", total_reward)
                    print("episode_len", ep_len)
                    total_reward = 0
                    ep_len = 0
                    env.top_down_renderer.generate_gif()
                    env.reset()
                    break