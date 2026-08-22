# -*-coding:utf-8-*-
import os
import time
import re
import torch
import datetime
from head.evolution_engine.RLBoost.SAC.agent import SAC
from head.evolution_engine.common.utils import make_dir
import numpy as np
from collections import deque
import shutil
from head.evolution_engine.RLBoost.SAC.logger import Logger
from head.evolution_engine.env_builder.env import make_env
from head.manager.artifact_paths import evolution_paths, poly_checkpoint_roots, resolve_poly_checkpoint
from head.manager.closed_loop_metrics import ClosedLoopMetricsRecorder
from pathlib import Path


def clip(a, low, high):
    return min(max(a, low), high)


def generate_paths(args) -> dict:
    """
    根据任务名称和地图名称生成相关路径。

    Args:
        args: 包含任务名称（args.task）和地图名称（args.scenario.map）的参数对象。

    Returns:
        包含以下路径的字典：
        - base_result_path: 基础结果路径
        - model_save_path: 模型保存路径
        - log_save_path: 日志保存路径
        - eval_save_path: 评估结果保存路径
    """
    paths = evolution_paths(args)
    task = paths["task"]
    map_name = paths["map"]
    print("\033[1;33m[INFO]\033[0m Task Name: \033[1;32m{}\033[0m, Map Name: \033[1;34m{}\033[0m".format(task, map_name))

    result = {
        "base_result_path": str(paths["logs"]),
        "model_save_path": str(paths["weights"]),
        "log_save_path": str(paths["logs"]),
        "eval_save_path": str(paths["evaluation"]),
        "task": task,
        "map": map_name,
    }
    return result



def get_dir_path(path):
    """
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    """

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # os.makedirs(path)
        return path, None
    else:
        path, path_origin = directory_check(path)
        # os.makedirs(path)
        return path + '/', path_origin + '/'


def directory_check(directory_check):
    temp_directory_check = directory_check
    i = 1
    while i:

        if os.path.exists(temp_directory_check):
            search = '_'
            numList = [m.start() for m in re.finditer(search, temp_directory_check)]
            numList[-1]
            temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
            i = i + 1
        else:
            return temp_directory_check, temp_directory_check[0:numList[-1] + 1] + str(i - 2)


class SACConfig:
    def __init__(self, args):
        self.algo = 'SAC'
        self.env_name = args.scenario.environment
        self.train_name = args.train_name
        # 生成路径
        path_dict = generate_paths(args)

        self.base_result_path = path_dict["base_result_path"]
        self.model_save_path = path_dict["model_save_path"]
        self.log_save_path = path_dict["log_save_path"]
        self.eval_save_path = path_dict["eval_save_path"]
        self.path_task = path_dict["task"]
        self.path_map = path_dict["map"]
        self.train_eps = 200 # 1000000
        self.eps_max_steps = args.evaluation.max_steps
        self.eval_eps = args.evaluation.episodes
        self.total_steps = getattr(args, 'total_steps', 1e8)
        self.gamma = 0.99
        self.soft_tau = 5e-3
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000
        self.model_save_interval = 5000
        self.eval_interval = args.evaluation.interval_steps
        self.eval_total_steps = 60000
        self.hidden_dim = 256
        self.batch_size = 256
        self.alpha_lr = 3e-4
        self.AUTO_ENTROPY = True
        self.DETERMINISTIC = False
        requested_device = args.runtime.device
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.device is 'cuda', but CUDA is not available")
        self.device = torch.device(requested_device)
        if args.train_flag is not None:
            if args.train_flag not in (0, 1):
                raise ValueError("train_flag must be 0 (evaluation) or 1 (training)")
            self.train_flag = args.train_flag
        elif args.workflow.type == 'deploy':
            self.train_flag = 0
        elif args.workflow.type in ('evolution', 'evo'):
            self.train_flag = 1
        self.args = args
        if args.simulation["vectorized"] and args.simulation["render"]:
            print("Warning: simulation.render is disabled for vectorized environments.")
            args.simulation["render"] = False


class SAC_Learner:

    def __init__(self, SAC_cfg):

        self.collision_rate_buffer = None
        self.env = None
        self.writer = None
        self.rewards = None
        self.ma_rewards = None
        self.SAC_cfg = SAC_cfg
        self.model_save_path = self.SAC_cfg.model_save_path
        self.model_path = str(Path(self.model_save_path) / self.SAC_cfg.train_name) + '/'
        self.log_path = str(Path(self.SAC_cfg.log_save_path) / self.SAC_cfg.train_name) + '/'
        self.eval_path = str(Path(self.SAC_cfg.eval_save_path) / self.SAC_cfg.train_name) + '/'

        self.fps = 0.0

        if self.SAC_cfg.train_flag:

            self.model_path, _ = get_dir_path(self.model_path)
            make_dir(self.model_path)

            # 定义训练信息
            self.L = Logger(self.SAC_cfg.base_result_path, self.SAC_cfg.args)
        else:
            self.L = None
        self.agent = None
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

        self.ep_info_buffer = None
        self.ep_len_buffer = None
        self.ep_info_buffer = []
        self.ep_len_buffer = []

        self.collision_rate_buffer = []
        self.ep_velocity_buffer = []
        self.mean_velocity_buffer = []
        self.eval_ep_info_buffer = []
        self.eval_ep_len_buffer = []
        self.eval_collision_rate_buffer = []
        self.eval_ep_velocity_buffer = []
        self.eval_mean_velocity_buffer = []

        self.ep_len = 0

    def agent_initialize(self):
        self.env = make_env(self.SAC_cfg)
        print('agent is initializing')
        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[0]


        self.agent = SAC(state_dim, action_dim, self.SAC_cfg)
        print(self.SAC_cfg.algo + ' algorithm is starting')


    def generate_env(self):
        self.env.close()
        self.env = make_env(self.SAC_cfg)


    def update_config(self, env_name):

        if env_name == 'S':
            self.SAC_cfg.args.task = 'straight_config_traffic-v0'
            self.SAC_cfg.args.scenario.map = 'SSSSSSSSSSSSSS'
            # TODO 在random_env中，这两个参数暂时固定
            self.SAC_cfg.args.scenario.difficulty =  1
            self.SAC_cfg.args.scenario.pedestrians =  True
        else:
            self.SAC_cfg.args.task = 'multi_scenario-v0'
            self.SAC_cfg.args.scenario.map = env_name

    def load(self):
        checkpoint = resolve_poly_checkpoint(self.SAC_cfg.args)
        if checkpoint is None:
            checked = ", ".join(str(path) for path in poly_checkpoint_roots(self.SAC_cfg.args))
            print(
                "[警告] 未找到 Poly checkpoint，保持随机初始化。Checked: "
                f"{checked}"
            )
            return
        self.agent.load(str(checkpoint))
        print(f"agent {self.SAC_cfg.train_name} is loaded from {checkpoint}")

    def render(self):
        if self.SAC_cfg.args.task == 'straight_config_traffic-v0':
            self.env.head_renderer.render(mode="topdown",
                            screen_record=False,
                            scaling=6,
                            film_size=(6000, 400),
                            show_plan_traj=True,
                            )
        elif self.SAC_cfg.args.task in ['multi_scenario-v0', 'muti_scenario-v0', 'single_scenario-v0']:
            self.env.head_renderer.render(mode="topdown",
                            screen_record=False,
                            show_plan_traj=True,
                            )
        elif self.SAC_cfg.args.task == 'real_scenario-v0':
            self.env.head_renderer.render(mode="topdown",
                            show_plan_traj=True,
                            show_agent_name=False,
                            film_size=(5500, 5500),
                            scaling=3,
                            screen_size=(800, 800),
                            screen_record=False,
                            text={
                                # "step": t,
                                "scenario index": self.env.engine.global_seed + self.env.config["start_scenario_index"],
                            }
                            )

    def train(self):
        if self.SAC_cfg.args.simulation.vectorized:
            self.train_vec_env()
        else:
            self.train_standalone_env()

    def eval(self):
        if self.SAC_cfg.args.simulation.vectorized:
            self.eval_vec_env()
        else:
            self.eval_standalone_env()

    def train_standalone_env(self):

        print('Start to train !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        total_nums = 0
        frame_count = 0
        save_flags = [False] * int(self.SAC_cfg.total_steps / self.SAC_cfg.model_save_interval)
        eval_flags = [False] * int(self.SAC_cfg.total_steps / self.SAC_cfg.eval_interval)
        self.ep_info_buffer = deque(maxlen=30)
        self.ep_len_buffer = deque(maxlen=30)
        self.collision_rate_buffer = deque(maxlen=1500)
        start_time = time.time()
        t_start = time.time()
        ep_reward = 0.0
        ep_len = 0.0
        collision_flag = 0.0
        for i_ep in range(self.SAC_cfg.train_eps):
            self.generate_env()
            state, _ = self.env.reset()
            ###########
            if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.init(
                self.env)
            ##############
            train_metrics = {}
            for i_step in range(self.SAC_cfg.eps_max_steps):
                total_nums = total_nums + 1

                n_state = state
                action = self.agent.policy_net.get_action(n_state)

                next_state, reward, done, termin, info = self.env.step(action)
                ###########
                if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None:
                    self.L.video.record(self.env)
                elif self.SAC_cfg.args.simulation.render:
                    self.render()
                ################
                frame_count += 1
                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)
                state = next_state
                if time.time() - start_time >= 1:
                    self.fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

                ep_len = info['episode_length']
                ep_reward = info['episode_reward']
                collision_flag = info['crash'] or info['out_of_road']

                if done or termin:
                    ############
                    if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.save()
                    ###########
                    break

            self.ep_info_buffer.append(ep_reward)
            self.ep_len_buffer.append(ep_len)
            self.collision_rate_buffer.append(collision_flag)
            mean_eps_reward = np.mean([ep_info for ep_info in self.ep_info_buffer])
            mean_eps_len = np.mean([ep_len for ep_len in self.ep_len_buffer])
            mean_coll_rate = np.mean([coll for coll in self.collision_rate_buffer])
            re_time = time.strftime("%H:%M:%S",
                                    time.gmtime((time.time() - t_start) / total_nums * self.SAC_cfg.total_steps))

            print(
                f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_eps_reward:.3f}, fps:{self.fps}, ep_len: {mean_eps_len}, collision_rate:{mean_coll_rate},"
                f"remaining time:{re_time}")
            print(f'总步数：{total_nums}')
            info = {'ep_rew_mean': mean_eps_reward,
                    'ep_len': mean_eps_len,
                    'fps': self.fps,
                    'coll_rate_mean': mean_coll_rate,
                    'env_step': total_nums}
            train_metrics.update(info)
            self.L.log(train_metrics, category='train')

            for m in range(len(save_flags)):
                if total_nums > (m + 1) * self.SAC_cfg.model_save_interval and not save_flags[m]:
                    path_l = self.model_path + 'stage_' + str(m)
                    path = self.model_path + 'stage_' + str(m + 1)
                    self.save(path)
                    if m != 0:
                        shutil.rmtree(path_l)
                    print('save', total_nums)
                    save_flags[m] = True

            for n in range(len(eval_flags)):
                if total_nums > (n + 1) * self.SAC_cfg.eval_interval and not eval_flags[n]:
                    print('eval', total_nums)
                    self.eval_standalone_env()
                    eval_flags[n] = True

            if total_nums >= self.SAC_cfg.total_steps:
                break

        print('Complete training！')

    def train_vec_env(self):

        print('Start to train !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        total_nums = 0
        frame_count = 0
        save_flags = [False] * int(self.SAC_cfg.total_steps / self.SAC_cfg.model_save_interval)
        eval_flags = [False] * int(self.SAC_cfg.total_steps / self.SAC_cfg.eval_interval)
        self.ep_info_buffer = deque(maxlen=30)
        self.ep_len_buffer = deque(maxlen=30)
        self.collision_rate_buffer = deque(maxlen=1500)
        ep_velocity_buffer = []
        start_time = time.time()
        t_start = time.time()
        ep_reward = np.zeros(self.env.num_envs)
        ep_len = np.zeros(self.env.num_envs)
        ep_speed = np.zeros(self.env.num_envs)
        collision_flag = np.zeros(self.env.num_envs)

        for i_ep in range(self.SAC_cfg.train_eps):
            self.generate_env()

            state = self.env.reset()

            reset_flag = np.array([True for i in range(self.env.num_envs)])
            train_metrics = {}
            for i_step in range(self.SAC_cfg.eps_max_steps):
                total_nums = total_nums + 1

                action = self.agent.policy_net.get_action(state)
                # print(action)
                next_state, reward, done, info = self.env.step([row for row in action])
                for d in range(self.env.num_envs):
                    ep_speed[d] = info[d]['velocity']

                ep_velocity_buffer.append(ep_speed[d])
                for i in range(self.env.num_envs):
                    if reset_flag[i]:
                        frame_count += 1
                        self.agent.memory.push(state[i], action[i], reward[i], next_state[i], done[i])

                frame_count += 1

                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)
                state = next_state
                if time.time() - start_time >= 1:
                    self.fps = frame_count / (time.time() - start_time)
                    frame_count = 0
                    start_time = time.time()

                for d in range(self.env.num_envs):
                    if done[d]:
                        reset_flag[d] = False
                        total_nums += info[d]['episode_length']
                        ep_len[d] = info[d]['episode_length']
                        ep_reward[d] = info[d]['episode_reward']
                        collision_flag[d] = info[d]['crash'] or info[d]['out_of_road']


                if not reset_flag.any():
                    break

            self.ep_info_buffer.append(ep_reward[0])
            self.ep_len_buffer.append(ep_len[0])
            self.collision_rate_buffer.append(collision_flag[0])
            mean_eps_reward = np.mean([ep_info for ep_info in self.ep_info_buffer])
            mean_eps_len = np.mean([ep_len for ep_len in self.ep_len_buffer])
            mean_coll_rate = np.mean([coll for coll in self.collision_rate_buffer])
            mean_ep_speed = np.array(ep_velocity_buffer).mean()
            re_time = time.strftime("%H:%M:%S",
                                    time.gmtime((time.time() - t_start) / total_nums * self.SAC_cfg.total_steps))

            print(
                f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_eps_reward:.3f}, fps:{self.fps}, ep_len: {mean_eps_len}, collision_rate:{mean_coll_rate}, mean_ep_speed:{mean_ep_speed}"
                f"remaining time:{re_time}")
            print(f'总步数：{total_nums}')
            info = {'ep_rew_mean': mean_eps_reward,
                    'ep_len': mean_eps_len,
                    'fps': self.fps,
                    'coll_rate_mean': mean_coll_rate,
                    'env_step': total_nums,
                    'average_speed': mean_ep_speed,
                    'overtake_vehicle_num': info[d]['overtake_vehicle_num'],
                    'completed_vehicle_num':info[d]['route_completion'],}
            train_metrics.update(info)
            self.L.log(train_metrics, category='train')
            ep_velocity_buffer = []
            for m in range(len(save_flags)):
                if total_nums > (m + 1) * self.SAC_cfg.model_save_interval and not save_flags[m]:
                    path_l = self.model_path + 'stage_' + str(m)
                    path = self.model_path + 'stage_' + str(m + 1)
                    self.save(path)
                    if m != 0:
                        shutil.rmtree(path_l)
                    print('save', total_nums)
                    save_flags[m] = True

            for n in range(len(eval_flags)):
                if total_nums > (n + 1) * self.SAC_cfg.eval_interval and not eval_flags[n]:
                    self.eval_vec_env()
                    eval_flags[n] = True

            if total_nums >= self.SAC_cfg.total_steps:
                break

        print('Complete training！')

    def eval_vec_env(self):
        print('start_eval_env')
        self.SAC_cfg.args.simulation.vectorized = False
        self.generate_env()
        print('Start closed-loop evaluation')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        ma_rewards = []  # moving average reward
        total_nums = 0
        ep_reward = 0.0
        collision_flag = 0.0
        if self.SAC_cfg.train_flag:
            eval_eps = self.SAC_cfg.args.evaluation.episodes
        else:
            eval_eps = self.SAC_cfg.eval_eps
        closed_loop_metrics = ClosedLoopMetricsRecorder(self.SAC_cfg)

        for i_ep in range(eval_eps):

            state, _ = self.env.reset()
            if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.init(
                self.env)
            eps_reward = 0.0
            ep_len = 0
            closed_loop_metrics.start_episode()
            for i_step in range(self.SAC_cfg.eps_max_steps):

                total_nums = total_nums + 1

                # lidar_state, navi_state_info, vehicle_state_info = state_process(self.env)
                # state[19:] = np.array(lidar_state)
                # state[9:19] = navi_state_info
                # state[0:2] = vehicle_state_info
                action = self.agent.policy_net.get_action(state)

                next_state, reward, done, termin, info = self.env.step(action)
                closed_loop_metrics.step(info, next_state, i_step, self.env)
                # print('time =', i_step/10)
                if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None:
                    self.L.video.record(self.env)
                elif self.SAC_cfg.args.simulation.render:
                    self.render()
                state = next_state
                eps_reward += reward
                ep_len += 1
                ep_len = info['episode_length']
                ep_reward = info['episode_reward']
                collision_flag = info['crash'] or info['out_of_road']
                ep_speed = info['velocity']

                self.eval_ep_velocity_buffer.append(ep_speed)

                if done or termin:
                    print("episode_len", ep_len)
                    if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.save()
                    break

            closed_loop_metrics.finish_episode(
                episode_index=i_ep + 1,
                reward=eps_reward,
                length=ep_len,
                env=self.env,
            )

            self.eval_ep_info_buffer.append(ep_reward)
            self.eval_ep_len_buffer.append(ep_len)
            self.eval_collision_rate_buffer.append(collision_flag)

            mean_eps_reward = np.mean([ep_info for ep_info in self.eval_ep_info_buffer])
            mean_eps_len = np.mean([ep_len for ep_len in self.eval_ep_len_buffer])
            mean_coll_rate = np.mean([coll for coll in self.eval_collision_rate_buffer])
            mean_eps_velocity = np.mean([coll for coll in self.eval_ep_velocity_buffer])

            self.eval_mean_velocity_buffer.append(mean_eps_velocity)
            mean_velocity = np.mean([coll for coll in self.eval_mean_velocity_buffer])
            velocity_bar = (np.max(self.eval_mean_velocity_buffer) - mean_velocity) + (
                    mean_velocity - np.min(self.eval_mean_velocity_buffer)) / 2
            reward_bar = 0.0

            print(
                f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_eps_reward:.3f},reward_bar:{reward_bar:.3f} ,ep_len: {mean_eps_len}, collision_rate:{mean_coll_rate},"
                f"ep_velocity:{mean_eps_velocity}, mean_velocity: {mean_velocity}, velocity_bar:{velocity_bar}")

            print(f'总步数：{total_nums}')

            if total_nums >= self.SAC_cfg.total_steps:
                break

        print('Complete closed-loop evaluation')

        metrics_path = closed_loop_metrics.save()
        if metrics_path is not None:
            print(f"[信息] 闭环指标已保存: {metrics_path}")

        self.SAC_cfg.args.simulation.vectorized = True
        return rewards, ma_rewards





    def eval_standalone_env(self):
        print('Start closed-loop evaluation')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        ma_rewards = []  # moveing average reward
        total_nums = 0
        ep_speed = []
        if self.SAC_cfg.train_flag:
            eval_eps = self.SAC_cfg.args.evaluation.episodes
        else:
            eval_eps = self.SAC_cfg.eval_eps
        closed_loop_metrics = ClosedLoopMetricsRecorder(self.SAC_cfg)

        for i_ep in range(eval_eps):
            self.generate_env()
            state, _ = self.env.reset()
            if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.init(
                self.env)
            eps_reward = 0.0
            ep_len = 0
            closed_loop_metrics.start_episode()

            for i_step in range(self.SAC_cfg.eps_max_steps):

                total_nums = total_nums + 1

                action = self.agent.policy_net.get_eval_action(state)

                next_state, reward, done, termin, info = self.env.step(action)
                closed_loop_metrics.step(info, next_state, i_step, self.env)
                ep_speed.append(info['velocity'])
                if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None:
                    self.L.video.record(self.env)
                else:
                    self.render()
                state = next_state
                eps_reward += reward
                ep_len += 1
                if done or termin:
                    print("episode_len", ep_len)
                    if self.SAC_cfg.train_flag and self.SAC_cfg.args.evaluation.save_video and self.L.video is not None: self.L.video.save()
                    break

            closed_loop_metrics.finish_episode(
                episode_index=i_ep + 1,
                reward=eps_reward,
                length=ep_len,
                env=self.env,
            )

            # mean_reward = eps_reward / i_step
            mead_speed = np.array(ep_speed).mean()
            ep_speed = []
            rewards.append(eps_reward)
            print(f"Episode:{i_ep + 1}/{self.SAC_cfg.eval_eps}, MeanSpeed:{mead_speed:.3f},  Reward:{eps_reward:.3f}")
            print(f'总步数：{total_nums}')
            if total_nums >= self.SAC_cfg.total_steps:
                break

        print('Complete closed-loop evaluation')
        metrics_path = closed_loop_metrics.save()
        if metrics_path is not None:
            print(f"[信息] 闭环指标已保存: {metrics_path}")
        return rewards, ma_rewards

    def save(self, path):
        make_dir(path)
        self.agent.save(path)
