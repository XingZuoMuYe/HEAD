import sys
import os
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf

CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'),
                  ('total_time', 'T', 'time')]
AGENT_METRICS = ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']



def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def cfg_to_group(cfg, return_list=False):
    """Return a wandb-safe group name for logging. Optionally returns group name as list."""
    lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
    return lst if return_list else '-'.join(lst)


class VideoRecorder:
    """Utility class for logging evaluation videos."""

    def __init__(self, root_dir, wandb, cfg, render_size=384, fps=80):
        self.save_dir = (root_dir + '/eval_video') if root_dir else None
        self._wandb = wandb
        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.enabled = False
        self.cfg = cfg

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir and self._wandb and enabled
        self.record(env)

    def record(self, env):
        if not self.enabled:
            return

        # Common render parameters
        # render_params = {
        #     "mode": "topdown",
        #     "screen_record": False,
        #     "window": False,
        #     "show_plan_traj": True
        # }

        render_params = {
            "mode": "topdown",
            "screen_record": False,
            "window": False,
            # "show_plan_traj": True
        }

        # Specific task configurations
        if self.cfg.task == 'straight_config_traffic-v0':
            render_params.update({
                "scaling": 6,
                "film_size": (6000, 400)
            })
            frame = env.render(**render_params)

        elif self.cfg.task in ['muti_scenario-v0', 'single_scenario-v0']:
            frame = env.render(**render_params)

        else:
            frame = []
            print(f"Unknown task: {self.cfg.task}")

        self.frames.append(frame)

    def save(self):
        if self.enabled:
            frames = np.stack(self.frames).transpose(0, 3, 1, 2)

            self._wandb.log({"video": self._wandb.Video(frames, fps=self.fps, format="mp4")})


class Logger(object):
    """Primary logger object. Logs either locally or using wandb."""

    def __init__(self, log_dir, cfg):
        self._log_dir = make_dir(log_dir + '/wandb_info')
        self._group = cfg_to_group(cfg)
        self._eval = []
        project, entity = cfg.wandb.wandb_project, cfg.wandb.wandb_entity
        run_offline = not cfg.wandb.use_wandb or project == 'none' or entity == 'none'
        if run_offline:
            print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
            self._wandb = None
        else:
            try:
                os.environ["WANDB_SILENT"] = "true"
                import wandb
                wandb.init(project=project,
                           entity=entity,
                           name=str(cfg.train_name),
                           group=self._group,
                           tags=cfg_to_group(cfg, return_list=True) + [f'seed:{cfg.misc.seed}'],
                           dir=self._log_dir,
                           # config=OmegaConf.to_container(cfg, resolve=True))
                           config=cfg)
                print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
                self._wandb = wandb
            except:
                print('Warning: failed to init wandb. Logs will be saved locally.')
                self._wandb = None
        self._video = VideoRecorder(log_dir, self._wandb, cfg) if self._wandb and cfg.misc.save_video else None

    @property
    def video(self):
        return self._video

    def finish(self, agent):
        if self._wandb:
            self._wandb.finish()

    def log(self, d, category='train'):
        assert category in {'train', 'eval'}
        if self._wandb is not None:
            for k, v in d.items():
                self._wandb.log({category + '/' + k: v}, step=d['env_step'])
        if category == 'eval':
            keys = ['env_step', 'episode_reward']
            self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
            pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
