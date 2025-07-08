
from src.algo.SAC.cfg import parse_cfg
import argparse
from src.algo.SAC.SAC_learner import SAC_Learner, SACConfig
from pathlib import Path

__CONFIG__, __LOGS__ = 'config', 'logs'


def to_dict(config):
    ans = dict()
    for i in dir(config):
        if i.startswith("__"): continue
        x = getattr(config, i)
        ans[i] = x
    return ans


def merge_two_dicts(x, y):
    x = to_dict(x)
    y = to_dict(y)
    z = x.copy()
    z.update(y)
    return argparse.Namespace(**z)


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_flag', type=int, default=1,
                        help='train = 1 or eval = 0')
    parser.add_argument('--train_name', type=str, default='train_1')
    parser.add_argument('--total_steps', type=float, default=1e6)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    conf = parse_args_cfgs()
    args = parse_cfg(Path().cwd().parent / __CONFIG__ / "SAC")

    cfg = SACConfig(merge_two_dicts(conf, args))

    if bool(cfg.train_flag):
        SAC = SAC_Learner(cfg)
        SAC.agent_initialize()
        SAC.train()

    else:
        SAC = SAC_Learner(cfg)
        SAC.agent_initialize()
        SAC.load()
        SAC.eval()
