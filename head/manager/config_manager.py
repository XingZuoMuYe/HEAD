"""
Author: ShuoYang
Date: 2025-07-10
Description: config_manager.py
"""

# config_manager.py

import argparse
from pathlib import Path
from head.evolution_engine.RLBoost.SAC.cfg import parse_cfg
from head.evolution_engine.RLBoost.SAC.SAC_learner import SACConfig

__CONFIG__ = 'head/configs'  # 相对 main 文件路径

def to_dict(config):
    ans = dict()
    for i in dir(config):
        if i.startswith("__"):
            continue
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

def get_final_config():
    """
    获取合并后的最终配置（命令行参数 + 配置文件）
    """
    conf = parse_args_cfgs()
    args = parse_cfg(Path().cwd().parent / __CONFIG__)
    merged_args = merge_two_dicts(conf, args)
    return SACConfig(merged_args)
