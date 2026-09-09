"""
作者: ShuoYang
日期: 2025-11-10
描述: imitation_selector.py - 解析并实例化选定的模仿学习策略。
"""

import random

import numpy as np
import torch
from omegaconf import OmegaConf
from head.manager.artifact_paths import resolve_imitation_checkpoint
from head.policy.imitation_policy.unitraj_loader import ensure_unitraj_path


def resolve_imitation_strategy(cfg):
    """
    从外部 cfg 中读取 imitation method 名字，并加载对应的内部 config。
    
    Args:
        cfg: 外层配置对象
        
    Returns:
        model: 构建好的模型
        merged_cfg: 合并后的配置
    """
    imitation_cfg = cfg.args.workflow.policies.imitation
    source = imitation_cfg.get("source", None)
    source_root = ensure_unitraj_path(source)
    # 1. 从外层配置拿 method 名字
    method_name = str(imitation_cfg.model)
    print(f"[外层配置指定的 imitation method] {method_name}")

    # 2. 找到对应的内部 config 文件
    METHOD_CONFIG_DIR = source_root / "unitraj" / "configs" / "method"
    GLOBAL_CONFIG_DIR = source_root / "unitraj" / "configs"
    
    method_config_name = "Pluto" if method_name.lower() == "pluto" else method_name
    method_cfg_path = METHOD_CONFIG_DIR / f"{method_config_name}.yaml"
    global_cfg_path = GLOBAL_CONFIG_DIR / "config.yaml"
    
    if not method_cfg_path.exists():
        raise FileNotFoundError(f"未找到 method 配置文件: {method_cfg_path}")
    if not global_cfg_path.exists():
        raise FileNotFoundError(f"未找到全局配置文件: {global_cfg_path}")

    # 3. 加载内部配置
    method_cfg = OmegaConf.load(method_cfg_path)
    global_cfg = OmegaConf.load(global_cfg_path)
    merged_cfg = OmegaConf.merge({"method": method_cfg}, method_cfg, global_cfg)
    merged_cfg.model_name = method_config_name
    merged_cfg.ckpt_path = str(resolve_imitation_checkpoint(cfg.args))
    merged_cfg["eval"] = True
    seed = int(merged_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. 构建模型
    if method_name.lower() == "pluto":
        from unitraj.models.pluto.pluto_model import PlanningModel
        model_class = PlanningModel
    elif method_name == "wayformer":
        from unitraj.models.wayformer.wayformer import Wayformer
        model_class = Wayformer
    elif method_name == "autobot":
        from unitraj.models.autobot.autobot import AutoBotEgo
        model_class = AutoBotEgo
    elif method_name == "MTR":
        from unitraj.models.mtr.MTR import MotionTransformer
        model_class = MotionTransformer
    else:
        raise ValueError(f"Unsupported imitation model: {method_name}")
    model = model_class(config=merged_cfg)
    print(f"[✅ 使用的 method config] {method_cfg_path}")
    
    return model, merged_cfg
