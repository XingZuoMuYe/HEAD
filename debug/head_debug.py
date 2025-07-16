"""
Author: ShuoYang
Date: 2025-07-10
Description: main_head.py
"""

# main_head.py

from head.manager.config_manager import get_final_config
from head.manager.evolution_selector import resolve_evolution_strategy


if __name__ == '__main__':
    # 获取配置
    cfg = get_final_config()

    # 解析选择进化策略类
    EvolutionAlgoClass = resolve_evolution_strategy(cfg)

    # 初始化策略对象
    evolution_algo = EvolutionAlgoClass(cfg)

    # 初始化策略内部组件（如 agent、环境等）
    evolution_algo.agent_initialize()

    # 根据训练标志执行训练或评估
    if bool(cfg.train_flag):
        evolution_algo.train()
    else:
        evolution_algo.load()
        evolution_algo.eval()
