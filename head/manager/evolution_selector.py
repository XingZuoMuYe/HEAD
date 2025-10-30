"""
作者: ShuoYang
日期: 2025-07-10
描述: evolution_selector.py - 解析并实例化选定的进化策略。
"""

from head.evolution_engine.RLBoost.SAC.SAC_learner import SAC_Learner, SACConfig

# 策略映射表（可扩展）
EVOLUTION_STRATEGY_MAPPING = {
    'RLBoost': {
        'SAC': SAC_Learner,
        'PPO': None,
    },
    'DreamMethod': {
        'HeadMethodInDream': None,
    }
}


def resolve_evolution_strategy(cfg):
    """
    根据 evolution_method_type.main / sub 选择对应的策略类。
    """
    sel_cfg = cfg.args.algorithm.evolutionary['evolution_method_type']
    main = sel_cfg.get('main')
    sub = sel_cfg.get('sub')
    candidates = sel_cfg.get('candidates', {})

    # 基本合法性检查
    if main not in candidates:
        raise ValueError(f"主策略 '{main}' 无效，可选项为: {list(candidates.keys())}")
    if sub not in candidates[main]:
        raise ValueError(f"子策略 '{sub}' 不属于主策略 '{main}' 的候选范围，可选项为: {candidates[main]}")

    # 映射表检查
    if main not in EVOLUTION_STRATEGY_MAPPING or sub not in EVOLUTION_STRATEGY_MAPPING[main]:
        raise ValueError(f"策略 '{main}/{sub}' 尚未实现，请检查配置或扩展映射表。")

    # 返回策略类
    StrategyClass = EVOLUTION_STRATEGY_MAPPING[main][sub]
    print(f"[信息] 已选择进化策略：{main}/{sub}")
    return StrategyClass
