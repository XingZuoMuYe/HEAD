"""
Author: ShuoYang
Date: 2025-07-10
Description: evolution_selector.py - Resolve and instantiate selected evolution strategy.
"""

from head.evolution_engine.RLBoost.SAC.SAC_learner import SAC_Learner, SACConfig

# ✅ 策略映射表
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
    根据 selection_method.main / sub 选择对应策略类。
    """
    sel_cfg = cfg.args.algorithm.evolutionary['selection_method']
    main = sel_cfg.get('main')
    sub = sel_cfg.get('sub')
    candidates = sel_cfg.get('candidates', {})

    # 基本合法性检查
    if main not in candidates:
        raise ValueError(f"Invalid main strategy '{main}'. Available: {list(candidates.keys())}")
    if sub not in candidates[main]:
        raise ValueError(f"Invalid sub-strategy '{sub}' under '{main}'. Available: {candidates[main]}")

    # 映射表检查
    if main not in EVOLUTION_STRATEGY_MAPPING or sub not in EVOLUTION_STRATEGY_MAPPING[main]:
        raise ValueError(f"Strategy '{main}/{sub}' not implemented.")

    # 返回策略类实例（只传 cfg）
    StrategyClass = EVOLUTION_STRATEGY_MAPPING[main][sub]
    return StrategyClass
