"""Select a simulator policy through its strategy-family adapter."""
from head.model import resolve_policy_binding


def resolve_agent_policy(cfg):
    policy_class = resolve_policy_binding(cfg).policy_class
    print(f"[信息] 已选择基础策略：{cfg.args.workflow.policy}")
    return policy_class


def __getattr__(name):
    # Public legacy imports refer to the canonical implementations.
    if name == "ZeroPolicy":
        from head.model.zero.policy import ZeroPolicy
        return ZeroPolicy
    if name == "RandomPolicy":
        from head.model.poly.adapter import RandomPolicy
        return RandomPolicy
    if name == "RLPlanningPolicy":
        from head.model.poly.policy import RLPlanningPolicy
        return RLPlanningPolicy
    raise AttributeError(name)
