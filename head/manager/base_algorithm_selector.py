from head.policy.evolvable_policy.poly_planning_policy import RLPlanningPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import IDMPolicy
from head.policy.imitation_policy.imitation_planning_policy import ImitationPlanningPolicy
from head.manager.artifact_paths import has_poly_checkpoint

# 映射关系（可扩展）
class ZeroPolicy(BasePolicy):
    """Deployment baseline that always emits a zero control action."""

    def act(self, agent_id):
        action = [0.0, 0.0]
        self.action_info["action"] = action
        return action


class RandomPolicy(EnvInputPolicy):
    """Explicit deploy fallback when Poly has no checkpoint."""

    def act(self, agent_id):
        action = self.get_input_space().sample()
        self.action_info["action"] = action
        return action


EVO_POLICY_MAPPING = {
    'IDM': IDMPolicy,
    'imitation': ImitationPlanningPolicy,
    'Poly': RLPlanningPolicy,
    'Zero': EnvInputPolicy,
}
DEPLOYMENT_POLICY_MAPPING = {
    'IDM': IDMPolicy,
    'Poly': RLPlanningPolicy,
    'imitation': ImitationPlanningPolicy,
    'Zero': ZeroPolicy,
}


def resolve_agent_policy(cfg):
    """
    根据配置 cfg 中的 algorithm 字段，解析出对应的 agent_policy 类。
    ``workflow.type`` 表示 deploy/evolution，``workflow.policy`` 表示四个
    同级策略（IDM、Poly、Zero、imitation）。
    """
    mode = cfg.args.workflow.type
    if mode == "evo":
        mode = "evolution"
    if mode not in ("evolution", "deploy"):
        raise ValueError(f"无效的 workflow.type '{mode}'，必须是 'deploy' 或 'evolution'。")

    # ============ 进化流程 ============
    if mode == "evolution":
        main_algo = getattr(cfg.args.workflow, "policy", None)

        if main_algo not in EVO_POLICY_MAPPING:
            raise ValueError(f"未知的基础策略 '{main_algo}'")
        policy_class = EVO_POLICY_MAPPING[main_algo]
        print(f"[信息] 已选择基础算法：{main_algo}")
        return policy_class

    # ============ 部署流程 ============
    elif mode == "deploy":
        algo_type = getattr(cfg.args.workflow, "policy", None)

        if algo_type == "Poly":
            if not has_poly_checkpoint(cfg.args):
                print("[警告] deploy + Poly 未找到 checkpoint，使用 action_space.sample()")
                return RandomPolicy

        if algo_type not in DEPLOYMENT_POLICY_MAPPING:
            raise ValueError(f"未知的部署基础策略 '{algo_type}'，请在 DEPLOYMENT_POLICY_MAPPING 中注册。")

        policy_class = DEPLOYMENT_POLICY_MAPPING[algo_type]
        print(f"[信息] 已选择部署基础策略：{algo_type}")
        return policy_class

    return None
