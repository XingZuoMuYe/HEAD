from head.policy.evolvable_policy.poly_planning_policy import RLPlanningPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.idm_policy import IDMPolicy

# 映射关系（可扩展）
BASE_POLICY_MAPPING = {
    'Poly': RLPlanningPolicy,
    'Zero': EnvInputPolicy
    # 更多可扩展项...
}
DEPLOYMENT_POLICY_MAPPING = {
    'IDM': IDMPolicy,
    'Poly': RLPlanningPolicy,
    # 更多可扩展项...
}


def resolve_agent_policy(cfg):
    """
    根据配置 cfg 中的 algorithm 字段，解析出对应的 agent_policy 类。
    支持 evolutionary / deployment 两种模式，互斥。
    """
    mode = getattr(cfg.args.algorithm, "mode", None)
    if mode not in ("evolutionary", "deployment"):
        raise ValueError(f"无效的算法模式 '{mode}'，必须是 'evolutionary' 或 'deployment'。")

    # ============ 进化算法模式 ============
    if mode == "evolutionary":
        evo_cfg = cfg.args.algorithm.evolutionary

        # 使用新的字段 'base_algorithm_type'
        main_algo = evo_cfg.base_algorithm_type.get("main", None)
        candidates = evo_cfg.base_algorithm_type.get("candidates", {})

        if main_algo not in candidates:
            raise ValueError(f"主算法 '{main_algo}' 不在候选列表中: {list(candidates.keys())}")

        policy_class = BASE_POLICY_MAPPING[main_algo]
        print(f"[信息] 已选择基础算法：{main_algo}")
        return policy_class

    # ============ 部署算法模式 ============
    elif mode == "deployment":
        dep_cfg = cfg.args.algorithm.deployment
        algo_type = dep_cfg.deployment_method.get("main", None)

        if algo_type not in DEPLOYMENT_POLICY_MAPPING:
            raise ValueError(f"未知的部署算法类型 '{algo_type}'，请在 DEPLOYMENT_POLICY_MAPPING 中注册。")

        policy_class = DEPLOYMENT_POLICY_MAPPING[algo_type]
        print(f"[信息] 已选择部署算法：{algo_type}")
        return policy_class

    return None
