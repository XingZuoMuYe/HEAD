from head.policy.evolvable_policy.poly_planning_policy import RLPlanningPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.idm_policy import IDMPolicy

# 映射关系（可扩展）
EVOLUTIONARY_POLICY_MAPPING = {
    'Poly': RLPlanningPolicy,
    'Zero': EnvInputPolicy
    # 更多可扩展项...
}
DEPLOYMENT_POLICY_MAPPING = {
    'IDM': IDMPolicy
    # 更多可扩展项...
}

def resolve_agent_policy(cfg):
    """
    根据配置 cfg 中的 algorithm_type 字段，解析出对应的 agent_policy 类。
    """
    use_evolutionary = cfg.args.algorithm.evolutionary['use_evolutionary']
    use_deployment = cfg.args.algorithm.deployment['use_deployment']

    if use_evolutionary and use_deployment:
        print("[Warning] Both evolutionary and deployment algorithms are enabled.")
        print("→ Evolutionary algorithm takes priority; deployment algorithm will be ignored.")

    if use_evolutionary:
        algo_type = cfg.args.algorithm.evolutionary.algorithm_type['main']
        candidates = cfg.args.algorithm.evolutionary.algorithm_type.get('candidates', [])

        if algo_type not in candidates:
            raise ValueError(f"'{algo_type}' not listed in candidates: {candidates}")
        if algo_type not in EVOLUTIONARY_POLICY_MAPPING:
            raise ValueError(f"Unknown evolutionary algorithm_type '{algo_type}'.")
        policy_class = EVOLUTIONARY_POLICY_MAPPING[algo_type]

        print(f"[✅ Agent Policy Selected] Evolutionary → {algo_type}")
        return policy_class

    else:
        algo_type = cfg.args.algorithm.deployment['deployment_method']
        if algo_type not in DEPLOYMENT_POLICY_MAPPING:
            raise ValueError(f"Unknown deployment_method '{algo_type}'.")
        policy_class = DEPLOYMENT_POLICY_MAPPING[algo_type]
        print(f"[✅ Agent Policy Selected] Deployment → {algo_type}")
        return policy_class
