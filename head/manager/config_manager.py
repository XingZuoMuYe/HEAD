"""
Author: ShuoYang
Date: 2025-07-10
Description: config_manager.py
"""

# config_manager.py

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
import argparse
from pathlib import Path
from head.evolution_engine.RLBoost.SAC.cfg import parse_cfg
from head.evolution_engine.RLBoost.SAC.SAC_learner import SACConfig

__CONFIG__ = 'head/configs'  # 相对 main 文件路径
VALID_TASKS = {
    "multi_scenario-v0",
    "muti_scenario-v0",  # legacy alias
    "straight_config_traffic-v0",
    "single_scenario-v0",
    "real_scenario-v0",
}
VALID_BASE_POLICIES = {"IDM", "imitation", "Poly", "Zero"}
VALID_WORKFLOWS = {"deploy", "evolution"}
VALID_EVOLUTION_STRATEGIES = {("RLBoost", "SAC")}
VALID_IMITATION_MODELS = {"wayformer", "pluto"}

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
    parser.add_argument('--train_flag', type=int, default=None,
                        help='train = 1 or eval = 0')
    parser.add_argument('--train_name', type=str, default='experiment_1')
    parser.add_argument('--total_steps', type=float, default=1e6)
    # Keep task/config overrides (for example ``task=straight_config_traffic-v0``)
    # for OmegaConf.from_cli(), which parses the rest of sys.argv.
    args, unknown = parser.parse_known_args()
    invalid_options = [item for item in unknown if item.startswith("--")]
    if invalid_options:
        parser.error(f"unrecognized option(s): {', '.join(invalid_options)}")
    return args

def get_final_config():
    """
    获取合并后的最终配置（命令行参数 + 配置文件）
    """
    conf = parse_args_cfgs()
    project_root = Path(__file__).resolve().parent.parent.parent
    args = parse_cfg(project_root / __CONFIG__)
    merged_args = merge_two_dicts(conf, args)
    validate_config(merged_args)
    return SACConfig(merged_args)


def validate_config(args):
    """Validate the small set of fields required before environment creation."""
    if args.task not in VALID_TASKS:
        raise ValueError(f"Unknown task '{args.task}'. Choose one of: {sorted(VALID_TASKS)}")
    workflow = args.workflow
    if args.task == "muti_scenario-v0":
        args.task = "multi_scenario-v0"
    # ``evo`` and ``base_policy`` were used by older README examples. Accept
    # them as a migration aid, but expose one canonical schema to selectors.
    # Accept direct policy names as a concise deploy alias, e.g.
    # ``workflow.type=imitation``. The canonical representation remains
    # ``type=deploy`` plus ``policy=imitation``.
    if workflow.type in VALID_BASE_POLICIES:
        workflow.policy = workflow.type
        workflow.type = "deploy"
    if workflow.type == "evo":
        workflow.type = "evolution"
    if workflow.type not in VALID_WORKFLOWS:
        raise ValueError("workflow.type must be 'deploy' or 'evolution'")
    if args.runtime.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.device must be 'auto', 'cpu', or 'cuda'")
    if args.simulation.num_envs < 1:
        raise ValueError("simulation.num_envs must be at least 1")
    if args.evaluation.episodes < 1:
        raise ValueError("evaluation.episodes must be at least 1")
    if getattr(args.evaluation, "mode", "closed_loop") != "closed_loop":
        raise ValueError("evaluation.mode must be 'closed_loop'; HEAD evaluation is environment-stepped")
    if not args.artifacts.root:
        raise ValueError("artifacts.root must not be empty")
    if not args.artifacts.weights.evolution:
        raise ValueError("artifacts.weights.evolution must not be empty")
    if not args.artifacts.weights.imitation:
        raise ValueError("artifacts.weights.imitation must not be empty")
    policy = getattr(workflow, "policy", None)
    legacy_base = getattr(workflow, "base_policy", None)
    if legacy_base is not None and (policy is None or policy == "IDM"):
        policy = legacy_base
        workflow.policy = policy
    if policy is None:
        raise ValueError("workflow.policy is required (IDM, Poly, Zero, or imitation)")
    if policy not in VALID_BASE_POLICIES:
        raise ValueError(f"Unknown workflow.policy '{policy}'")
    policies = getattr(workflow, "policies", None)
    if policies is None or not hasattr(policies, policy):
        raise ValueError(f"workflow.policies.{policy} configuration is required")
    selected = getattr(policies, policy)
    checkpoint = selected.get("checkpoint", None)
    if checkpoint is not None and not isinstance(checkpoint, str):
        raise ValueError(f"workflow.policies.{policy}.checkpoint must be a path or null")
    if checkpoint == "auto" and policy != "Poly":
        raise ValueError(f"workflow.policies.{policy}.checkpoint=auto is only valid for Poly")
    if policy == "imitation":
        imitation = selected
        if args.task != "real_scenario-v0" or not args.scenario.capabilities.closed_loop_imitation:
            raise ValueError(
                "workflow.policy=imitation is only supported by real_scenario-v0 "
                "with scenario.capabilities.closed_loop_imitation=true"
            )
        model = imitation.get("model")
        if model not in VALID_IMITATION_MODELS:
            raise ValueError("workflow.policies.imitation.model must be 'wayformer' or 'pluto'")
        if model == "pluto":
            raise ValueError("imitation model 'pluto' is reserved but not implemented yet")
        if not imitation.get("source") or not imitation.get("checkpoint"):
            raise ValueError(
                "workflow.policies.imitation.source and checkpoint are required"
            )
        if int(imitation.get("warmup_steps", 0)) < 0:
            raise ValueError("workflow.policies.imitation.warmup_steps must be non-negative")
        if int(imitation.get("replan_frequency", 1)) < 1:
            raise ValueError("workflow.policies.imitation.replan_frequency must be at least 1")
    if workflow.type == "evolution":
        strategy = (workflow.evolution.strategy, workflow.evolution.learner)
        if strategy not in VALID_EVOLUTION_STRATEGIES:
            raise ValueError(f"Unsupported evolutionary strategy: {'/'.join(strategy)}")
    return args
