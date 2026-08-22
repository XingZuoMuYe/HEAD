import sys

import pytest

from head.manager.base_algorithm_selector import resolve_agent_policy
from head.manager.config_manager import get_final_config
from head.manager.evolution_selector import resolve_evolution_strategy


@pytest.mark.parametrize(
    "workflow_type,policy,policy_name,strategy_name",
    [
        ("deploy", "IDM", "IDMPolicy", "NoEvolutionStrategy"),
        ("deploy", "imitation", "ImitationPlanningPolicy", "ImitationStrategy"),
        ("deploy", "Poly", "RLPlanningPolicy", "SAC_Learner"),
        ("deploy", "Zero", "ZeroPolicy", "NoEvolutionStrategy"),
        ("evolution", "IDM", "IDMPolicy", "SAC_Learner"),
        ("evolution", "imitation", "ImitationPlanningPolicy", "SAC_Learner"),
        ("evolution", "Poly", "RLPlanningPolicy", "SAC_Learner"),
        ("evolution", "Zero", "EnvInputPolicy", "SAC_Learner"),
    ],
)
def test_workflow_policy_matrix(
    monkeypatch, workflow_type, policy, policy_name, strategy_name
):
    overrides = [
        f"workflow.type={workflow_type}",
        f"workflow.policy={policy}",
    ]
    if policy == "imitation":
        overrides.append("task=real_scenario-v0")
        overrides.extend([
            "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
            "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
        ])
    else:
        overrides.append("task=straight_config_traffic-v0")
    monkeypatch.setattr(sys, "argv", ["main_head.py", *overrides])
    cfg = get_final_config()
    assert resolve_agent_policy(cfg).__name__ == policy_name
    assert resolve_evolution_strategy(cfg).__name__ == strategy_name


def test_deploy_poly_without_checkpoint_uses_random_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main_head.py", "workflow.policy=Poly", "artifacts.root=/tmp/head-missing-artifacts"
    ])
    cfg = get_final_config()
    assert resolve_agent_policy(cfg).__name__ == "RandomPolicy"
    assert resolve_evolution_strategy(cfg).__name__ == "NoEvolutionStrategy"


def test_poly_auto_checkpoint_uses_random_when_run_is_missing(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "main_head.py", "workflow.policy=Poly", "artifacts.root=/tmp/head-missing-artifacts"
    ])
    cfg = get_final_config()
    assert cfg.args.workflow.policies.Poly.checkpoint == "auto"
    assert resolve_agent_policy(cfg).__name__ == "RandomPolicy"
