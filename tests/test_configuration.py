import sys

import pytest

from head.manager.config_manager import get_final_config


def load_config(monkeypatch, *overrides):
    monkeypatch.setattr(sys, "argv", ["main_head.py", *overrides])
    return get_final_config()


def test_default_configuration_is_runnable(monkeypatch):
    cfg = load_config(monkeypatch)
    assert cfg.args.task == "real_scenario-v0"
    assert cfg.args.workflow.type == "deploy"
    assert cfg.args.workflow.policy == "imitation"
    assert cfg.args.scenario.kind == "recorded"
    assert cfg.args.scenario.capabilities.closed_loop_imitation is True
    assert cfg.args.artifacts.weights.evolution == "weights/evolution"
    assert cfg.args.artifacts.weights.imitation == "weights/imitation"
    assert cfg.args.workflow.policies.Poly.checkpoint == "auto"
    assert cfg.args.workflow.policies.imitation.checkpoint == (
        "artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt"
    )


def test_imitation_policy_accepts_capable_task(monkeypatch):
    cfg = load_config(
        monkeypatch,
        "task=real_scenario-v0",
        "workflow.type=deploy",
        "workflow.policy=imitation",
        "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
        "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
    )
    assert cfg.args.workflow.policy == "imitation"
    assert cfg.args.scenario.kind == "recorded"
    assert cfg.args.scenario.capabilities.closed_loop_imitation is True


def test_imitation_rejects_incapable_task(monkeypatch):
    with pytest.raises(ValueError, match="closed_loop_imitation=true"):
        load_config(
            monkeypatch,
            "task=straight_config_traffic-v0",
            "workflow.policy=imitation",
            "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
            "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
        )


def test_evolution_accepts_poly_policy(monkeypatch):
    cfg = load_config(
        monkeypatch,
        "workflow.type=evolution",
        "workflow.policy=Poly",
        "task=straight_config_traffic-v0",
    )
    assert cfg.args.workflow.evolution.strategy == "RLBoost"


@pytest.mark.parametrize("workflow_type", ["deploy", "evolution"])
@pytest.mark.parametrize("base_policy", ["IDM", "Poly", "Zero"])
def test_workflow_matrix_accepts_regular_bases(monkeypatch, workflow_type, base_policy):
    cfg = load_config(
        monkeypatch,
        f"workflow.type={workflow_type}",
        f"workflow.policy={base_policy}",
        "task=straight_config_traffic-v0",
    )
    assert cfg.args.workflow.policy == base_policy


@pytest.mark.parametrize("workflow_type", ["deploy", "evolution"])
def test_workflow_matrix_accepts_imitation_on_capable_task(monkeypatch, workflow_type):
    cfg = load_config(
        monkeypatch,
        "task=real_scenario-v0",
        f"workflow.type={workflow_type}",
        "workflow.policy=imitation",
        "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
        "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
    )
    assert cfg.args.workflow.policy == "imitation"


def test_invalid_task_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="Unknown task"):
        load_config(monkeypatch, "task=does-not-exist-v0")


def test_invalid_workflow_type_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="workflow.type"):
        load_config(monkeypatch, "workflow.type=missing")


@pytest.mark.parametrize("policy", ["IDM", "Poly", "Zero"])
def test_direct_policy_alias_is_normalized(monkeypatch, policy):
    cfg = load_config(monkeypatch, f"workflow.type={policy}")
    assert cfg.args.workflow.type == "deploy"
    assert cfg.args.workflow.policy == policy


def test_direct_imitation_alias_is_normalized(monkeypatch):
    cfg = load_config(
        monkeypatch,
        "task=real_scenario-v0",
        "workflow.type=imitation",
        "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
        "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
    )
    assert cfg.args.workflow.type == "deploy"
    assert cfg.args.workflow.policy == "imitation"


def test_pluto_is_rejected_before_environment_creation(monkeypatch):
    with pytest.raises(ValueError, match="pluto.*not implemented"):
        load_config(
            monkeypatch,
            "task=real_scenario-v0",
            "workflow.policy=imitation",
            "workflow.policies.imitation.model=pluto",
            "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
            "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
        )


def test_imitation_missing_checkpoint_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="checkpoint"):
        load_config(
            monkeypatch,
            "task=real_scenario-v0",
            "workflow.policy=imitation",
            "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
            "workflow.policies.imitation.checkpoint=",
        )


def test_invalid_environment_count_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="num_envs"):
        load_config(monkeypatch, "simulation.num_envs=0")


def test_pixels_modality_is_rejected_until_implemented(monkeypatch):
    with pytest.raises(ValueError, match="only 'state' is currently supported"):
        load_config(monkeypatch, "modality=pixels")


def test_unknown_option_is_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_head.py", "--totl_steps=1"])
    with pytest.raises(SystemExit):
        get_final_config()


def test_train_flag_override_is_respected(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_head.py", "--train_flag=1"])
    cfg = get_final_config()
    assert cfg.train_flag == 1


def test_task_override_loads_task_specific_scenario(monkeypatch):
    cfg = load_config(monkeypatch, "task=real_scenario-v0")
    assert cfg.args.scenario.dataset.name == "waymo"
    assert list(cfg.args.scenario.dataset.supported.custom) == ["geely"]


def test_multi_scenario_is_canonical_task_name(monkeypatch):
    cfg = load_config(monkeypatch, "task=multi_scenario-v0", "workflow.policy=IDM")
    assert cfg.args.task == "multi_scenario-v0"
    assert cfg.args.scenario.map == "XCO"


def test_legacy_muti_scenario_alias_is_normalized(monkeypatch):
    cfg = load_config(monkeypatch, "task=muti_scenario-v0", "workflow.policy=IDM")
    assert cfg.args.task == "multi_scenario-v0"


def test_legacy_profile_selector_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="replaced by.*workflow"):
        load_config(monkeypatch, "profile=imitation")
