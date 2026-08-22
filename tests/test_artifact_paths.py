from head.manager.artifact_paths import evolution_paths, imitation_checkpoint_candidates
from head.manager.config_manager import get_final_config


def test_evolution_paths_are_strategy_scoped(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main_head.py", "task=straight_config_traffic-v0", "workflow.policy=IDM"])
    cfg = get_final_config()
    paths = evolution_paths(cfg.args)
    assert str(paths["weights"]).endswith("weights/evolution/RLBoost/SAC/straight_config_traffic/straight_road")
    assert str(paths["logs"]).endswith("logs/RLBoost/SAC/straight_config_traffic/straight_road")


def test_imitation_candidates_include_dedicated_and_legacy_locations(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "main_head.py", "task=real_scenario-v0", "workflow.policy=imitation",
        "workflow.policies.imitation.source=/home/test/git_shuo/UniTraj_benchmark_sample",
        "workflow.policies.imitation.checkpoint=brier_fde=1.45.ckpt",
    ])
    cfg = get_final_config()
    candidates = [str(path) for path in imitation_checkpoint_candidates(cfg.args)]
    assert any("artifacts/weights/imitation/wayformer" in path for path in candidates)
    assert any(path.endswith("head/policy/imitation_policy/checkpoints/brier_fde=1.45.ckpt") for path in candidates)


def test_default_imitation_checkpoint_is_project_relative(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main_head.py", "task=real_scenario-v0", "workflow.policy=imitation"])
    # The selected imitation config is supplied by the default file.
    cfg = get_final_config()
    candidates = [str(path) for path in imitation_checkpoint_candidates(cfg.args)]
    assert candidates[0].endswith(
        "artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt"
    )
