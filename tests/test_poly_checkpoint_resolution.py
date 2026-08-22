from pathlib import Path

from head.manager.artifact_paths import resolve_poly_checkpoint
from head.manager.config_manager import get_final_config


def test_poly_auto_finds_legacy_stage_checkpoint(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main_head.py", "task=straight_config_traffic-v0", "workflow.policy=Poly"])
    cfg = get_final_config()
    checkpoint = resolve_poly_checkpoint(cfg.args)
    assert checkpoint is not None
    assert checkpoint.name.startswith("stage_")
    assert (checkpoint / "sac_policy").is_file()
