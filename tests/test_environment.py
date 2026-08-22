import sys

from head.manager.config_manager import get_final_config
from head.evolution_engine.env_builder.env import make_env


def test_default_environment_reset_and_policy_step(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main_head.py"])
    cfg = get_final_config()
    env = make_env(cfg)
    try:
        observation, _ = env.reset()
        assert observation is not None
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(float(reward), float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    finally:
        env.close()
