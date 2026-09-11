"""Contract tests for plug-in algorithms and the single closed-loop runner."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from head.agents import AgentInput, BaseAgent, Trajectory, create_agent, get_agent_class
from head.agents.wayformer.agent import Agent as WayformerAgent
from head.policy.imitation_policy.closed_loop_inference import ClosedLoopInference


class ToyAgent(BaseAgent):
    @classmethod
    def load_config(cls):
        return OmegaConf.create({"past_len": 1, "step_interval": 0.1})

    def initialize(self, checkpoint):
        self.checkpoint = checkpoint
        self.resets = 0

    def compute_trajectory(self, agent_input):
        return Trajectory(np.array([[1, 0], [2, 0]], dtype=np.float32))

    def reset(self):
        self.resets += 1


def test_new_algorithm_needs_no_core_changes(monkeypatch, tmp_path):
    module = types.ModuleType("head.agents.toy.agent")
    module.Agent = ToyAgent
    monkeypatch.setitem(sys.modules, module.__name__, module)
    checkpoint = tmp_path / "toy.weights"
    checkpoint.touch()
    cfg = OmegaConf.create({"model": "toy", "options": {"past_len": 3}})
    inference = ClosedLoopInference(cfg, checkpoint)
    assert inference.agent.history_steps == 3
    assert inference.predict({}, 4).shape == (2, 2)
    inference.reset()
    assert inference.trajectory is None
    assert inference.agent.resets == 1
    assert get_agent_class("head.agents.toy.agent:Agent") is ToyAgent
    # Config validation also has no hard-coded model allow-list.
    from head.manager.config_manager import get_final_config
    monkeypatch.setattr(sys, "argv", ["main_head.py", "task=real_scenario-v0",
        "workflow.policy=imitation", "workflow.policies.imitation.model=toy",
        f"workflow.policies.imitation.checkpoint={checkpoint}"])
    assert get_final_config().args.workflow.policies.imitation.model == "toy"


@pytest.mark.parametrize("samples", [
    np.zeros((1, 2)), np.zeros((3, 3)), np.zeros((2, 5)),
    np.full((2, 2), np.nan), np.full((2, 4), np.inf),
])
def test_invalid_output_rejected(samples):
    with pytest.raises(ValueError):
        Trajectory(samples)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_dt_rejected(dt):
    with pytest.raises(ValueError):
        Trajectory(np.zeros((2, 2)), dt=dt)


def test_reference_frame_is_explicit():
    trajectory = Trajectory(np.zeros((2, 4)), reference_offset=2)
    np.testing.assert_allclose(
        trajectory.control_position([10, 5], np.pi / 2), [10, 3], atol=1e-6)


def test_dt_mismatch_is_not_silent():
    engine = ClosedLoopInference.__new__(ClosedLoopInference)
    engine.agent = ToyAgent({})
    engine.controller_dt = 0.2
    with pytest.raises(ValueError, match="dt"):
        engine.predict({}, 1)


def test_bad_plugin_output_rejected():
    engine = ClosedLoopInference.__new__(ClosedLoopInference)
    engine.agent = types.SimpleNamespace(compute_trajectory=lambda _: np.zeros((2, 2)))
    engine.controller_dt = 0.1
    with pytest.raises(TypeError, match="Trajectory"):
        engine.predict({}, 1)


def test_wayformer_selects_probability_not_mode_zero():
    trajectories = torch.zeros(1, 3, 4, 5)
    trajectories[0, 1, :, :2] = 7
    trajectories[..., 2:] = 999  # distribution parameters, not velocity
    selected = WayformerAgent.select_trajectory({
        "predicted_trajectory": trajectories,
        "predicted_probability": torch.tensor([[0.1, 0.8, 0.1]]),
    })
    np.testing.assert_equal(selected, np.full((4, 2), 7))


def test_wayformer_world_transform_and_velocity_semantics():
    agent = WayformerAgent({})
    agent.dataset = types.SimpleNamespace(process_scenario=lambda *_: (
        {}, np.array([[10, 20, 0, 0, 0, 0, np.pi / 2]])))
    local = torch.tensor([[[[1., 0., 999., 999., 999.],
                             [2., 0., 999., 999., 999.]]]])
    agent.model = lambda _: {"predicted_trajectory": local,
                            "predicted_probability": torch.ones(1, 1)}
    result = agent.compute_trajectory(AgentInput({}, 21))
    assert result.reference_offset == 0
    assert result.samples.shape == (2, 2)
    np.testing.assert_allclose(result.samples, [[10, 21], [10, 22]], atol=1e-5)


def test_nonfinite_probability_rejected():
    with pytest.raises(ValueError, match="finite"):
        WayformerAgent.select_trajectory({
            "predicted_trajectory": torch.zeros(1, 2, 4, 5),
            "predicted_probability": torch.tensor([[float("nan"), 0.5]]),
        })


def test_importing_agents_does_not_import_networks():
    import subprocess
    code = ("import sys; from head.agents import get_agent_class; "
            "get_agent_class('pluto'); get_agent_class('wayformer'); "
            "assert 'head.agents.wayformer.model' not in sys.modules; "
            "assert 'head.agents.pluto.model.pluto_model' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_no_hidden_external_checkout_imports():
    root = Path(__file__).parents[1]
    for parent in [root / "head/agents", root / "head/policy/imitation_policy"]:
        for path in parent.rglob("*.py"):
            text = path.read_text()
            assert "from unitraj." not in text, path
            assert "import unitraj." not in text, path
            assert "imitation_policy.UniTraj" not in text, path
    assert not (root / "head/policy/imitation_policy/pluto_closed_loop_inference.py").exists()


def test_pluto_neural_only_is_default():
    config = get_agent_class("pluto").load_config()
    assert config.trajectory_selection_mode == "neural_only"
    assert config.rule_based_score_weight == 0
    assert config.learning_based_score_weight == 1


def test_scalar_normalization_does_not_change_source():
    from head.agents.common.scenario import normalize_scalar_states
    values = np.arange(3).reshape(-1, 1)
    scenario = {"tracks": {"ego": {"state": {"heading": values}}}}
    normalized = normalize_scalar_states(scenario)
    assert normalized["tracks"]["ego"]["state"]["heading"].shape == (3,)
    assert scenario["tracks"]["ego"]["state"]["heading"].shape == (3, 1)
    normalized["tracks"]["ego"]["state"]["heading"] = np.zeros((3, 1))
    np.testing.assert_equal(scenario["tracks"]["ego"]["state"]["heading"], values)
