"""Strategy families and model plugins share selection, not algorithm internals."""
import sys
import types
from pathlib import Path

import pytest
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.env_input_policy import EnvInputPolicy

from head.model import BasePolicyAdapter, PolicyBinding, get_policy_adapter, resolve_policy_binding
from head.manager.base_algorithm_selector import resolve_agent_policy
from head.manager.config_manager import get_final_config
from head.manager.evolution_selector import resolve_evolution_strategy


class ExamplePolicy(BasePolicy):
    def act(self, agent_id):
        return [0.0, 0.0]


class ExampleAdapter(BasePolicyAdapter):
    @classmethod
    def resolve(cls, args, mode):
        return PolicyBinding(ExamplePolicy, "learner" if mode == "evolution" else "policy")


@pytest.mark.parametrize("mode,runner,strategy", [
    ("deploy", "policy", "NoEvolutionStrategy"),
    ("evolution", "learner", "SAC_Learner"),
])
def test_new_family_needs_no_selector_changes(monkeypatch, mode, runner, strategy):
    module = types.ModuleType("head.model.example.adapter")
    module.Adapter = ExampleAdapter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, "argv", ["main_head.py", f"workflow.type={mode}",
        "workflow.policy=example", "workflow.policies.example.checkpoint=unused"])
    cfg = get_final_config()
    assert get_policy_adapter("example") is ExampleAdapter
    assert get_policy_adapter("head.model.example.adapter:Adapter") is ExampleAdapter
    assert resolve_agent_policy(cfg) is ExamplePolicy
    assert resolve_policy_binding(cfg).runner == runner
    assert resolve_evolution_strategy(cfg).__name__ == strategy


def test_family_dependency_error_is_not_hidden(monkeypatch):
    from head.model import loader

    def missing_dependency(name):
        raise ModuleNotFoundError("Missing dependency", name="optional_dependency")

    monkeypatch.setattr(loader.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError) as error:
        get_policy_adapter("example")
    assert error.value.name == "optional_dependency"


def test_unknown_family_has_actionable_error():
    with pytest.raises(ValueError, match="Unknown workflow.policy"):
        get_policy_adapter("missing_family")


def test_family_interface_is_checked(monkeypatch):
    module = types.ModuleType("head.model.invalid.adapter")
    module.Adapter = object
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError, match="BasePolicyAdapter"):
        get_policy_adapter("invalid")


@pytest.mark.parametrize("policy,runner", [(object, "policy"), (ExamplePolicy, "unknown")])
def test_invalid_binding_is_rejected(policy, runner):
    with pytest.raises((TypeError, ValueError)):
        PolicyBinding(policy, runner)


def test_old_public_imports_resolve_to_same_classes():
    from head.model import BaseAdapter, ModelInput, get_adapter_class
    from head.model.imitation import BaseAdapter as NewBase, ModelInput as NewInput
    from head.model.imitation.wayformer.adapter import Adapter as WayformerAdapter
    from head.policy.evolvable_policy.poly_planning_policy import RLPlanningPolicy as OldPoly
    from head.model.poly.policy import RLPlanningPolicy
    from head.manager.base_algorithm_selector import ZeroPolicy as OldZero, RandomPolicy as OldRandom
    from head.model.zero.policy import ZeroPolicy
    from head.model.poly.adapter import RandomPolicy
    assert BaseAdapter is NewBase and ModelInput is NewInput
    assert get_adapter_class("wayformer") is WayformerAdapter
    assert OldPoly is RLPlanningPolicy
    assert OldZero is ZeroPolicy and OldRandom is RandomPolicy


def test_zero_keeps_distinct_mode_contracts(monkeypatch):
    from head.model.zero.policy import ZeroPolicy
    monkeypatch.setattr(sys, "argv", ["main_head.py", "workflow.policy=Zero"])
    args = get_final_config().args
    adapter = get_policy_adapter("Zero")
    assert adapter.resolve(args, "deploy") == PolicyBinding(ZeroPolicy, "policy")
    assert adapter.resolve(args, "evolution") == PolicyBinding(EnvInputPolicy, "learner")
    policy = ZeroPolicy.__new__(ZeroPolicy)
    policy.action_info = {}
    assert policy.act("ego") == [0.0, 0.0]


def test_family_discovery_does_not_import_networks():
    import subprocess
    code = (
        "import sys; from head.model import get_policy_adapter; "
        "[get_policy_adapter(n) for n in ['imitation', 'Poly', 'Zero', 'IDM']]; "
        "assert 'head.model.poly.policy' not in sys.modules; "
        "assert 'head.model.imitation.wayformer.model' not in sys.modules; "
        "assert 'head.model.imitation.pluto.model.pluto_model' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_imitation_models_are_nested_under_family():
    root = Path(__file__).parents[1] / "head/model"
    for model in ("pluto", "wayformer"):
        assert (root / "imitation" / model / "adapter.py").is_file()
        assert not (root / model / "adapter.py").exists()
    for family in ("imitation", "poly", "zero", "idm"):
        assert (root / family / "adapter.py").is_file()


def test_poly_configuration_moves_with_policy():
    from head.model.poly import policy
    config = Path(policy.__file__).parent / "common/cfgs/config.yaml"
    assert config.is_file()
