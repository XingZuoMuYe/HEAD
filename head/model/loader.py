"""Lazy strategy-family discovery and trajectory-model compatibility exports."""
import importlib
import re
from .base import BasePolicyAdapter, PolicyBinding
from .imitation.loader import create_adapter, get_adapter_class


def get_policy_adapter(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("workflow.policy must name a strategy family")
    if ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid workflow.policy {name!r}")
        module_name, class_name = f"head.model.{name.lower()}.adapter", "Adapter"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(exc.name + "."):
            raise ValueError(
                f"Unknown workflow.policy {name!r}; add head/model/<family>/adapter.py"
            ) from exc
        raise
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, BasePolicyAdapter):
        raise TypeError(f"{module_name}:{class_name} must inherit BasePolicyAdapter")
    return cls


def resolve_policy_binding(cfg):
    mode = cfg.args.workflow.type
    if mode == "evo":
        mode = "evolution"
    if mode not in {"deploy", "evolution"}:
        raise ValueError("workflow.type must be 'deploy' or 'evolution'")
    binding = get_policy_adapter(cfg.args.workflow.policy).resolve(cfg.args, mode)
    if not isinstance(binding, PolicyBinding):
        raise TypeError("Strategy Adapter.resolve must return PolicyBinding")
    return binding
