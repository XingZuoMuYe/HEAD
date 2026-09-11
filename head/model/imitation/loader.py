"""Convention-based lazy discovery: head.model.imitation.<name>.adapter:Adapter."""
import importlib
import re
from omegaconf import OmegaConf
from .base import BaseAdapter


def get_adapter_class(name):
    """Also accept 'my_package.my_adapter:MyAdapter' for external plugins."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("workflow.policies.imitation.model must name an adapter")
    if ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid adapter name: {name!r}")
        module_name, class_name = f"head.model.imitation.{name.lower()}.adapter", "Adapter"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(exc.name + "."):
            raise ValueError(
                f"Unknown adapter {name!r}. Add head/model/imitation/<name>/adapter.py with "
                "an Adapter class, or use package.module:ClassName."
            ) from exc
        raise
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, BaseAdapter):
        raise TypeError(f"{module_name}:{class_name} must inherit BaseAdapter")
    return cls


def create_adapter(imitation_config, checkpoint, device="cpu"):
    cls = get_adapter_class(str(imitation_config.model))
    config = OmegaConf.merge(cls.load_config(), imitation_config.get("options", {}))
    config["sae"] = OmegaConf.to_container(imitation_config.get("sae", OmegaConf.create({})))
    adapter = cls(config, device=device)
    adapter.initialize(checkpoint)
    if adapter.input_scope != "history_only":
        import warnings
        warnings.warn(
            f"Adapter input scope: {adapter.input_scope}. This legacy adapter can "
            "use recorded future context; do not label it history-only evaluation.",
            UserWarning,
        )
    return adapter
