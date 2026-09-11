"""Convention-based lazy discovery: head.agents.<name>.agent:Agent."""
import importlib
import re
from omegaconf import OmegaConf
from .base import BaseAgent


def get_agent_class(name):
    """Also accept 'my_package.my_agent:MyAgent' for external plugins."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("workflow.policies.imitation.model must name an agent")
    if ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid agent name: {name!r}")
        module_name, class_name = f"head.agents.{name.lower()}.agent", "Agent"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(exc.name + "."):
            raise ValueError(
                f"Unknown agent {name!r}. Add head/agents/<name>/agent.py with "
                "an Agent class, or use package.module:ClassName."
            ) from exc
        raise
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type) or not issubclass(cls, BaseAgent):
        raise TypeError(f"{module_name}:{class_name} must inherit BaseAgent")
    return cls


def create_agent(imitation_config, checkpoint, device="cpu"):
    cls = get_agent_class(str(imitation_config.model))
    config = OmegaConf.merge(cls.load_config(), imitation_config.get("options", {}))
    config["sae"] = OmegaConf.to_container(imitation_config.get("sae", OmegaConf.create({})))
    agent = cls(config, device=device)
    agent.initialize(checkpoint)
    if agent.input_scope != "history_only":
        import warnings
        warnings.warn(
            f"Agent input scope: {agent.input_scope}. This legacy adapter can "
            "use recorded future context; do not label it history-only evaluation.",
            UserWarning,
        )
    return agent
