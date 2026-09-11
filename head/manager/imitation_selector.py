"""Compatibility bare-network builder using the pluggable agent contract."""
from omegaconf import OmegaConf
from head.agents import get_agent_class


def resolve_imitation_strategy(cfg):
    imitation = cfg.args.workflow.policies.imitation
    cls = get_agent_class(str(imitation.model))
    config = OmegaConf.merge(cls.load_config(), imitation.get("options", {}))
    agent = cls(config)
    return agent.build_model(config), config
