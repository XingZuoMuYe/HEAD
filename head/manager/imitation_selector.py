"""Compatibility bare-network builder using the pluggable adapter contract."""
from omegaconf import OmegaConf
from head.model.imitation import get_adapter_class


def resolve_imitation_strategy(cfg):
    imitation = cfg.args.workflow.policies.imitation
    cls = get_adapter_class(str(imitation.model))
    config = OmegaConf.merge(cls.load_config(), imitation.get("options", {}))
    adapter = cls(config)
    return adapter.build_model(config), config
