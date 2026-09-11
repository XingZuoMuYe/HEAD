"""Preserve Zero's different deployment and evolution contracts."""
from head.model.base import BasePolicyAdapter, PolicyBinding


class Adapter(BasePolicyAdapter):
    @classmethod
    def resolve(cls, args, mode):
        if mode == "evolution":
            from metadrive.policy.env_input_policy import EnvInputPolicy
            return PolicyBinding(EnvInputPolicy, "learner")
        from .policy import ZeroPolicy
        return PolicyBinding(ZeroPolicy)
