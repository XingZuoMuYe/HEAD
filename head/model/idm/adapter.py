"""Use the same MetaDrive IDM policy in both existing workflow modes."""
from head.model.base import BasePolicyAdapter, PolicyBinding


class Adapter(BasePolicyAdapter):
    @classmethod
    def resolve(cls, args, mode):
        from metadrive.policy.idm_policy import IDMPolicy
        return PolicyBinding(IDMPolicy, "learner" if mode == "evolution" else "policy")
