"""Select Poly's original policy or its explicit no-checkpoint fallback."""
from metadrive.policy.env_input_policy import EnvInputPolicy
from head.model.base import BasePolicyAdapter, PolicyBinding


class RandomPolicy(EnvInputPolicy):
    """Explicit deploy fallback when Poly has no checkpoint."""
    def act(self, agent_id):
        action = self.get_input_space().sample()
        self.action_info["action"] = action
        return action


class Adapter(BasePolicyAdapter):
    supports_auto_checkpoint = True

    @classmethod
    def resolve(cls, args, mode):
        from head.manager.artifact_paths import has_poly_checkpoint
        if mode == "deploy" and not has_poly_checkpoint(args):
            print("[警告] deploy + Poly 未找到 checkpoint，使用 action_space.sample()")
            return PolicyBinding(RandomPolicy)
        from .policy import RLPlanningPolicy
        return PolicyBinding(RLPlanningPolicy, "learner")
