from metadrive.policy.base_policy import BasePolicy


class ZeroPolicy(BasePolicy):
    """Deployment baseline that always emits a zero control action."""
    def act(self, agent_id):
        action = [0.0, 0.0]
        self.action_info["action"] = action
        return action
