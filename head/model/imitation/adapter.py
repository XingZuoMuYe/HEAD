"""Connect trajectory models to the existing imitation policy."""
from head.model.base import BasePolicyAdapter, PolicyBinding


class Adapter(BasePolicyAdapter):
    @classmethod
    def validate(cls, args, selected):
        super().validate(args, selected)
        if args.task != "real_scenario-v0" or not args.scenario.capabilities.closed_loop_imitation:
            raise ValueError(
                "workflow.policy=imitation is only supported by real_scenario-v0 "
                "with scenario.capabilities.closed_loop_imitation=true"
            )
        from .loader import get_adapter_class
        get_adapter_class(selected.get("model"))
        if not selected.get("checkpoint"):
            raise ValueError("workflow.policies.imitation.checkpoint is required")
        if int(selected.get("warmup_steps", 0)) < 0:
            raise ValueError("workflow.policies.imitation.warmup_steps must be non-negative")
        if int(selected.get("replan_frequency", 1)) < 1:
            raise ValueError("workflow.policies.imitation.replan_frequency must be at least 1")

    @classmethod
    def resolve(cls, args, mode):
        from head.policy.imitation_policy.imitation_planning_policy import ImitationPlanningPolicy
        return PolicyBinding(ImitationPlanningPolicy, "learner" if mode == "evolution" else "imitation")
