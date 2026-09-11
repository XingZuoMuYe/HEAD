"""Strategy-family interface at HEAD's existing MetaDrive policy boundary."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .imitation.base import BaseAdapter, ModelInput, Trajectory


@dataclass(frozen=True)
class PolicyBinding:
    """A simulator policy and execution path, independent of family names.

    policy: internal control; imitation: recorded-history trajectory inference;
    learner: configured evolution learner supplies the policy's input action.
    Each policy retains its own get_input_space/act contract and controller.
    """
    policy_class: type
    runner: str = "policy"

    def __post_init__(self):
        from metadrive.policy.base_policy import BasePolicy
        if not isinstance(self.policy_class, type) or not issubclass(self.policy_class, BasePolicy):
            raise TypeError("policy_class must inherit MetaDrive BasePolicy")
        if self.runner not in {"policy", "imitation", "learner"}:
            raise ValueError(f"Unsupported execution path: {self.runner}")


class BasePolicyAdapter(ABC):
    """A strategy family exports Adapter in head/model/<family>/adapter.py."""
    supports_auto_checkpoint = False

    @classmethod
    def validate(cls, args, selected):
        if selected.get("checkpoint") == "auto" and not cls.supports_auto_checkpoint:
            raise ValueError(f"workflow.policies.{args.workflow.policy}.checkpoint=auto is not supported")

    @classmethod
    @abstractmethod
    def resolve(cls, args, mode):
        """Return PolicyBinding without creating a simulator or loading weights."""
