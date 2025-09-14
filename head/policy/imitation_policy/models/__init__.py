from head.policy.imitation_policy.models.autobot.autobot import AutoBotEgo
from head.policy.imitation_policy.models.mtr.MTR import MotionTransformer
from head.policy.imitation_policy.models.wayformer.wayformer import Wayformer

__all__ = {
    'autobot': AutoBotEgo,
    'wayformer': Wayformer,
    'MTR': MotionTransformer,
}


def build_model(config):
    model = __all__[config.method.model_name](
        config=config
    )

    return model
