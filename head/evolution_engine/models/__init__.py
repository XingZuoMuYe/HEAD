from head.evolution_engine.models.autobot.autobot import AutoBotEgo
from head.evolution_engine.models.mtr.MTR import MotionTransformer
from head.evolution_engine.models.wayformer.wayformer import Wayformer

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
