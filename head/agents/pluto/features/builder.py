from torch.utils.data import Dataset

from head.agents.pluto.features.dataset import PlutoDataset


class PlutoTestDataset(Dataset):
    """
    Dataset class for Pluto feature format reading NuPlan db files (ScenarioDescription).
    """

    def __init__(self, config=None, is_validation=True):
        if config is None:
            config = {}
        self._builder = PlutoDataset(config=config, is_validation=is_validation)

    def process_scenario(self, scenario, current_step, controlled_agent_id=None):
        output = self.process_scenario_data(
            scenario,
            current_step,
            controlled_agent_id=controlled_agent_id,
        )
        return output

    def process_scenario_data(self, scenario, current_step, controlled_agent_id=None):
        intermediate = self._builder.preprocess(
            scenario,
            current_step=current_step,
            controlled_agent_id=controlled_agent_id,
        )
        pluto_feature = self._builder.process(
            intermediate,
            current_step=current_step,
            controlled_agent_id=controlled_agent_id,
        )
        output = self._builder.postprocess(pluto_feature)
        return output
