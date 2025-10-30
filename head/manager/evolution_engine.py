from head.policy.evolvable_policy.poly_planning_policy import RLPlanningPolicy
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.idm_policy import IDMPolicy

DEPLOYMENT_POLICY_MAPPING = {
    'IDM': IDMPolicy,
    'Poly': RLPlanningPolicy,
    # 更多可扩展项...
}

class evolution_engine(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.evolution_algo = None

    def start(self,evolution_algo):
        self.evolution_algo = evolution_algo

    def train(self):
        self.evolution_algo.train()

    def eval(self):
        if self.cfg.args.algorithm['deployment']['deployment_method']['main'] ==  'Poly':
            self.evolution_algo.load()
        self.evolution_algo.eval()