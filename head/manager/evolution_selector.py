"""
作者: ShuoYang
日期: 2025-07-10
描述: evolution_selector.py - 解析并实例化选定的进化策略。
"""
from datetime import datetime
import os
from pathlib import Path

from head.evolution_engine.RLBoost.SAC.SAC_learner import SAC_Learner, SACConfig
from head.evolution_engine.env_builder.env import make_env
from head.manager.artifact_paths import has_poly_checkpoint
from head.manager.closed_loop_metrics import ClosedLoopMetricsRecorder
import torch

SAVE_DIR = Path(__file__).resolve().parent.parent / "policy" / "imitation_policy" / "figure"
os.makedirs(SAVE_DIR, exist_ok=True)
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def _closed_loop_metrics_for(cfg):
    """Create the common recorder for every environment evaluation."""
    return ClosedLoopMetricsRecorder(cfg)


# 策略映射表(可扩展)
EVOLUTION_STRATEGY_MAPPING = {
    'RLBoost': {
        'SAC': SAC_Learner,
        'PPO': None,
    },
    'DreamMethod': {
        'HeadMethodInDream': None,
    }
}


def to_device(batch, device="cuda"):
    """
    递归地将数据（tensors, dicts, lists）移动到指定的设备（如GPU）。
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [to_device(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(to_device(v, device) for v in batch)
    else:
        return batch

class NoEvolutionStrategy:
    """
    占位类,用于不需要进化策略的部署场景(如IDM)。
    提供与进化策略类相同的基本接口,但不执行任何实际操作。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.env = None
        self.closed_loop_metrics = None
        print("[信息] 当前部署基础策略不需要进化策略")
    
    def agent_initialize(self):
        """部署模式下初始化环境"""
        self.env = make_env(self.cfg)
        self.closed_loop_metrics = _closed_loop_metrics_for(self.cfg)
        print('[信息] 环境已初始化')
    
    def train(self):
        """部署模式下不需要训练"""
        pass
    
    def eval(self):
        """部署模式下执行评估,运行环境step循环"""
        if self.env is None:
            print("[警告] 环境未初始化,请先调用agent_initialize()")
            return
        
        print('[信息] 开始闭环评测')
        eval_eps = self.cfg.args.evaluation.episodes
        eps_max_steps = self.cfg.args.evaluation.max_steps
        
        for i_ep in range(eval_eps):
            state, _ = self.env.reset()
            if self.closed_loop_metrics is not None:
                self.closed_loop_metrics.start_episode()
            ep_reward = 0.0
            ep_len = 0
            
            for i_step in range(eps_max_steps):
                # 使用环境的agent进行决策
                # The environment's configured policy (IDM/Poly/imitation)
                # computes the action internally. Passing None avoids replacing
                # that policy action with a random sample.
                action = None
                next_state, reward, done, termin, info = self.env.step(action)
                if self.closed_loop_metrics is not None:
                    self.closed_loop_metrics.step(info, next_state, i_step, self.env)
                
                # 渲染
                if self.cfg.args.simulation.render:
                    self._render()
                
                state = next_state
                ep_reward += reward
                ep_len += 1
                
                if done or termin:
                    break
            if self.closed_loop_metrics is not None:
                self.closed_loop_metrics.finish_episode(
                    episode_index=i_ep + 1,
                    reward=ep_reward,
                    length=ep_len,
                    env=self.env,
                )
        
        print('[信息] 闭环评测完成')
        if self.closed_loop_metrics is not None:
            metrics_path = self.closed_loop_metrics.save()
            if metrics_path is not None:
                print(f"[信息] 闭环指标已保存: {metrics_path}")
        self.env.close()
    
    def _render(self):
        """渲染环境"""
        if self.cfg.args.task == 'straight_config_traffic-v0':
            self.env.head_renderer.render(mode="topdown",
                            screen_record=False,
                            scaling=6,
                            film_size=(6000, 400),
                            show_plan_traj=True,
                            )
        elif self.cfg.args.task in ['multi_scenario-v0', 'muti_scenario-v0', 'single_scenario-v0']:
            self.env.head_renderer.render(mode="topdown",
                            screen_record=False,
                            show_plan_traj=True,
                            )
        elif self.cfg.args.task == 'real_scenario-v0':
            self.env.head_renderer.render(mode="topdown",
                            show_plan_traj=True,
                            show_agent_name=False,
                            film_size=(5500, 5500),
                            scaling=3,
                            screen_size=(800, 800),
                            screen_record=False,
                            )
    
    def load(self):
        """部署模式下不需要加载"""
        pass


class ImitationStrategy(NoEvolutionStrategy):
    """
    模仿学习策略类,用于加载和运行模仿学习模型。
    注意: 实际的eval逻辑需要参考UnitrajInference类实现完整的数据处理和推理流程。
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.env = None
        self.model = None
        self.imitation_cfg = None
        self.inference_engine = None
        self._UnitrajInference = None
        self._visualization = None
        requested_device = cfg.args.runtime.device
        device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device == "auto":
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("runtime.device is 'cuda', but CUDA is not available")
        self.device = torch.device(device)
        print("[信息] 初始化模仿学习策略")

    def agent_initialize(self):
        """Initialize the environment; the MetaDrive policy owns inference."""
        self.env = make_env(self.cfg)
        self.closed_loop_metrics = _closed_loop_metrics_for(self.cfg)
        policy = self.env.engine.get_policy(self.env.agents["default_agent"].name)
        self.model = getattr(policy, "model", None)
        self.max_closed_loop_steps = getattr(policy, "max_closed_loop_steps", None)
        print('[信息] 环境和模仿学习模型已初始化')

    def train(self):
        """模仿学习模式下不需要训练"""
        pass

    def eval(self):
        """
        执行模仿学习评估
        注意: 这里只是占位实现,完整的实现需要参考UnitrajInference类,
        包括scenario数据处理、agent和map数据准备、batch创建等步骤。
        """
        if self.env is None:
            print("[警告] 环境或模型未初始化,请先调用agent_initialize()")
            return

        print('[信息] 开始闭环评测')

        try:
            for i_ep in range(self.cfg.args.evaluation.episodes):
                if i_ep > 0:
                    self.env.reset()
                if self.closed_loop_metrics is not None:
                    self.closed_loop_metrics.start_episode()
                ep_reward = 0.0
                ep_len = 0
                scenario = self.env.engine.data_manager.current_scenario
                sdc_id = str(scenario["metadata"]["sdc_id"])
                scenario_steps = len(scenario["tracks"][sdc_id]["state"]["position"])
                max_steps = min(self.cfg.args.evaluation.max_steps, scenario_steps - 1)
                if self.max_closed_loop_steps is not None:
                    max_steps = min(max_steps, self.max_closed_loop_steps)
                for i_step in range(max_steps):
                    next_state, reward, done, termin, info = self.env.step(None)
                    if self.closed_loop_metrics is not None:
                        self.closed_loop_metrics.step(info, next_state, i_step, self.env)
                    ep_reward += float(reward)
                    ep_len += 1
                    if self.cfg.args.simulation.render:
                        self._render()
                    if done or termin:
                        break
                if self.closed_loop_metrics is not None:
                    self.closed_loop_metrics.finish_episode(
                        episode_index=i_ep + 1,
                        reward=ep_reward,
                        length=ep_len,
                        env=self.env,
                    )
            print('[信息] 闭环评测完成')
            if self.closed_loop_metrics is not None:
                metrics_path = self.closed_loop_metrics.save()
                if metrics_path is not None:
                    print(f"[信息] 闭环指标已保存: {metrics_path}")
        finally:
            self.env.close()
        # 3. 使用结果进行可视化


    def load(self):
        """The closed-loop policy loads its model during environment creation."""
        return None

def resolve_evolution_strategy(cfg):
    """
    根据配置选择对应的策略类。
    - deploy + IDM/Zero: NoEvolutionStrategy
    - deploy + imitation: ImitationStrategy
    - deploy + Poly: evolution learner in evaluation mode when a checkpoint exists
    - evolution + any policy: configured evolution learner
    """
    mode = cfg.args.workflow.type
    if mode == "evo":
        mode = "evolution"
    
    # 部署模式
    if mode == "deploy":
        policy = cfg.args.workflow.policy
        
        if policy == 'IDM':
            print("[信息] 检测到部署模式,基础策略为:IDM,不使用进化策略")
            return NoEvolutionStrategy
        elif policy == 'imitation':
            print("[信息] 检测到imitation基础策略,使用模仿学习策略")
            return ImitationStrategy
        elif policy == 'Poly':
            if not has_poly_checkpoint(cfg.args):
                print("[信息] deploy + Poly 未找到权重，使用随机 action_space.sample()")
                return NoEvolutionStrategy
            print("[信息] 检测到部署模式,基础策略为:Poly,需要加载进化权重")
        elif policy == 'Zero':
            print("[信息] 检测到部署模式,基础策略为:Zero,不使用进化策略")
            return NoEvolutionStrategy
        else:
            raise ValueError(f"未知的部署策略 '{policy}'")
    
    # 进化模式 或 需要进化的部署模式
    if mode == "evolution" or mode == "deploy":
        sel_cfg = cfg.args.workflow.evolution
        main = sel_cfg.strategy
        sub = sel_cfg.learner

        # 映射表检查
        if main not in EVOLUTION_STRATEGY_MAPPING or sub not in EVOLUTION_STRATEGY_MAPPING[main]:
            raise ValueError(f"策略 '{main}/{sub}' 尚未实现,请检查配置或扩展映射表。")

        # 返回策略类
        StrategyClass = EVOLUTION_STRATEGY_MAPPING[main][sub]
        print(f"[信息] 已选择进化策略:{main}/{sub}")
        return StrategyClass
    
    else:
        raise ValueError(f"无效的 workflow.type '{mode}',必须是 'evolution' 或 'deploy'")
