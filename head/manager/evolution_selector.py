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
from head.model import resolve_policy_binding
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
    模型专用处理由 head.model.imitation 下的适配器负责，公共流程无需判断模型名称。
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
        仿真状态由统一 Adapter 接口完成输入处理与推理，公共控制器执行轨迹，
        evaluation 记录实际闭环表现；此处不包含模型专用逻辑。
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
    """Choose an execution path from the family binding, not the algorithm name."""
    binding = resolve_policy_binding(cfg)
    if binding.runner == "policy":
        return NoEvolutionStrategy
    if binding.runner == "imitation":
        return ImitationStrategy
    selected = cfg.args.workflow.evolution
    main, sub = selected.strategy, selected.learner
    strategy_class = EVOLUTION_STRATEGY_MAPPING.get(main, {}).get(sub)
    if strategy_class is None:
        raise ValueError(f"策略 '{main}/{sub}' 尚未实现,请检查配置或扩展映射表。")
    print(f"[信息] 已选择进化策略:{main}/{sub}")
    return strategy_class
