"""
作者: ShuoYang
日期: 2025-07-10
描述: evolution_selector.py - 解析并实例化选定的进化策略。
"""
from datetime import datetime
import os

from head.evolution_engine.RLBoost.SAC.SAC_learner import SAC_Learner, SACConfig
from head.evolution_engine.env_builder.env import make_env
import torch
from head.policy.imitation_policy.utils.inference_engine import UnitrajInference
from head.policy.imitation_policy.utils import visualization

SAVE_DIR = "/home/test/git_shuo/HEAD/head/policy/imitation_policy/figure"
os.makedirs(SAVE_DIR, exist_ok=True)
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"


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
        print("[信息] 当前部署方法不需要进化策略")
    
    def agent_initialize(self):
        """部署模式下初始化环境"""
        self.env = make_env(self.cfg)
        print('[信息] 环境已初始化')
    
    def train(self):
        """部署模式下不需要训练"""
        pass
    
    def eval(self):
        """部署模式下执行评估,运行环境step循环"""
        if self.env is None:
            print("[警告] 环境未初始化,请先调用agent_initialize()")
            return
        
        print('[信息] 开始评估')
        eval_eps = self.cfg.args.misc.eval_episodes if hasattr(self.cfg.args, 'misc') else 1
        eps_max_steps = 2000
        
        for i_ep in range(eval_eps):
            state, _ = self.env.reset()
            ep_reward = 0.0
            ep_len = 0
            
            for i_step in range(eps_max_steps):
                # 使用环境的agent进行决策
                action = self.env.action_space.sample()
                next_state, reward, done, termin, info = self.env.step(action)
                
                # 渲染
                if self.cfg.args.training.show_render_info:
                    self._render()
                
                state = next_state
                ep_reward += reward
                ep_len += 1
                
                if done or termin:
                    print(f"Episode:{i_ep + 1}/{eval_eps}, Reward:{ep_reward:.3f}, Length:{ep_len}")
                    break
        
        print('[信息] 评估完成')
    
    def _render(self):
        """渲染环境"""
        if self.cfg.args.task == 'straight_config_traffic-v0':
            self.env.head_renderer.render(mode="topdown",
                            screen_record=False,
                            scaling=6,
                            film_size=(6000, 400),
                            show_plan_traj=True,
                            )
        elif self.cfg.args.task == 'muti_scenario-v0' or self.cfg.args.task == 'single_scenario-v0':
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[信息] 初始化模仿学习策略")

    def agent_initialize(self):
        """初始化环境和模仿学习模型"""
        from head.manager.imitation_selector import resolve_imitation_strategy

        # 初始化环境
        self.env = make_env(self.cfg)

        # 加载模仿学习模型
        self.model, self.imitation_cfg = resolve_imitation_strategy(self.cfg)
        self.inference_engine = UnitrajInference(self.imitation_cfg)
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
        if self.env is None or self.model is None:
            print("[警告] 环境或模型未初始化,请先调用agent_initialize()")
            return

        print('[信息] 开始推理')

        last_batch_dict, last_prediction = self.inference_engine.run_inference_step(to_device_func=to_device)
        if self.cfg.args.training.show_render_info:
            self._render()

        if (last_batch_dict is not None) and (last_prediction is not None):
            print("推理完成，正在生成可视化结果...")
            # 生成带时间戳的文件名，避免覆盖
            ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            final_path = os.path.join(SAVE_DIR, f"prediction_vs_gt_{ts}.jpg")

            visualization.visualize_prediction(
                last_batch_dict,
                last_prediction,
                save_path=final_path,
                rotate=180
            )
            print(f"✅ 可视化图片已保存至: {final_path}")

        print('[信息] 评估完成')
        print('[信息] 评估完成')
        # 3. 使用结果进行可视化


    def load(self):
        """加载模型权重"""
        # TODO: 实现模型权重加载逻辑
        ckpt_path = self.cfg.args.algorithm['deployment']['config']['imitation']['model']['pretrained_path']
        ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model = self.model.to(self.device)
        self.inference_engine.init_env_and_model(self.env, self.model)

def resolve_evolution_strategy(cfg):
    """
    根据配置选择对应的策略类。
    - 如果 algorithm.mode 为 'deployment' 且部署方法为 IDM,返回NoEvolutionStrategy
    - 如果 algorithm.mode 为 'deployment' 且部署方法为 imitation,返回ImitationStrategy
    - 如果 algorithm.mode 为 'deployment' 且部署方法为 Poly,走正常进化流程
    - 如果 algorithm.mode 为 'evolutionary',根据 evolution_method_type 选择对应的策略类
    """
    mode = getattr(cfg.args.algorithm, "mode", None)
    
    # 部署模式
    if mode == "deployment":
        deployment_method = cfg.args.algorithm.deployment.deployment_method.get('main')
        
        # 根据不同的部署方法返回不同的策略
        if deployment_method == 'IDM':
            print(f"[信息] 检测到部署模式,部署方法为:IDM,不使用进化策略")
            return NoEvolutionStrategy
        elif deployment_method == 'imitation':
            print(f"[信息] 检测到部署模式,部署方法为:imitation,使用模仿学习策略")
            return ImitationStrategy
        elif deployment_method == 'Poly':
            print(f"[信息] 检测到部署模式,部署方法为:Poly,需要使用进化策略")
            # 继续往下执行,走evolutionary的逻辑
        else:
            raise ValueError(f"未知的部署方法 '{deployment_method}'")
    
    # 进化模式 或 需要进化的部署模式
    if mode == "evolutionary" or (mode == "deployment"):
        sel_cfg = cfg.args.algorithm.evolutionary['evolution_method_type']
        main = sel_cfg.get('main')
        sub = sel_cfg.get('sub')
        candidates = sel_cfg.get('candidates', {})

        # 基本合法性检查
        if main not in candidates:
            raise ValueError(f"主策略 '{main}' 无效,可选项为: {list(candidates.keys())}")
        if sub not in candidates[main]:
            raise ValueError(f"子策略 '{sub}' 不属于主策略 '{main}' 的候选范围,可选项为: {candidates[main]}")

        # 映射表检查
        if main not in EVOLUTION_STRATEGY_MAPPING or sub not in EVOLUTION_STRATEGY_MAPPING[main]:
            raise ValueError(f"策略 '{main}/{sub}' 尚未实现,请检查配置或扩展映射表。")

        # 返回策略类
        StrategyClass = EVOLUTION_STRATEGY_MAPPING[main][sub]
        print(f"[信息] 已选择进化策略:{main}/{sub}")
        return StrategyClass
    
    else:
        raise ValueError(f"无效的算法模式 '{mode}',必须是 'evolutionary' 或 'deployment'")
