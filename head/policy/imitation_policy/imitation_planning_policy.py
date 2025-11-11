"""
作者: ShuoYang
日期: 2025-11-10
描述: imitation_planning_policy.py - 模仿学习规划策略
"""

from metadrive.policy.base_policy import BasePolicy


class ImitationPlanningPolicy(BasePolicy):
    """
    模仿学习规划策略类,用于在MetaDrive环境中使用模仿学习模型。
    这是一个占位实现,实际的推理逻辑需要在ImitationStrategy中实现。
    """
    def __init__(self, control_object, random_seed):
        super(ImitationPlanningPolicy, self).__init__(control_object, random_seed)
