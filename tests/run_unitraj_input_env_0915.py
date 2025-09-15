"""
Author: ShuoYang, ShuaixiPan
Date: 2025-07-10, 0913
Description: main_head.py, 之前基础上封装函数等
"""
import time
import os
os.environ["MPLBACKEND"] = "Agg"   # 离线保存最稳
from datetime import datetime
import torch
# main_head.py
from collections import defaultdict
from head.manager.config_manager import get_final_config
from head.manager.imitation_selector import resolve_imitation_strategy
from head.evolution_engine.env_builder.env import make_env
from head.policy.imitation_policy.dataset_builder.common_utils import get_polyline_dir, find_true_segments, generate_mask, is_ddp, \
    get_kalman_difficulty, get_trajectory_type, interpolate_polyline
import numpy as np
from metadrive.scenario.scenario_description import MetaDriveType
from head.policy.imitation_policy.dataset_builder.types import object_type, polyline_type
default_value = 0
object_type = defaultdict(lambda: default_value, object_type)
polyline_type = defaultdict(lambda: default_value, polyline_type)
from head.policy.imitation_policy.dataset_builder import common_utils
from head.policy.imitation_policy.utils import visualization
from head.policy.imitation_policy.utils.map_utils import get_map_data, get_manually_split_map_data
from head.policy.imitation_policy.utils.agent_utils import transform_trajs_to_center_coords, get_agent_data, get_interested_agents, trajectory_filter
from head.policy.imitation_policy.utils.inference_engine import UnitrajInference



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


# --- 主脚本的常量和设置 ---
SAVE_DIR = "/home/dw/HEAD/tests/figure"
os.makedirs(SAVE_DIR, exist_ok=True)
# 你想在最终的可视化图片中查看第几个样本
DRAW_INDEX = 1

if __name__ == '__main__':
    # 加载项目配置
    cfg = get_final_config()

    # 1. 初始化推理引擎
    print("正在初始化推理引擎...")
    inference_engine = UnitrajInference(cfg)

    # 2. 运行推理并获取最终结果
    print("开始运行推理...")
    last_batch_dict, last_prediction = inference_engine.run_inference(
        to_device_func=to_device,
        num_iterations=20
    )

    # 3. 使用结果进行可视化
    if (last_batch_dict is not None) and (last_prediction is not None):
        print("推理完成，正在生成可视化结果...")
        # 生成带时间戳的文件名，避免覆盖
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        final_path = os.path.join(SAVE_DIR, f"prediction_vs_gt_{ts}.jpg")

        visualization.visualize_prediction(
            last_batch_dict,
            last_prediction,
            draw_index=DRAW_INDEX,
            save_path=final_path,
            rotate=180
        )
        print(f"✅ 可视化图片已保存至: {final_path}")
    else:
        print("未能获取有效的推理结果，无法生成可视化图片。")