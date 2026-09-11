# 闭环评价

计算闭环驾驶的 TTC、舒适性和综合分，结合碰撞、越界与到达终点事件评价模型表现。

| 文件 | 用途 |
| --- | --- |
| [closed_loop_metrics.py](closed_loop_metrics.py) | 从实际运动状态计算 TTC 与舒适性 |
| [compute_puffer_nuplan_style_scores.py](compute_puffer_nuplan_style_scores.py) | 综合评分与结果后处理 |
| [native_evaluation.py](native_evaluation.py) | 原生评测工具 |
| [test_puffer_nuplan_style_scores.py](test_puffer_nuplan_style_scores.py) | 综合评分测试 |

MetaDrive 状态接入位于 [head/manager/evaluation_v2.py](../head/manager/evaluation_v2.py)。
