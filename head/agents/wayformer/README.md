# WayFormer

HEAD 的 WayFormer 模型、特征构建与闭环适配实现。

| 文件 | 功能 |
| --- | --- |
| [agent.py](agent.py) | 权重加载、最高概率模式选择、局部到世界坐标转换 |
| [model.py](model.py) | WayFormer 网络前向计算 |
| [layers.py](layers.py) | Perceiver 编码器、解码器及注意力模块 |
| [features.py](features.py) | 当前滑动历史窗口；不截断仿真器时间戳 |
| [config.yaml](config.yaml) | 与发布 checkpoint 对应的网络参数 |

[模型权重页面](https://huggingface.co/GALLERVICH/WayFormer-head)：
[brier_fde=1.45.ckpt](https://huggingface.co/GALLERVICH/WayFormer-head/resolve/main/brier_fde%3D1.45.ckpt)。
权重不进入 Git，推荐路径为 `artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt`。

从仓库根目录、在 Conda 环境中运行：

```bash
python -m head.scripts.main_head \
  task=real_scenario-v0 workflow.policy=imitation \
  workflow.policies.imitation.model=wayformer \
  'workflow.policies.imitation.checkpoint=artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt' \
  simulation.render=false evaluation.max_steps=120
```

WayFormer 输出多模态运动预测；HEAD 选择网络概率最高的一条轨迹，转世界坐标，
再由公共控制器执行。这是“WayFormer + HEAD 控制器”的闭环系统，
不能称为 WayFormer 原论文的官方规划评测。
高斯分布参数不作速度使用，保持公共控制器的几何速度估计。

源码、许可证和新增算法约定见[算法接入说明](../README.md)。
