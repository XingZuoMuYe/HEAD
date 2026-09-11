# Pluto

Pluto 的网络实现、场景特征处理和规划推理代码。

| 文件或目录 | 用途 |
| --- | --- |
| [model/](model/) | 神经网络及各层模块 |
| [features/](features/) | 场景输入预处理与特征构建 |
| [planner.py](planner.py) | 候选轨迹生成与选择 |
| [adapter.py](adapter.py) | 加载权重，将模型输出转换为控制器轨迹 |
| [config.yaml](config.yaml) | 模型与推理参数 |
| [trajectory_evaluator.py](trajectory_evaluator.py) | 可选的候选轨迹混合评分 |
