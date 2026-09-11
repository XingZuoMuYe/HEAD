# 模仿学习模型

存放 Pluto、WayFormer 等模型的网络、输入预处理和轨迹输出适配代码。

| 文件或目录 | 用途 |
| --- | --- |
| [pluto/](pluto/) | Pluto 网络、特征处理与规划推理 |
| [wayformer/](wayformer/) | WayFormer 网络、特征处理与轨迹预测 |
| [common/](common/) | 共享的数据处理工具与配置 |
| [base.py](base.py) | 模型输入、轨迹输出和适配器接口 |
| [loader.py](loader.py) | 根据配置加载模型适配器 |
| [adapter.py](adapter.py) | 将模仿学习类别接入 HEAD 策略流程 |
