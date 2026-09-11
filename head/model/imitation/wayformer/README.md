# WayFormer

WayFormer 的网络实现、场景特征处理和轨迹预测代码。

| 文件 | 用途 |
| --- | --- |
| [model.py](model.py) | 网络前向计算 |
| [layers.py](layers.py) | 编码器、解码器与注意力层 |
| [features.py](features.py) | 历史窗口与模型输入构建 |
| [adapter.py](adapter.py) | 加载权重，选取最高概率轨迹并转换坐标 |
| [agent_utils.py](agent_utils.py)、[map_utils.py](map_utils.py) | 交通参与者与地图数据处理 |
| [config.yaml](config.yaml) | 模型与推理参数 |
