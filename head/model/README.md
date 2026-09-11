# 驾驶策略

存放驾驶策略及其接入代码，供 HEAD 仿真和评测使用。

| 目录或文件 | 用途 |
| --- | --- |
| [imitation/](imitation/) | Pluto、WayFormer 等模仿学习模型 |
| [poly/](poly/) | 多项式轨迹规划与 SAC 目标参数接入 |
| [zero/](zero/) | 零控制指令基线与直接控制接入 |
| [idm/](idm/) | MetaDrive IDM 规则策略接入 |
| [base.py](base.py)、[loader.py](loader.py) | 策略适配接口与配置加载 |
