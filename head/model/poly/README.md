# Poly

使用目标横向位置、速度和规划时长生成多项式轨迹，并由 PID 控制车辆。

| 文件或目录 | 用途 |
| --- | --- |
| [policy.py](policy.py) | 轨迹规划与车辆控制 |
| [common/](common/) | Frenet 规划器、控制工具与参数 |
| [adapter.py](adapter.py) | SAC 目标参数接入及无权重时的随机动作回退 |
