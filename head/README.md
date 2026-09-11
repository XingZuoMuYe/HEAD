# HEAD 核心代码

负责场景构建、驾驶策略推理、车辆控制、闭环评测与进化训练。

| 目录 | 用途 |
| --- | --- |
| [model/](model/) | Pluto、WayFormer 等模型实现与适配 |
| [configs/](configs/) | 运行参数与场景任务配置 |
| [envs/](envs/)、[component/](component/) | 仿真环境、地图与交通组件 |
| [policy/](policy/) | 驾驶策略与轨迹控制器 |
| [manager/](manager/) | 配置校验、策略调度与评价接入 |
| [evolution_engine/](evolution_engine/) | 进化训练与 RLBoost/SAC |
| [renderer/](renderer/) | 场景渲染与可视化 |
| [scenario_datasets/](scenario_datasets/) | 场景数据 |
| [scenario_reproduction/](scenario_reproduction/) | 日志与地图转换工具 |
| [scripts/](scripts/) | 项目运行入口 |
