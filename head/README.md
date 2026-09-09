# 仿真、规划与模型评测

HEAD 的核心源码：负责配置加载、场景构建、策略推理、车辆控制、闭环评测与进化训练。

| 目录 | 用途 |
| --- | --- |
| [`configs/`](configs/) | 默认配置与各类场景任务参数 |
| [`envs/`](envs/) | MetaDrive 仿真环境与实车日志场景环境 |
| [`component/`](component/) | 地图、车道、导航及交通灯组件 |
| [`policy/`](policy/) | 基础策略、可进化策略、Pluto/WayFormer 推理接入与轨迹控制 |
| [`manager/`](manager/) | 配置校验、策略选择、运行组织与评价接口 |
| [`evolution_engine/`](evolution_engine/) | 环境封装、RLBoost/SAC 与训练公共组件 |
| [`renderer/`](renderer/) | 场景渲染与俯视图可视化 |
| [`scenario_datasets/`](scenario_datasets/) | 仓库自带的场景压缩包及解压位置 |
| [`scenario_reproduction/`](scenario_reproduction/) | 实车日志与地图数据转换工具 |
| [`scripts/`](scripts/) | 运行入口 `main_head.py` |

闭环指标的底层定义集中在根目录的 [`evaluation/`](../evaluation/README.md)，本目录负责将仿真状态接入评价。

从仓库根目录、在已配置的环境中运行 `python -m head.scripts.main_head`。
安装、模型下载和任务参数见[项目说明](../README.md)。
