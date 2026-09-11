# 驾驶算法接入

每个算法拥有独立目录，模型实现、输入预处理、配置和输出适配放在一起。
本次检查结果见[验证记录](VALIDATION.md)，包含已发现的旧版输入限制。
组织方式参考 [NAVSIM 的 Agent 接口](https://github.com/autonomousvision/navsim/blob/main/navsim/agents/abstract_agent.py)；
这里实现的是 HEAD/MetaDrive 接口，不依赖 NAVSIM，也不是 NAVSIM 评测结果。

```text
head/agents/
├── base.py                 # AgentInput、Trajectory、BaseAgent
├── loader.py               # 按模块路径加载，无模型名称白名单
├── common/                 # 共享场景预处理、配置、上游许可证
├── pluto/
│   ├── agent.py            # 加载权重、调用规划器、后轴坐标适配
│   ├── config.yaml
│   ├── model/              # Pluto 网络
│   ├── features/           # Pluto 特征构建
│   ├── planner.py          # Pluto 候选轨迹生成与选择
│   └── trajectory_evaluator.py # 可选混合选择；默认不调用
└── wayformer/
    ├── agent.py            # 加载权重、最高概率选轨迹、转世界坐标
    ├── config.yaml
    ├── model.py            # WayFormer 网络
    ├── layers.py           # Perceiver 编解码等网络层
    ├── features.py         # 可持续滑动的历史窗口，支持超过 80 帧
    ├── agent_utils.py
    └── map_utils.py
```

统一运行链路：

```text
MetaDrive 场景及已执行的自车历史
  → ClosedLoopInference → 算法 Agent → Trajectory
  → 公共 TrajectoryController → 仿真步进 → evaluation
```

公共入口是 [closed_loop_inference.py](../policy/imitation_policy/closed_loop_inference.py)。
它只处理接口、时间间隔与参考点检查，不判断模型是 Pluto 还是 WayFormer。
模型内部仍然可以拥有自己的解码器/规划器；这不构成另一套闭环运行流程。

## 旧版 Pluto 输入口径限制

**当前 Pluto 路线估计会使用整段日志的自车位置，包括当前帧之后的位置。**
因此它被明确标记为 `input_scope=legacy_logged_route`，不能把已有结果称为
严格只使用历史观测的评测。固定历史、扰动未来轨迹的检查已确认输出会变化。
本次目录/接口整理没有自行改变这项旧行为；修正需要单独变更协议并重跑基准。
WayFormer 不使用这一 Pluto 路线推断路径，已在样例上通过未来轨迹扰动不变性检查。

## 输入输出约定

| 项目 | 约定 |
| --- | --- |
| 输入 | `AgentInput(scenario, current_step)`；场景为 MetaDrive ScenarioDescription |
| 历史 | current_step 是零基当前帧；新接入 history_only 算法不得利用未来真值；Pluto 的旧例外见上文 |
| 只读要求 | 不改变仿真器拥有的轨迹或时间戳；模型私有缓存放在 Agent/特征构建器中 |
| 输出 | `Trajectory(samples, dt, reference_offset)`，只返回一条已选轨迹 |
| 坐标和单位 | 世界坐标，米；速度为 m/s；角度为弧度 |
| samples | `[T,2]` 的 x,y，或 `[T,4]` 的 x,y,vx,vy，T≥2，全部为有限值 |
| 时间 | 等间隔 dt 必须与公共控制器 dt 一致；不一致须在 Agent 内重采样 |
| 参考点 | 车体中心为 0；后轴轨迹填中心到后轴的正向距离；框架统一转换控制位置 |
| 无显式速度 | 控制器沿用原来的几何前瞻速度估计；不能把模型其他输出列当速度 |
| reset | 清理场景缓存，不重复加载权重 |

原始 ScenarioDescription 容器可能含未来记录，用于离线真值和评价；
接口不是隔离沙箱，新接入算法必须声明 input_scope、遵守历史范围，并增加未来扰动不变性测试。
当前实现面向矢量化场景；图像、激光雷达等新传感器输入需要额外数据接口，
不承诺仅改一个文件就能支持尚未提供的观测类型。

## 新增算法：不修改其他算法、控制器或指标

1. 新建 `head/agents/my_model/`，放入 `__init__.py`、`config.yaml`、`agent.py` 和网络/预处理文件。
2. `agent.py` 导出继承 `BaseAgent` 的 `Agent`，实现权重加载和 `compute_trajectory`。
3. 指定 `workflow.policies.imitation.model=my_model` 与 checkpoint，即走公共闭环入口。
   也可指定 `model=my_package.agent:MyAgent` 加载已安装的外部实现。
4. 模型自己的配置覆盖通过 `workflow.policies.imitation.options.<key>=<value>` 传入。
5. 为坐标变换、轨迹选择、时间间隔、无未来泄漏、真实权重和闭环步进补测试。

最小接入骨架（需自行实现模型推理，不是可运行的示例算法）：

```python
from head.agents import BaseAgent, Trajectory

class Agent(BaseAgent):
    def initialize(self, checkpoint):
        self.model = load_my_model(checkpoint, self.device)

    def compute_trajectory(self, agent_input):
        features = build_my_features(agent_input.scenario, agent_input.current_step)
        output = self.model(features)
        world_xy = select_and_transform(output)
        return Trajectory(world_xy, dt=0.1, reference_offset=0.0)
```

按目录约定动态发现，不需要编辑 loader、配置白名单或打包文件。
测试 `tests/test_agents.py::test_new_algorithm_needs_no_core_changes` 验证这个约定。

## 迁移与运行口径

- 旧的 `vendor/unitraj_benchmark` 已拆入以上算法目录和 common；运行不需要外部 UniTraj checkout。
- 旧 `pluto_closed_loop_inference.py` 已并入统一入口，特有逻辑进入 Pluto Agent。
- `imitation.source` 仅保留作旧权重路径搜索，不再导入那里的代码；外部算法用模块路径接入。
- 默认 21 帧预热、每 5 帧重规划、最多 120 帧；场景提前结束会缩短实际可评帧数。
- Pluto 默认神经网络选轨迹，规则权重 0；WayFormer 默认最高概率模式，非固定第 0 个模式。
- `evaluation/` 仍是唯一闭环评价来源：score、collision、offroad、completion、TTC strict、comfort strict。
  不修改指标公式，不把模型轨迹选择的规则权重混同为评价权重。
- 此次为发布代码重构，不重写既有实验结果，也未上传私有研究代码、模型权重和数据缓存。

## 来源和许可证

模型和预处理来自项目此前集成的 Pluto/UniTraj 推理源码，WayFormer 恢复自已用于
120 帧实验的源码快照。模型目录重组和 import 路径变化不改变网络参数名称。
WayFormer 推理包装直接继承 PyTorch Module，移除了对 Lightning 训练评价基类的运行依赖；
网络层、参数和前向计算保留，完整 UniTraj 训练工作流不在本次发布范围内。

保留集成源码原有 [许可证声明](common/LICENSE) 和 [AGPLv3 全文](common/LICENSE.AGPL)，
目录改名不更改上游代码许可证。项目根目录许可证不覆盖这些第三方代码的原有声明。
权重条款与来源另见各模型 README；此处不为第三方权重重新指定许可证。
