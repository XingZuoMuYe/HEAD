# Pluto

## 输入口径限制

旧版特征构建器通过整段日志的自车轨迹推断路线，其中包含未来位置。
本次接口迁移保留旧行为，标记为 `input_scope=legacy_logged_route`，
**不是严格只使用历史观测的评测**。消除这一依赖会改变路线和推理结果，
需另立协议、重新评测，不应把修正前后的分数直接混用。

## 模型与运行

模型、特征构建和规划适配集中在本目录：
`model/` 为网络，`features/` 为输入预处理，`planner.py` 为候选生成，
`agent.py` 负责权重加载、后轴参考点和输出列转换。

将已有官方权重 `pluto_1M_aux_cil.ckpt` 放在
`artifacts/weights/imitation/pluto/`。权重不进入 Git。

从仓库根目录、在 Conda 环境中运行：

```bash
python -m head.scripts.main_head \
  task=real_scenario-v0 workflow.policy=imitation \
  workflow.policies.imitation.model=pluto \
  workflow.policies.imitation.checkpoint=artifacts/weights/imitation/pluto/pluto_1M_aux_cil.ckpt \
  simulation.render=false evaluation.max_steps=120
```

默认 `trajectory_selection_mode=neural_only`、
`rule_based_score_weight=0`、`learning_based_score_weight=1`；
不构建或调用候选轨迹规则评价器。闭环结果仍由根目录的 evaluation 计算。
原始控制轨迹列为 x,y,cos(yaw),sin(yaw),vx,vy，Agent 转为 x,y,vx,vy。

吉利日志中的 [T,1] 标量状态和 nuPlan/Waymo 的 [T] 形式在适配器中统一，
不改变仿真器拥有的原始数据。SAE 研究钩子仅在额外安装研究代码且显式启用时使用，
不属于公开发布版的默认推理依赖。

来源、许可证与统一接口见[算法接入说明](../README.md)。
