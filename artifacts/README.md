# 模型权重与运行输出

保存模型权重、训练日志和闭环评测结果。具体路径由 `head/configs/default.yaml` 的 `artifacts` 配置控制。

| 路径 | 用途 |
| --- | --- |
| `weights/imitation/` | Pluto、WayFormer 等模仿学习模型的下载权重 |
| `weights/evolution/` | 按策略、学习器和任务保存的训练权重 |
| `eval/closed_loop/` | 闭环评测产生的 `metrics.json` 等结果 |
| `logs/` | 训练和运行日志 |
| `models/` | 仓库原有的历史 RLBoost/SAC 模型文件 |

下载或运行后才会生成部分目录。新增权重与实验结果应保留在本地，不随源码提交；已有历史模型不在本次文档整理中变更。

模型下载与运行方式见[项目说明](../README.md)。
