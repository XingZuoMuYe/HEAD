# 配置、策略与评价测试

保存 HEAD 的自动化回归测试及环境调试示例。

| 测试范围 | 相关文件 |
| --- | --- |
| 配置解析与任务校验 | `test_configuration.py` |
| 策略选择与检查点解析 | `test_strategy_selection.py`、`test_poly_checkpoint_resolution.py` |
| 权重和输出路径 | `test_artifact_paths.py` |
| 仿真环境初始化与步进 | `test_environment.py` |
| 模仿学习闭环接入 | `test_imitation_closed_loop.py` |
| 闭环结果与评价指标 | `test_closed_loop_metrics.py`、`test_evaluation_v2.py` |

在项目依赖已经安装的 Conda 环境中，从仓库根目录执行：

```bash
python -m pytest -q tests
```

环境相关测试需要对应的 MetaDrive 依赖与资源；`drive_in_real_env.py`、`env_render_plot.py` 和 `run_env.py` 是手动调试示例，不等同于自动化测试结果。

完整环境要求见[项目说明](../README.md)。
