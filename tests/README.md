# 测试

覆盖配置解析、策略选择、模型适配、环境步进和闭环评价。
`test_model_checkpoints.py` 用于显式提供本地权重后的模型推理测试。

在已配置的 Conda 环境中，从仓库根目录运行：

```bash
python -m pytest -q tests
```
