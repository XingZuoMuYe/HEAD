# Zero

部署时输出零转向、零油门/制动指令；进化模式下接收学习器的直接控制动作。

- [policy.py](policy.py)：零控制指令基线。
- [adapter.py](adapter.py)：根据运行模式选择控制策略。
