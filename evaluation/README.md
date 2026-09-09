# 闭环评价指标

定义 HEAD 闭环评测的 TTC、舒适性及综合评分，并提供原生接口和结果后处理工具。
HEAD 与 MetaDrive 的状态接入位于 [`head/manager/evaluation_v2.py`](../head/manager/evaluation_v2.py)。

| 文件 | 用途 |
| --- | --- |
| [`closed_loop_metrics.py`](closed_loop_metrics.py) | 基于运动状态计算 TTC 与舒适性指标 |
| [`compute_puffer_nuplan_style_scores.py`](compute_puffer_nuplan_style_scores.py) | nuPlan-style 综合分的后处理工具 |
| [`native_evaluation.py`](native_evaluation.py) | 原生评测接口工具，独立于 HEAD 的 MetaDrive 接入 |
| [`test_puffer_nuplan_style_scores.py`](test_puffer_nuplan_style_scores.py) | 综合分后处理测试 |

以下保留指标定义、适用范围和有效性规则。

## HEAD closed-loop evaluation

This package is the single source of HEAD's closed-loop metrics. The former
UniTraj `EvaluateMetrics` recorder has been removed; `evaluation.updated_metrics`
defaults to `true` and there is no second metric path to fall back to.
The pure-NumPy `closed_loop_metrics.py` is the supplied
`closed_loop_metrics_v2` implementation. HEAD adds only optional `focal_mask`
aggregation: background obstacles still contribute to ego TTC, but background
vehicles' scores do not dilute the controlled planner's scores. Original tests
are in `tests/test_evaluation_v2.py` with their import adapted to this package.

The PufferDrive C simulator/binding files and its checkpoint evaluation driver
are not compiled or substituted into MetaDrive. They describe another simulator.
The pre-existing native/postprocessing Python utilities are retained separately.

Enabled by default (`evaluation.updated_metrics=true`). Results are written
under each episode's `evaluation_v2` in
`artifacts/eval/closed_loop/<policy>/<task>/<map>/metrics.json`, with an
independent summary. The `source`, `legacy_source` and `error` keys remain in
that file, always `null`, so existing readers need no special case.
This is **not official nuPlan benchmark scoring** and not PufferDrive native score.

Mapping:

| Quantity | HEAD source / interpretation |
| --- | --- |
| position, heading, velocity | executed MetaDrive state; metres, radians, m/s |
| dimensions | actual spawned object LENGTH/WIDTH; oriented-box TTC |
| focal agents | controlled ego only |
| obstacles | spawned vehicles, traffic participants, traffic objects |
| warmup | excluded by policy's `closed_loop_stage` marker |
| lifecycle | invalid on absence; stable object IDs; no respawn/stop action in this adapter |
| collision / off-road / completion | post-warmup MetaDrive native flags |
| progress | remains separately labelled legacy route progress; not binary completion |

TTC: constant velocity and heading, horizon 3 s, projection quantization 0.1 s,
safe threshold 0.95 s. Directed relevance excludes same-direction rear obstacles.
No road boundaries are TTC obstacles. Comfort uses native velocities and heading,
Savitzky-Golay derivatives, six threshold checks, with invalid/boundary frames
excluded. Short episodes without valid comfort samples are unavailable, not passed.

The optional HEAD nuPlan-style scores use the user postprocessing weights
`(5*completion + 5*TTC + 2*comfort)/12`, gated by no collision and no off-road.
Strict and frame-rate variants are separate. Completion is a MetaDrive arrival
event, not route progress or PufferDrive goal-counter completion. A truncated
episode can therefore have high progress and zero completion.
