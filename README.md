<img src="./assets/HEAD-icon.jpg" alt="HEAD icon" style="display:block; margin: 0 auto; width: 400px;">

# HEAD:Holistic Evolutionary Autonomous Driving
HEAD is a holistic suite of evolutionary autonomous driving software, based on the MetaDrive simulation platform, that seamlessly imports driving scenarios, uploads training models, and efficiently performs continuous training designed to significantly improve the performance of arbitrary models.
## Introduction
**HEAD (Holistic Evolutionary Autonomous Driving)** is an Autonomous Driving Platform with the following key features: 
- **A General Self-Evolutionary Autonomous Driving Software Tool**: It combines learning-based, optimization-based, and rule-based algorithms to efficiently handle complex driving scenarios and ensure safety and performance.
- **Integration with Simulation Testing**: It is deeply integrated with the MetaDrive simulation platform, enabling comprehensive testing and optimization.
- **A Closed-Loop Data-Driven Platform**: It provides a complete closed-loop system from scenario generation to algorithm evolution, enhancing adaptability and reliability in unseen scenarios through adversarial testing and continuous learning.
![](./assets/HEAD.jpg)
## 🔧 Quick Start

The commands below are the tested, minimal setup for the MetaDrive environments and
the RLBoost examples. Linux, Python 3.9--3.11, a C++ compiler and (for rendering)
an X11-capable display are recommended. A GPU is optional for basic usage.

1. **Clone the repository**

   ```bash
   git clone https://github.com/XingZuoMuYe/HEAD.git
   cd HEAD
   ```

2. **Create and activate a virtual environment**

   `uv venv` must target a directory that does not replace the repository itself.

   ```bash
   python3 -m pip install --upgrade uv
   uv venv --python 3.9 .venv
   source .venv/bin/activate
   export LD_LIBRARY_PATH=
   export CUDA_HOME=
   uv pip install -r requirements.txt
   ```

   `requirements.txt` installs the core MetaDrive and RLBoost dependencies. Optional
   Waymo and UniTraj dependencies are not included.

3. **PyTorch/CUDA compatibility**

   `requirements.txt` pins the PyTorch 2.5.0 CUDA 12.1 build used by this
   project. Do not mix it with a NATTEN or PyTorch wheel built for another CUDA
   version. CPU-only environments should replace the three PyTorch packages with
   matching CPU wheels before running the project.

4. **Optional: extract the bundled scenario archives**

   These archives are only required for `real_scenario-v0`. Run from the repository
   root; `unzip` creates the destination directories and is safe to rerun.

   ```bash
   unzip -o head/scenario_datasets/geely.zip -d head/scenario_datasets/
   unzip -o head/scenario_datasets/waymo.zip -d head/scenario_datasets/
   ```

5. **Optional: build the local planner C++ extension**

   The Python planner works without this extension. On Debian/Ubuntu, install Eigen3,
   pybind11 and CMake, then build from the repository root:

   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential cmake libeigen3-dev pybind11-dev
   cmake -S head/policy/evolvable_policy/common/local_planner \
         -B head/policy/evolvable_policy/common/local_planner/build \
         -DCMAKE_BUILD_TYPE=Release
   cmake --build head/policy/evolvable_policy/common/local_planner/build \
         --parallel
   ```

6. **Configure and run the project**

   Keep `LD_LIBRARY_PATH=` and `CUDA_HOME=` in the environment when launching the
   project, especially on machines with ROS or a system CUDA installation. The
   default configuration selects Poly on the generated straight-road scenario.
   It does not require UniTraj; Poly uses its automatic checkpoint directory and
   falls back to random actions when no trained run exists.

   ```bash
   export LD_LIBRARY_PATH=
   export CUDA_HOME=
   python -m head.scripts.main_head
   ```

   Configuration is split by responsibility:

   - `runtime`: random seed and device selection
   - `simulation`: vectorization, rendering, and environment sizing
   - `workflow.type`: lifecycle selection (`deploy` or `evolution`)
   - `workflow.policy`: one of four peer policies (`IDM`, `Poly`, `Zero`, or `imitation`)
   - `workflow.evolution`: evolution strategy and learner
   - `workflow.policies`: per-policy checkpoints and imitation settings
   - `evaluation`: closed-loop mode, episode count, interval, and video output
   - `logging`: external experiment tracking
   - `artifacts`: separate roots for evolution weights, imitation weights, logs, and evaluation outputs
   - `task`: choose `straight_config_traffic-v0`, `single_scenario-v0`,
     `multi_scenario-v0`, or `real_scenario-v0`
   - `scenario`: task-specific map and dataset values loaded from `configs/tasks`
   - `scenario.capabilities`: task capabilities used for compatibility validation

   Values can be overridden without editing YAML, for example:

   ```bash
   python -m head.scripts.main_head simulation.render=false runtime.device=auto
   ```

   Evaluation is intentionally fixed to closed-loop interaction:

   ```yaml
   evaluation:
     mode: closed_loop
     episodes: 1
     max_steps: 400
     save_video: false
   ```

   `evaluation.mode` must remain `closed_loop`; changing it is rejected during
   configuration validation.

   IDM is one of the four peer policies in the deploy workflow:

   ```yaml
   task: straight_config_traffic-v0
   workflow:
     type: deploy
     policy: IDM
   ```

   Both workflow types accept the same policy set. `deploy` executes or
   evaluates the selected policy, while `evolution` attaches the configured
   evolution learner.

   `IDM` and `Zero` run directly as rule policies. Poly defaults to the
   automatically resolved directory
   `artifacts/weights/evolution/RLBoost/SAC/<task>/<map>/<train_name>/`.
   If that directory contains `sac_policy`, it is loaded; otherwise Poly warns
   and uses `action_space.sample()`. Evolution checkpoints are stored at
   `artifacts/weights/evolution/<strategy>/<learner>/<task>/<map>/<train_name>/`.

   For convenience, `workflow.type=IDM`, `workflow.type=Poly`,
   `workflow.type=Zero`, and `workflow.type=imitation` are accepted as
   shorthand deploy commands and are normalized to the schema above.

   All policy-specific values belong under `workflow.policies`; only the
   selected policy is validated. Relative paths are resolved from the project
   root:

   ```yaml
   workflow:
     type: deploy
     policy: Poly
     policies:
       Poly:
         checkpoint: auto
       imitation:
         model: pluto            # pluto | wayformer
         source: vendor/unitraj_benchmark
         checkpoint: artifacts/weights/imitation/pluto/pluto_1M_aux_cil.ckpt
   ```

   | Workflow | Allowed policies |
   | --- | --- |
   | `deploy` | `IDM`, `imitation`, `Poly`, `Zero` |
   | `evolution` | `IDM`, `imitation`, `Poly`, `Zero` |

   Before selecting imitation learning, install the benchmark-specific packages:

   ```bash
   uv pip install lightning pytorch-lightning hydra-core easydict einops h5py torch-geometric
   uv pip install scenarionet
   ```

   The UniTraj inference code for both models ships with this repository at
   `vendor/unitraj_benchmark`, so `workflow.policies.imitation.source` needs no
   external checkout. Model weights are **not** in git — download them first:

   ```bash
   # Wayformer
   mkdir -p artifacts/weights/imitation/wayformer
   curl -L -o "artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt" \
     https://huggingface.co/GALLERVICH/WayFormer-head/resolve/main/brier_fde%3D1.45.ckpt

   # verify the download (182,642,508 bytes)
   sha256sum "artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt"
   # f0dc49d961f6bb239855da285dd9e56ba05a13a14aa7acd37809a64a074f657b

   # Pluto (official release)
   mkdir -p artifacts/weights/imitation/pluto
   # place pluto_1M_aux_cil.ckpt in artifacts/weights/imitation/pluto/
   ```

   Model page: <https://huggingface.co/GALLERVICH/WayFormer-head>

   **Benchmark scenarios.** The frozen 171-scenario closed-loop regression set
   (101 in `sg-one-north`, 70 in `us-ma-boston`, converted from the official
   nuPlan v1.1 `val` split) is published separately:

   ```bash
   curl -L -o test_scenes.zip \
     https://huggingface.co/datasets/GALLERVICH/mini_test_nuplan/resolve/main/test_scenes.zip
   unzip test_scenes.zip           # -> scenes/, SHA256SUMS, fixed171_120_v2.json
   sha256sum -c SHA256SUMS         # verify all 171 scenarios
   ```

   Dataset page: <https://huggingface.co/datasets/GALLERVICH/mini_test_nuplan>

   `fixed171_120_v2.json` records each scenario's token, source log, city,
   scenario type and sha256, so the set is reproducible and verifiable. Point
   `scenario.dataset.directory` at the extracted `scenes/` directory to run
   against it. The archive is 433 MB (700 MB extracted).

   HEAD then runs ego-only closed-loop control: it warms up from recorded
   history, replans periodically, and converts the predicted trajectory to
   MetaDrive steering and throttle actions with PID control.

   ```bash
   export LD_LIBRARY_PATH=
   export CUDA_HOME=
   uv run python -m head.scripts.main_head \
     task=real_scenario-v0 \
     workflow.type=deploy \
     workflow.policy=imitation \
     runtime.device=auto \
     simulation.render=false
   ```

   That runs the configured default. To switch models, override two values:

   ```bash
   workflow.policies.imitation.model=wayformer \
   workflow.policies.imitation.checkpoint="artifacts/weights/imitation/wayformer/brier_fde=1.45.ckpt"
   ```

   Pluto executes the network's top-1 trajectory by default
   (`trajectory_selection_mode: neural_only` in
   `vendor/unitraj_benchmark/unitraj/configs/method/Pluto.yaml`); the rule-based
   evaluator is not constructed at all. Set it to `hybrid` to re-score
   candidates with the rule-based evaluator instead.

   `imitation` is only accepted when the selected task configuration
   declares `scenario.capabilities.closed_loop_imitation: true`. Generated-road
   tasks declare this capability as false and are rejected before environment
   creation.

   Every closed-loop evaluation writes results to
   `artifacts/eval/closed_loop/<policy>/<task>/<map>/metrics.json`. Each episode
   records reward, length, collision, out-of-road, destination arrival, and
   success. The aggregate also contains `collision_rate`, `out_of_road_rate`,
   `arrive_dest_rate`, and `success_rate`.

   Closed-loop metrics come from the `evaluation` package, which is the single
   metric source; the former UniTraj `EvaluateMetrics` recorder has been
   removed. Each episode gets an `evaluation_v2` block scoring the ego only,
   with two composite scores — `head_nuplan_style_strict_score` and
   `head_nuplan_style_frame_score` — over completion, collision, off-road, TTC
   and comfort. Episodes too short for a valid TTC or comfort sample report
   `null` rather than zero.

   Metric definitions, weights and validity rules are in
   [`evaluation/README.md`](evaluation/README.md). These numbers are **not**
   official nuPlan benchmark scoring.

   During evaluation, the key values are printed as they are collected:

   ```text
   [HEAD evaluation] 7a2fe26f15115574 validity: {'events': True, 'ttc': True, 'comfort': True} strict: 0.83 frame: 0.79
   [闭环] Episode:1 Reward:43.037 Length:80 Collision:False OutOfRoad:False ArriveDest:True Success:True
   [闭环汇总] Episodes:1 MeanReward:43.037 SuccessRate:1.000 CollisionRate:0.000 OutOfRoadRate:0.000 ArriveDestRate:1.000
   ```

   HEAD evaluation is always environment-stepped closed-loop evaluation; there
   is no offline/open-loop evaluation entry point in `main_head.py`. The complete
   policy matrix is valid. Only the selected policy is validated:

   | Workflow | IDM | Zero | Poly | imitation |
   | --- | --- | --- | --- | --- |
   | `deploy` | rule policy | rule policy | checkpoint or random action | Pluto / Wayformer checkpoint |
   | `evolution` | SAC | SAC | SAC | SAC + UniTraj |

   For `deploy + Poly`, an empty `workflow.policies.Poly.checkpoint` prints a
   warning and uses `action_space.sample()`. For `evolution + IDM/Zero/Poly`, an
   empty policy checkpoint means random initialization. Both `pluto` and
   `wayformer` are supported for `workflow.policies.imitation.model`.

7. **Run the automated tests**

   ```bash
   export LD_LIBRARY_PATH=
   export CUDA_HOME=
   pytest -q
   ```

   The test suite validates configuration parsing and performs a reset/step of the
   default MetaDrive environment.

## References

If you use HEAD in your own work, please cite:
```text
@article{yang2024guarantee,
  title={How to guarantee driving safety for autonomous vehicles in a real-world environment: a perspective on self-evolution mechanisms},
  author={Yang, Shuo and Huang, Yanjun and Li, Li and Feng, Shuo and Na, Xiaoxiang and Chen, Hong and Khajepour, Amir},
  journal={IEEE Intelligent Transportation Systems Magazine},
  year={2024},
  publisher={IEEE}
}
```

## Acknowledgements

This project integrates and builds upon the following excellent open-source works:

- MetaDrive Simulation:
  - GitHub: https://github.com/metadriverse/metadrive
  - Website: https://metadriverse.github.io/metadrive/

- ScenarioNet:
  - GitHub: https://github.com/metadriverse/scenarionet

- UniTraj (Unified Trajectory Forecasting Framework):
  - GitHub: https://github.com/vita-epfl/UniTraj
  - Paper: https://arxiv.org/abs/2403.15098

We gratefully acknowledge their contributions to the autonomous driving and imitation learning communities.

``` text
@article{li2021metadrive,
  title={MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning},
  author={Li, Quanyi and Peng, Zhenghao and Xue, Zhenghai and Zhang, Qihang and Zhou, Bolei},
  journal={arXiv preprint arXiv:2109.12674},
  year={2021}
}
```

## Relevant Projects

**Metadrive: Composing diverse driving scenarios for generalizable reinforcement learning**
\
Li, Quanyi and Peng, Zhenghao and Feng, Lan and Zhang, Qihang and Xue, Zhenghai and Zhou, Bolei
\
*IEEE Transactions on Pattern Analysis and Machine Intelligence*
\
[
<a href="https://arxiv.org/pdf/2109.12674.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/metadrive-simulator/">Website</a>
|
<a href="https://github.com/metadriverse/metadrive">Code</a>
]

## License

All assets and code are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

## Project Structure

The repository is organized around configuration, environment construction,
policy selection, and closed-loop evaluation. Generated checkpoints, logs, and
metrics are kept under `artifacts/` and are not source files.

```text
HEAD/
├── head/
│   ├── configs/                 # default.yaml and per-task YAML files
│   ├── envs/                    # generated and recorded MetaDrive environments
│   ├── evolution_engine/        # environment builder and RLBoost/SAC
│   ├── manager/
│   │   ├── config_manager.py    # config merge and validation
│   │   ├── evolution_selector.py
│   │   ├── artifact_paths.py    # checkpoint and output resolution
│   │   ├── closed_loop_metrics.py  # MetaDrive events + evaluation delegation
│   │   └── evaluation_v2.py     # adapter into the evaluation package
│   ├── policy/
│   │   ├── basic_policy/        # IDM and Zero
│   │   ├── evolvable_policy/    # Poly and local planner
│   │   └── imitation_policy/    # Pluto / Wayformer inference and controller
│   ├── renderer/
│   └── scripts/main_head.py     # closed-loop entry point
├── evaluation/                 # closed-loop metric definitions (single source)
├── vendor/unitraj_benchmark/   # bundled UniTraj inference code for both models
├── tests/                      # configuration, environment, policy, metrics tests
├── artifacts/                  # generated eval/closed_loop and logs; weights are downloaded
├── assets/                     # figures and project images
├── requirements.txt
├── README.md
└── LICENSE
```

