<img src="./assets/HEAD-icon.jpg" alt="HEAD icon" style="display:block; margin: 0 auto; width: 400px;">

# HEAD:Holistic Evolutionary Autonomous Driving
HEAD is a holistic suite of evolutionary autonomous driving software, based on the MetaDrive simulation platform, that seamlessly imports driving scenarios, uploads training models, and efficiently performs continuous training designed to significantly improve the performance of arbitrary models.
## Introduction
**HEAD (Holistic Evolutionary Autonomous Driving)** is an Autonomous Driving Platform with the following key features: 
- **A General Self-Evolutionary Autonomous Driving Software Tool**: It combines learning-based, optimization-based, and rule-based algorithms to efficiently handle complex driving scenarios and ensure safety and performance.
- **Integration with Simulation Testing**: It is deeply integrated with the MetaDrive simulation platform, enabling comprehensive testing and optimization.
- **A Closed-Loop Data-Driven Platform**: It provides a complete closed-loop system from scenario generation to algorithm evolution, enhancing adaptability and reliability in unseen scenarios through adversarial testing and continuous learning.
![](./assets/HEAD.jpg)
## 🔧Quick Start
1. **Clone the repo**

   Start by cloning the HEAD repository to your local machine:
    ``` bash
    git clone https://github.com/XingZuoMuYe/HEAD.git
    cd HEAD
   ```
2. **Conda Env Settings and Install Dependencies**
    ``` bash
   pip install uv 
   uv venv --python 3.9 HEAD
   source HEAD/bin/activate
   uv pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   cd head
   pip install -e .
    ```
3. **After setting up the base environment, install PyTorch and compatible deep
learning libraries. Make sure your CUDA version matches (e.g., CUDA 12.1).**
    ``` bash
    uv pip install torch==2.5.0+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
    
    # Install NATTEN (Neighborhood Attention)
    uv pip install natten==0.21.1+torch280cu126 -f https://whl.natten.org
    
    # Install PyTorch Geometric dependencies
    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
    
    # Install Waymo Open Dataset (TensorFlow version)
    uv pip install waymo-open-dataset-tf-2-12-0 --extra-index-url https://pypi.org/simple
    ```
4. **Install scenarionet**
   ```bash
   git clone https://github.com/metadriverse/scenarionet.git
   cd scenarionet
   uv pip install -e .
   ```
5. **Extract scenario datasets**

   ```bash
   unzip head/scenario_datasets/geely.zip -d head/scenario_datasets/geely/
   unzip head/scenario_datasets/waymo.zip -d head/scenario_datasets/waymo/

6. **Build the `local_utils_cpp` module (for C++ accelerate)**

    ```bash
    # Go to the local planner module
    cd ~/HEAD/head/policy/evolvable_policy/common/local_planner
    # Install dependencies (pybind11, cmake, etc.)
    sudo apt update
    sudo apt install pybind11-dev
    # Create a new build directory
    mkdir -p build && cd build
    # Configure with CMake (Release mode recommended)
    cmake -DCMAKE_BUILD_TYPE=Release ..
    # Compile using all available cores
    cmake --build . -j$(nproc)

7. **Install Unitraj and Create a symbolic link to the policy module in the `/head/policy/imitation_policy/` directory:**

    ```bash
    ln -s /path/to//UniTraj .

   uv pip install torch==2.5.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   uv pip install natten==0.21.1+torch280cu126 -f https://whl.natten.org
   uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
   uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
   uv pip install waymo-open-dataset-tf-2-12-0 --extra-index-url https://pypi.org/simple

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

## 📁 Project Structure

```text
tree -L 6 -I '__pycache__|*.pyc|*.egg-info|venv' > structure.txt
```

```text
├.
├── artifacts
│   ├── eval
│   │   └── RLBoost_SAC
│   │       ├── muti_scenario
│   │       │   └── XCO
│   │       ├── real_scenario
│   │       │   └── real
│   │       ├── single_scenario
│   │       │   ├── interaction
│   │       │   └── roundabout
│   │       └── straight_config_traffic
│   │           └── straight_road
│   ├── logs
│   │   └── RLBoost_SAC
│   │       ├── muti_scenario
│   │       │   └── XCO
│   │       │       └── wandb_info
│   │       ├── real_scenario
│   │       │   └── real
│   │       │       └── wandb_info
│   │       ├── single_scenario
│   │       │   ├── interaction
│   │       │   │   └── wandb_info
│   │       │   └── roundabout
│   │       │       └── wandb_info
│   │       └── straight_config_traffic
│   │           └── straight_road
│   └── models
│       └── RLBoost_SAC
│           └── checkpoints
│               ├── muti_scenario
│               │   └── XCO
│               ├── real_scenario
│               │   ├── geely
│               │   ├── real
│               │   └── waymo
│               ├── single_scenario
│               │   ├── circle_road
│               │   ├── inRamp
│               │   ├── interaction
│               │   └── roundabout
│               └── straight_config_traffic
│                   ├── straight_road
│                   └── straight_road_no_pedestrian
├── assets
│   ├── closed_loop_structure.jpg
│   ├── experiment_2.jpg
│   ├── experiment.jpg
│   ├── HEAD-icon.jpg
│   ├── HEAD.jpg
│   └── HEAD-structure.png
├── debug
│   └── head_debug.py
├── geely
├── head
│   ├── component
│   │   ├── map
│   │   │   ├── custom_light_manager.py
│   │   │   ├── custom_map_manager.py
│   │   │   └── lane_utils.py
│   │   └── navigation
│   │       └── custom_navigation.py
│   ├── configs
│   │   ├── default.yaml
│   │   └── tasks
│   │       ├── default.yaml
│   │       ├── muti_scenario.yaml
│   │       ├── real_scenario.yaml
│   │       ├── single_scenario.yaml
│   │       └── straight_config_traffic.yaml
│   ├── envs
│   │   ├── config_traffic_metadrive_env.py
│   │   ├── __init__.py
│   │   ├── multi_scenario_metadrive_env.py
│   │   └── real_scenario_metadrive_env.py
│   ├── evolution_engine
│   │   ├── common
│   │   │   ├── __init__.py
│   │   │   ├── memory.py
│   │   │   ├── model.py
│   │   │   ├── multiprocessing_env.py
│   │   │   ├── plot.py
│   │   │   ├── running_mean_std.py
│   │   │   └── utils.py
│   │   ├── env_builder
│   │   │   ├── env.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   └── RLBoost
│   │       ├── __init__.py
│   │       └── SAC
│   │           ├── agent.py
│   │           ├── cfg.py
│   │           ├── __init__.py
│   │           ├── logger.py
│   │           ├── model.py
│   │           └── SAC_learner.py
│   ├── __init__.py
│   ├── manager
│   │   ├── base_algorithm_selector.py
│   │   ├── bev_img_manager
│   │   │   └── bev_img_manager.py
│   │   ├── config_manager.py
│   │   ├── config_pedestrain_manager.py
│   │   ├── config_traffic_manager.py
│   │   ├── evolution_engine.py
│   │   ├── evolution_selector.py
│   │   └── __init__.py
│   ├── policy
│   │   ├── basic_policy
│   │   │   ├── idm_policy_include_pedestrian.py
│   │   │   ├── idm_policy_with_osm.py
│   │   │   └── __init__.py
│   │   ├── evolvable_policy
│   │   │   ├── common
│   │   │   │   ├── cfgs
│   │   │   │   │   └── config.yaml
│   │   │   │   ├── config.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── local_planner
│   │   │   │   │   ├── 编译命令.txt
│   │   │   │   │   ├── CMakeLists.txt
│   │   │   │   │   ├── cubic_spline_planner.py
│   │   │   │   │   ├── frenet_optimal_trajectory.py
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── setup.py
│   │   │   │   │   ├── spline_utils.pyx
│   │   │   │   │   └── util.cpp
│   │   │   │   ├── low_level_controller
│   │   │   │   │   ├── controller.py
│   │   │   │   │   └── __init__.py
│   │   │   │   ├── tools
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── misc.py
│   │   │   │   │   └── utils.py
│   │   │   │   └── utils.py
│   │   │   ├── __init__.py
│   │   │   └── poly_planning_policy.py
│   │   └── __init__.py
│   ├── pyproject.toml
│   ├── renderer
│   │   ├── head_renderer.py
│   │   └── top_down_renderer.py
│   ├── scenario_datasets
│   │   ├── geely.zip
│   │   └── waymo.zip
│   ├── scenario_reproduction
│   │   ├── __init__.py
│   │   └── rosbag_pkl
│   │       ├── data_convert.py
│   │       ├── __init__.py
│   │       ├── README.md
│   │       └── util
│   │           ├── dataset_summary.py
│   │           ├── GNSS_info_process.py
│   │           ├── GNSS_Transform.py
│   │           ├── __init__.py
│   │           ├── obj_info.py
│   │           ├── osm_scenario.py
│   │           └── raw_data
│   │               ├── scenario_1
│   │               ├── scenario_2
│   │               ├── scenario_3
│   │               ├── scenario_4
│   │               └── scenario_5
│   └── scripts
│       ├── __init__.py
│       └── main_head.py
├── LICENSE
├── README.md
├── requirements.txt
├── start_train.sh
├── structure.txt
├── tests
│   ├── drive_in_real_env.py
│   ├── env_render_plot.py
│   ├── map.jpg
│   └── run_env.py
└── waymo

83 directories, 96 files

```



