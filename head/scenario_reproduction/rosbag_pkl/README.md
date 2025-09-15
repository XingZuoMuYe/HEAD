
# Scenario_reproduction / rosbag_pkl / data_convert.py
data_convert.py为将rosbag文件转换为pkl文件的主要脚本。它读取rosbag文件中的车辆和行人信息，并将其存储为pkl文件，以便在后续的自学习循环中使用。
![](./assets/HEAD.jpg)
## 🔧Quick Start
1. **运行data_convert.py，其内部会依次调用"osm_scenario.py", "obj_info.py", "dataset_summary.py"文件**


2. **osm_scenario.py从osm文件获取地图信息**


3. **obj_info.py从rosbag文件获取车辆和行人信息**


4. **dataset_summary.py将上述信息转换为pkl文件**


5. **metadrive渲染命令**
```bash
python -m scenarionet.sim -d /path_to_your scenario_reproduction/datasets --render 2D/3D
```
