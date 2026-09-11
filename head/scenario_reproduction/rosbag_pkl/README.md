# 日志转场景

将实车日志中的车辆、行人信息与 OSM 地图整理为 pkl 场景数据。

| 文件 | 用途 |
| --- | --- |
| [data_convert.py](data_convert.py) | 数据转换入口 |
| [util/osm_scenario.py](util/osm_scenario.py) | 地图处理 |
| [util/obj_info.py](util/obj_info.py) | 车辆与行人信息处理 |
| [util/dataset_summary.py](util/dataset_summary.py) | 场景数据整理与输出 |
