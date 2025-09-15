import numpy as np
import pickle as pkl
from collections import defaultdict
from metadrive.scenario.scenario_description import ScenarioDescription
from pathlib import Path

def get_project_root() -> Path:
    """获取项目的根路径（假设项目根目录是当前文件的上两级目录）。"""
    curr_path = Path(__file__).resolve()  # 当前文件的绝对路径
    return curr_path.parent  # 项目根路径


def get_number_summary(scenario, summary_dict):
    """Return the stats of all objects in a scenario.

    Examples:
        {'num_objects': 211,
         'object_types': {'CYCLIST', 'PEDESTRIAN', 'VEHICLE'},
         'num_objects_each_type': {'VEHICLE': 184, 'PEDESTRIAN': 25, 'CYCLIST': 2},
         'num_moving_objects': 69,
         'num_moving_objects_each_type': defaultdict(int, {'VEHICLE': 52, 'PEDESTRIAN': 15, 'CYCLIST': 2}),
         'num_traffic_lights': 8,
         'num_traffic_light_types': {'LANE_STATE_STOP', 'LANE_STATE_UNKNOWN'},
         'num_traffic_light_each_step': {'LANE_STATE_UNKNOWN': 164, 'LANE_STATE_STOP': 564},
         'num_map_features': 358,
         'map_height_diff': 2.4652252197265625}

    Args:
        scenario: The input scenario.

    Returns:
        A dict describing the number of different kinds of data.
    """
    number_summary_dict = {}

    # object
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS] = len(scenario[ScenarioDescription.TRACKS])
    number_summary_dict[ScenarioDescription.SUMMARY.OBJECT_TYPES] = \
        set(v["type"] for v in scenario[ScenarioDescription.TRACKS].values())
    object_types_counter = defaultdict(int)
    for v in scenario[ScenarioDescription.TRACKS].values():
        object_types_counter[v["type"]] += 1
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_OBJECTS_EACH_TYPE] = dict(object_types_counter)

    # # If object summary does not exist, fill them here
    # object_summaries = {}
    # for track_id, track in scenario[SD.TRACKS].items():
    #     object_summaries[track_id] = scenario.get_object_summary(object_dict=track, object_id=track_id)
    # scenario[scenario.METADATA][scenario.SUMMARY.OBJECT_SUMMARY] = object_summaries

    # moving object
    number_summary_dict.update(ScenarioDescription._calculate_num_moving_objects(scenario))

    # Number of different dynamic object states
    dynamic_object_states_types = set()
    dynamic_object_states_counter = defaultdict(int)
    for v in scenario[ScenarioDescription.DYNAMIC_MAP_STATES].values():
        for step_state in v["state"]["object_state"]:
            if step_state is None:
                continue
            dynamic_object_states_types.add(step_state)
            dynamic_object_states_counter[step_state] += 1
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS
    ] = len(scenario[ScenarioDescription.DYNAMIC_MAP_STATES])
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHT_TYPES] = dynamic_object_states_types
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_TRAFFIC_LIGHTS_EACH_STEP
    ] = dict(dynamic_object_states_counter)

    # map
    number_summary_dict[ScenarioDescription.SUMMARY.NUM_MAP_FEATURES
    ] = len(scenario[ScenarioDescription.MAP_FEATURES])
    number_summary_dict[ScenarioDescription.SUMMARY.MAP_HEIGHT_DIFF
    ] = ScenarioDescription.map_height_diff(scenario[ScenarioDescription.MAP_FEATURES])
    return number_summary_dict


if __name__ == '__main__':

    base_path = get_project_root()
    raw_data_path = base_path / 'raw_data' / 'scenario_1'
    dataset_path = base_path / 'datasets' / 'scenario_1'

    with open(dataset_path/ 'map_features.pkl', 'rb') as f:
        map_features = pkl.load(f)
    with open(dataset_path/'obj_info.pkl', 'rb') as f:
        obj_info = pkl.load(f)

    scenario_length = len(obj_info['vehicles'][0]['length'])
    #  scenario.pkl
    scenario = {"id": "1",
                "version": "2024-08-13"[300:],  # 这是什么意思
                "length": scenario_length,
                "tracks": {},
                "dynamic_map_states": {},
                "map_features": map_features,
                "metadata": {}}
    num_vehicle = 1
    for _, info in obj_info['vehicles'].items():
        vehicle_info = {str(num_vehicle): {"type": "VEHICLE",
                                           "state": {'position': info['position'],
                                                     "length": np.array(info['length']),
                                                     "width": np.array(info['width']),
                                                     "height": 1.6 * np.ones(scenario_length),
                                                     'heading': np.array(info['psi']),
                                                     'velocity': info['velocity'],
                                                     'valid': np.ones(scenario_length, dtype=bool)},
                                           "metadata": {}}}
        scenario['tracks'].update(vehicle_info)
        num_vehicle = num_vehicle + 1
    num_walker = num_vehicle
    for _, info in obj_info['walkers'].items():
        walker_info = {str(num_walker): {"type": "PEDESTRIAN",
                                         "state": {'position': info['position'],
                                                   "length": 0.6 * np.ones(scenario_length),
                                                   "width": 0.6 * np.ones(scenario_length),
                                                   "height": 1.8 * np.ones(scenario_length),
                                                   'heading': np.array(info['psi']),
                                                   'velocity': info['velocity'],
                                                   'valid': np.ones(scenario_length, dtype=bool)},
                                         "metadata": {}},
                       }
        scenario['tracks'].update(walker_info)
        num_walker = num_walker + 1

    # print(scenario['tracks']['18']['state']['position'])
    # summary.pkl
    dataset_summary = {}
    data_summary = {'id': "1",
                    'coordinate': "test",
                    "ts": np.arange(0, scenario_length/10, 0.1),
                    "metadrive_processed": False,
                    "sdc_id": "1",  # 主车为1，用于定位到主车
                    "datasets": "test",
                    "scenario_id": "1",
                    "source_file": "",
                    "track_length": scenario_length,
                    "current_time_index": 0,
                    # "sdc_track_index": 100,
                    "objects_of_interest": [],
                    "tracks_to_predict": {},
                    "object_summary": {},
                    "number_summary": {}}

    SD = ScenarioDescription
    # add agents summary
    summary_dict = {}
    for track_id, track in scenario[SD.TRACKS].items():
        summary_dict[track_id] = SD.get_object_summary(object_dict=track, object_id=track_id)
    scenario[SD.METADATA][SD.SUMMARY.OBJECT_SUMMARY] = summary_dict
    # count some objects occurrence
    scenario[SD.METADATA][SD.SUMMARY.NUMBER_SUMMARY] = get_number_summary(scenario, summary_dict)

    scenario["metadata"] = data_summary

    with open(dataset_path / "sd_scenario_test_1.pkl",
              "wb") as f:
        pkl.dump(scenario, f)

    dataset_summary["sd_scenario_test_1.pkl"] = data_summary
    with open(dataset_path/ "dataset_summary_1.pkl",
              "wb") as f:
        pkl.dump(dataset_summary, f)



