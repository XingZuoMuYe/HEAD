import csv
import re
import pickle as pkl
from pathlib import Path
import numpy as np

def get_project_root() -> Path:
    """获取项目的根路径（假设项目根目录是当前文件的上两级目录）。"""
    curr_path = Path(__file__).resolve()  # 当前文件的绝对路径
    return curr_path.parent  # 项目根路径


# 初始化字典来存储所有车辆和行人的数据
obj_data = {
    'vehicles': {},
    'walkers': {}
}

if __name__ == "__main__":
    map_name = 'Tongji_West'
    base_path = get_project_root()
    raw_data_path = base_path / 'raw_data'/ 'scenario_1'
    dataset_path = base_path / 'dataset' / 'scenario_1'
    with open(raw_data_path / '_Obj_info_batch.csv', 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)  # 读取标题行，获取所有列名

        # 遍历标题行，找到所有车辆和行人的数据列
        for header in headers:
            # 使用正则表达式匹配车辆的列
            match_vehicle = re.match(r'field\.vehicle_info_batch(\d+).*actor_pos\.x', header)
            if match_vehicle:
                vehicle_id = int(match_vehicle.group(1))  # 提取车辆ID
                if vehicle_id not in obj_data['vehicles']:
                    obj_data['vehicles'][vehicle_id] = {
                        'position': [],
                        'velocity': [],
                        'psi': [],
                        'width': [],
                        'length': []
                    }

            # 使用正则表达式匹配行人的列
            match_walker = re.match(r'field\.walker_info_batch(\d+).*actor_pos\.x', header)
            if match_walker:
                walker_id = int(match_walker.group(1))  # 提取行人ID
                if walker_id not in obj_data['walkers']:
                    obj_data['walkers'][walker_id] = {
                        'position': [],
                        'velocity': [],
                        'psi': [],
                        'width': [],
                        'length': []
                    }

        # 再次遍历文件，这次是数据行
        for row in reader:
            # 处理车辆数据
            for vehicle_id, vehicle_info in obj_data['vehicles'].items():
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.x')
                position_x = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.y')
                position_y = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.z')
                position_z = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.x')
                velocity_x = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.y')
                velocity_y = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.z')
                velocity_z = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_psi')
                psi = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_width')
                width = row[index] if index != -1 else None
                index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_length')
                length = row[index] if index != -1 else None

                # 将数据添加到对应的列表中
                vehicle_info['position'].append((float(position_x), -float(position_y), 0))
                vehicle_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
                vehicle_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))  # 为解决右舵驾驶问题地图关于x轴镜像，并进行角度调整
                # vehicle_info['psi'].append(float(psi))
                vehicle_info['width'].append(float(width))
                vehicle_info['length'].append(float(length))
                # vehicle_info['width'].append(float(2.6))
                # vehicle_info['length'].append(float(4.2))

            # 处理行人数据
            for walker_id, walker_info in obj_data['walkers'].items():
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.x')
                position_x = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.y')
                position_y = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.z')
                position_z = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.x')
                velocity_x = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.y')
                velocity_y = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.z')
                velocity_z = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_psi')
                psi = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_width')
                width = row[index] if index != -1 else None
                index = headers.index(f'field.walker_info_batch{walker_id}.actor_length')
                length = row[index] if index != -1 else None

                # 将数据添加到对应的列表中
                # walker_info['position'].append((abs(float(position_x)), abs(float(position_y)), float(position_z)))
                walker_info['position'].append((float(position_x), -float(position_y), 0))
                walker_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
                walker_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))
                walker_info['width'].append(float(width))
                walker_info['length'].append(float(length))

    # 筛除不存在的键值对
    obj_data['vehicles'] = {k: v for k, v in obj_data['vehicles'].items() if not all(w == 0 for w in v['width'])}
    obj_data['walkers'] = {k: v for k, v in obj_data['walkers'].items() if not all(w == 0 for w in v['width'])}
    for vehicle_id, vehicle_info in obj_data['vehicles'].items():
        vehicle_info['position'] = np.array(vehicle_info['position'])
    for walker_id, walker_info in obj_data['walkers'].items():
        walker_info['position'] = np.array(walker_info['position'])

    # 打印结果
    # for vehicle_id, vehicle_info in obj_data['vehicles'].items():
    #     print(f"Vehicle {vehicle_id} data: {vehicle_info}")
    #
    # for walker_id, walker_info in obj_data['walkers'].items():
    #     print(f"Walker {walker_id} data: {walker_info}")

    pickle_data = pkl.dumps(obj_data)
    # 保存到文件
    with open(dataset_path / 'obj_info.pkl', 'wb') as f:
        f.write(pickle_data)

