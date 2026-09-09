import math
import os
import pprint
from pathlib import Path
import numpy
import numpy as np
import pandas as pd
import torch
import cv2


def to_tensor(data):
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    elif isinstance(data, numpy.ndarray):
        if data.dtype == numpy.float64:
            return torch.from_numpy(data).float()
        else:
            return torch.from_numpy(data)
    elif isinstance(data, numpy.number):
        return torch.tensor(data).float()
    elif isinstance(data, list):
        return data
    elif isinstance(data, int):
        return torch.tensor(data)
    elif isinstance(data, tuple):
        return to_tensor(data[0])
    else:
        print(type(data), data)
        raise NotImplementedError


def to_numpy(data):
    if isinstance(data, dict):
        return {k: to_numpy(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.requires_grad:
            return data.detach().cpu().numpy()
        else:
            return data.cpu().numpy()
    else:
        print(type(data), data)
        raise NotImplementedError


def enable_grad(data):
    if isinstance(data, dict):
        return {k: enable_grad(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        if data.dtype == torch.float32:
            data.requires_grad = True
    else:
        raise NotImplementedError


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    else:
        # raise NotImplementedError
        return data


def print_dict_tensor(data, prefix=""):
    for k, v in data.items():
        if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
            print(f"{prefix}{k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"{prefix}{k}:")
            print_dict_tensor(v, "    ")


def save_dict_to_hdf5(h5_group, data_dict):
    """
    data_dict是PlutoFeature.data，字典
    递归地将嵌套字典（包含 ndarray、标量等）保存到 HDF5 Group 中。
    """
    for key, value in data_dict.items():
        if isinstance(value, dict):
            # 如果是字典，则在 HDF5 中创建一个对应的子组 (Sub-Group)，然后递归调用
            sub_group = h5_group.create_group(key)
            save_dict_to_hdf5(sub_group, value)

        elif isinstance(value, np.ndarray):
            if value.dtype == object:
                # 应对 object 类型的 numpy array，转成 string 存储
                try:
                    import json
                    val_str = json.dumps(value.tolist(), default=str)
                    h5_group.create_dataset(key, data=np.bytes_(val_str))
                except Exception:
                    pass
            else:
                # 正常的 numpy 数组直接存
                h5_group.create_dataset(key, data=value)

        elif isinstance(value, str):
            h5_group.create_dataset(key, data=np.bytes_(value))

        elif isinstance(value, (int, float, bool, np.generic)):
            # 基础标量类型直接存 (例如图中的 float64)
            h5_group.create_dataset(key, data=value)

        elif isinstance(value, (list, tuple)):
            # 如果是普通的 list/tuple，尝试转成 numpy 数组后存
            try:
                arr_val = np.array(value)
                if arr_val.dtype == object:
                    import json
                    val_str = json.dumps(value, default=str)
                    h5_group.create_dataset(key, data=np.bytes_(val_str))
                else:
                    h5_group.create_dataset(key, data=arr_val)
            except Exception:
                pass
