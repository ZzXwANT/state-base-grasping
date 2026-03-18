import os
import yaml
import copy
import numpy as np
from datetime import datetime


def _serialize(obj):
    # 1. 基础类型直接返回
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    # 2. Numpy 处理
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # 3. 核心修正：优先处理容器，确保 YAML 层级结构
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]

    # 4. 针对复杂类（如 NormalActionNoise 或 LinearSchedule）
    # 只要不是内置容器且不是基础类型，就转为字符串，避免 __dict__ 展开
    # 这样既保留了关键信息，又不会破坏 YAML 的结构
    if hasattr(obj, "__module__"):
        return str(obj)

    return str(obj)


def save_run_config(run_config: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    clean_config = _serialize(copy.deepcopy(run_config))

    output = dict(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        config=clean_config,
    )

    path = os.path.join(save_dir, "run_config.yaml")

    with open(path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Run config saved to {path}")