"""训练脚本共用的轻量工具：JSON、随机种子、设备和参数统计。"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


PathLike = Union[str, os.PathLike]


def load_json(path: PathLike) -> Dict[str, Any]:
    """以 UTF-8 读取 JSON 配置或已保存的实验摘要。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: PathLike) -> None:
    """创建父目录后以 UTF-8、缩进格式保存 JSON，保留中文。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 与 PyTorch 的主要随机源。

    训练仍可能因 CUDA 算子、不同硬件或分布式规约出现极小数值差异；这里优先保证
    同一环境下实验可重复，同时关闭 cuDNN 的自动算法搜索。
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def resolve_device(device: str):
    """将 ``auto`` 映射为可用 CUDA，否则保留调用方指定的设备字符串。"""
    import torch

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def count_parameters(model) -> int:
    """统计需要反向传播更新的参数总数。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
