"""CSV 读取、标准化和滑动窗口样本构造。

仅用拟合集估计标准化参数，验证、校准与测试沿用这些参数。官方训练序列按
70%/10%/20% 划分；SMD 各机器独立划分，窗口不跨子集或机器边界。
"""

from dataclasses import dataclass
import glob
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


POSITIVE_LABELS = {
    "1",
    "true",
    "attack",
    "attacked",
    "abnormal",
    "anomaly",
    "anomalous",
    "yes",
}
NEGATIVE_LABELS = {"0", "false", "normal", "benign", "no"}


@dataclass
class Standardizer:
    """按传感器通道独立计算的 Z-score 标准化器。"""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        """仅用训练部分估计每个传感器的均值和标准差。

        方差近零的恒定通道将标准差置为 1，避免除零且保留该通道与均值的偏差。
        """
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """将 ``[时间步, 通道]`` 原始数值转换为零均值、单位方差表示。"""
        return ((values - self.mean) / self.std).astype(np.float32)


PathLike = Union[str, Path]


def _read_one_csv(path: PathLike) -> pd.DataFrame:
    """读取单个 CSV；由 ``read_table`` 统一处理目录和通配符输入。"""
    return pd.read_csv(path)


def read_table(path_or_glob: PathLike) -> pd.DataFrame:
    """读取一个 CSV、CSV 目录或通配符匹配到的多个 CSV。"""
    text = str(path_or_glob)
    if any(ch in text for ch in ["*", "?", "["]):
        files = [Path(p) for p in sorted(glob.glob(text))]
        if not files:
            raise FileNotFoundError(f"No files matched pattern: {text}")
        return pd.concat([_read_one_csv(p) for p in files], ignore_index=True)

    path = Path(path_or_glob)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No csv files found in directory: {path}")
        return pd.concat([_read_one_csv(p) for p in files], ignore_index=True)
    return _read_one_csv(path)


def parse_labels(series: pd.Series) -> np.ndarray:
    """把常见的数值和文本标签统一为 0（正常）/1（异常）。"""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0).to_numpy() != 0).astype(np.int64)

    def convert(v) -> int:
        s = str(v).strip().lower()
        if s in POSITIVE_LABELS:
            return 1
        if s in NEGATIVE_LABELS:
            return 0
        try:
            return int(float(s) != 0.0)
        except ValueError:
            return 0

    return series.map(convert).to_numpy(dtype=np.int64)


def split_features_labels(
    df: pd.DataFrame,
    label_col: Optional[str],
    time_col: Optional[str],
    group_col: Optional[str],
    drop_cols: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[pd.Series], Optional[np.ndarray], List[str]]:
    """从表格中拆出模型输入、标签、时间戳、分组和特征名。

    无法解析的传感器值会被转为缺失值，并采用前向填充和 0 依次补齐，
    以保证送入 PyTorch 的特征矩阵始终是有限浮点数。这里刻意不使用后向填充：后向
    填充会把未来观测带到当前时间点，破坏在线/因果评测语义。
    """
    drop_cols = list(drop_cols or [])
    labels = None
    timestamps = None
    groups = None
    work = df.copy()

    if time_col and time_col in work.columns:
        timestamps = work[time_col].copy()
        work = work.drop(columns=[time_col])

    if group_col and group_col in work.columns:
        groups = work[group_col].astype(str).to_numpy()
        work = work.drop(columns=[group_col])

    if label_col and label_col in work.columns:
        labels = parse_labels(work[label_col])
        work = work.drop(columns=[label_col])

    existing_drops = [c for c in drop_cols if c in work.columns]
    if existing_drops:
        work = work.drop(columns=existing_drops)

    numeric = work.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if groups is None:
        numeric = numeric.ffill().fillna(0.0)
    else:
        numeric = numeric.groupby(pd.Series(groups, index=numeric.index), sort=False).ffill().fillna(0.0)
    return numeric.to_numpy(dtype=np.float32), labels, timestamps, groups, list(numeric.columns)


class ForecastWindowDataset(Dataset):
    """把连续时序转换为 ``窗口 -> 下一时刻`` 的预测样本。

    返回的 ``target_index`` 始终对应标签和分组中的原始时间位置，后续评分阶段据此
    将窗口级异常分数对齐到测试标签。给定 ``groups`` 时，会丢弃跨机器边界的窗口。
    """

    def __init__(
        self,
        values: np.ndarray,
        window_size: int,
        horizon: int = 1,
        stride: int = 1,
        labels: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        start_offset: int = 0,
        target_indices: Optional[np.ndarray] = None,
    ) -> None:
        """预先计算所有合法窗口的起点，避免每次取样重复判断边界。"""
        if values.ndim != 2:
            raise ValueError("values must have shape [time, channels]")
        self.values = values.astype(np.float32, copy=False)
        self.labels = labels
        self.groups = groups
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.start_offset = int(start_offset)
        self.target_indices = None if target_indices is None else np.asarray(target_indices, dtype=np.int64)
        if min(self.window_size, self.horizon, self.stride) <= 0:
            raise ValueError("window_size, horizon and stride must be positive")
        for name, aligned in [("groups", groups), ("labels", labels), ("target_indices", self.target_indices)]:
            if aligned is not None and len(aligned) != len(values):
                raise ValueError("{} must align with values".format(name))
        boundaries = [0, len(values)]
        if groups is not None and len(values):
            boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(values)]
        starts = [
            np.arange(begin, end - self.window_size - self.horizon + 1, self.stride, dtype=np.int64)
            for begin, end in zip(boundaries[:-1], boundaries[1:])
            if end - begin >= self.window_size + self.horizon
        ]
        self.starts = np.concatenate(starts) if starts else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        """返回输入窗口、预测目标，以及可选标签和其全局时间下标。"""
        start = int(self.starts[index])
        end = start + self.window_size
        target_index = end + self.horizon - 1
        original_index = (
            int(self.target_indices[target_index]) if self.target_indices is not None
            else target_index + self.start_offset
        )
        x = self.values[start:end]
        y = self.values[target_index]
        if self.labels is None:
            return torch.from_numpy(x), torch.from_numpy(y), original_index
        label = int(self.labels[target_index])
        return torch.from_numpy(x), torch.from_numpy(y), label, original_index


@dataclass
class TemporalSubset:
    """子集数值、机器标签以及在官方训练序列中的原始位置。"""

    values: np.ndarray
    groups: Optional[np.ndarray]
    indices: np.ndarray


def split_fit_validation_calibration(
    values: np.ndarray, groups: Optional[np.ndarray] = None,
) -> Tuple[TemporalSubset, TemporalSubset, TemporalSubset]:
    """逐机器按时间做 70/10/20 划分；边界取 floor(0.7*n)、floor(0.8*n)。"""
    values = np.asarray(values)
    groups = None if groups is None else np.asarray(groups, dtype=str)
    if values.ndim != 2 or (groups is not None and len(groups) != len(values)):
        raise ValueError("values must be [time, channels] with aligned groups")
    if groups is None:
        sequences = [np.arange(len(values))]
    else:
        sequences = [np.flatnonzero(groups == group) for group in pd.unique(groups)]
    parts = [[], [], []]
    for indices in sequences:
        n = len(indices)
        fit_end, val_end = (7 * n) // 10, (8 * n) // 10
        if not 0 < fit_end < val_end < n:
            raise ValueError("each sequence must have non-empty fitting, validation and calibration subsets")
        for bucket, selected in zip(parts, (indices[:fit_end], indices[fit_end:val_end], indices[val_end:])):
            bucket.append(selected)
    if not sequences:
        raise ValueError("official training data is empty")
    result = []
    for bucket in parts:
        indices = np.concatenate(bucket)
        result.append(TemporalSubset(values[indices], None if groups is None else groups[indices], indices))
    return tuple(result)


def standardize_subsets(
    fitting: TemporalSubset, validation: TemporalSubset, calibration: TemporalSubset,
    test_values: np.ndarray, test_groups: Optional[np.ndarray],
):
    """只用每个独立序列前 70% 估计统计量，并固定应用于其余子集。"""
    if (fitting.groups is None) != (test_groups is None):
        raise ValueError("training and test data must both have groups or both have no groups")
    inputs = [fitting.values, validation.values, calibration.values, test_values]
    if fitting.groups is None:
        scaler = Standardizer.fit(fitting.values)
        statistics = {"global": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()}}
        return (*[scaler.transform(values) for values in inputs], statistics)
    test_groups = np.asarray(test_groups, dtype=str)
    if len(test_groups) != len(test_values) or set(test_groups) != set(fitting.groups):
        raise ValueError("training and test data must contain the same independent machines")
    labels = [fitting.groups, validation.groups, calibration.groups, test_groups]
    outputs = [np.empty_like(values, dtype=np.float32) for values in inputs]
    statistics = {}
    for group in pd.unique(fitting.groups):
        scaler = Standardizer.fit(fitting.values[fitting.groups == group])
        statistics[str(group)] = {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()}
        for values, group_labels, output in zip(inputs, labels, outputs):
            mask = group_labels == group
            output[mask] = scaler.transform(values[mask])
    return (*outputs, statistics)


def train_val_split(values: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """按时间顺序切分训练尾部作为验证集，不打乱时序。"""
    if val_ratio <= 0:
        return values, values[:0]
    val_len = max(1, int(round(len(values) * val_ratio)))
    if val_len >= len(values):
        raise ValueError("val_ratio leaves no training samples")
    return values[:-val_len], values[-val_len:]


def split_with_optional_groups(
    values: np.ndarray,
    groups: Optional[np.ndarray],
    val_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """切分数值序列，并以完全一致的边界切分机器分组标签。"""
    train_values, val_values = train_val_split(values, val_ratio)
    if groups is None:
        return train_values, val_values, None, None
    split = len(train_values)
    return train_values, val_values, groups[:split], groups[split:]
