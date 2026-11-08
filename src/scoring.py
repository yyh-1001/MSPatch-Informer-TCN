"""预测残差到异常分数的可审计转换。

预测模型输出的是每个传感器下一时刻的预测值。直接平均所有通道的平方误差，会让只影响
少数传感器的异常被大量正常通道稀释。本模块提供仅利用训练残差拟合的稳健 Top-k 聚合：
先按通道的 MAD 尺度标准化绝对残差，再平均误差最大的若干通道。
"""

from typing import Dict, Tuple

import numpy as np


def fit_residual_mad_calibration(
    train_residuals: np.ndarray,
    min_scale: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """仅从正常训练残差估计每个通道的中位数和稳健尺度。

    ``1.4826 * MAD`` 与高斯分布下的标准差尺度一致。对近似恒定的通道使用最小尺度，
    防止除零；训练数据之外的任何标签或测试分数都不会参与该标定。
    """
    residuals = np.asarray(train_residuals, dtype=np.float64)
    if residuals.ndim != 2 or residuals.shape[0] == 0:
        raise ValueError("train_residuals must have shape [samples, channels] with at least one sample")
    if not np.isfinite(residuals).all():
        raise ValueError("train_residuals contains NaN or Inf")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")

    center = np.median(residuals, axis=0)
    mad = np.median(np.abs(residuals - center), axis=0)
    scale = np.maximum(1.4826 * mad, float(min_scale))
    return center, scale


def robust_topk_residual_scores(
    residuals: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    topk_fraction: float,
) -> Tuple[np.ndarray, int]:
    """将逐通道绝对残差转换为非负的稳健 Top-k 异常分数。

    低于训练期典型误差的残差不构成异常证据，故归一化后截断为 0。每个时间点只汇总
    误差最大的 ``topk_fraction`` 通道；例如 38 个 SMD 通道、比例 0.2 时取最大的 8 个。
    """
    values = np.asarray(residuals, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != len(center) or len(center) != len(scale):
        raise ValueError("residuals, center and scale have incompatible channel dimensions")
    if not 0.0 < float(topk_fraction) <= 1.0:
        raise ValueError("topk_fraction must be in (0, 1]")

    channel_count = values.shape[1]
    topk_channels = max(1, int(np.ceil(channel_count * float(topk_fraction))))
    normalized = np.maximum((values - center) / scale, 0.0)
    # partition 比完整排序更省时；右侧的 top-k 元素无需再按大小排序即可求均值。
    topk_values = np.partition(normalized, channel_count - topk_channels, axis=1)[:, -topk_channels:]
    return topk_values.mean(axis=1), topk_channels


def score_config(config: Dict[str, object]) -> Dict[str, object]:
    """规范化评分配置，并为历史配置保留原始均值 MSE 行为。"""
    raw = dict(config or {})
    method = str(raw.get("method", "model_default"))
    if method not in {"model_default", "robust_topk_residual"}:
        raise ValueError("scoring.method must be 'model_default' or 'robust_topk_residual'")
    return {
        "method": method,
        "topk_fraction": float(raw.get("topk_fraction", 0.2)),
        "min_scale": float(raw.get("min_scale", 1e-6)),
    }
