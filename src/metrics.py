"""异常分数后处理、阈值判别和逐点/事件级评估指标。

这里刻意把“分数如何变成异常标签”和“如何计算指标”独立于模型。这样不同论文
方法的训练目标可以不同，但最终评估协议完全一致。
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


TAB_LABEL_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "range_precision",
    "range_recall",
    "range_f1",
    "affiliated_precision",
    "affiliated_recall",
    "affiliated_f1",
)
TAB_SCORE_METRIC_KEYS = (
    "auc_pr",
    "auc_roc",
    "r_auc_pr",
    "r_auc_roc",
    "vus_pr",
    "vus_roc",
)


@lru_cache(maxsize=1)
def _load_tsb_ad_reference_implementations():
    """直接加载 TSB-AD 官方仓库的评测函数，而非下游项目的内置副本。"""
    project_root = Path(__file__).resolve().parents[1]
    tsb_root = project_root / "benchmark" / "tsb-ad"
    if not tsb_root.is_dir():
        raise RuntimeError("TSB-AD 官方评测目录不存在: {}".format(tsb_root))
    if str(tsb_root) not in sys.path:
        sys.path.insert(0, str(tsb_root))

    from TSB_AD.evaluation.metrics import get_metrics
    from TSB_AD.utils.slidingWindows import find_length_rank

    return find_length_rank, get_metrics


def tsb_ad_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    reference_values: np.ndarray,
) -> Dict[str, object]:
    """按 TSB-AD 官方代码计算六项主指标。

    ``reference_values`` 必须与 ``scores`` 和 ``y_true`` 一一对齐。MSPatch 是预测式模型，
    最早的 ``window_size`` 个点没有完整历史窗口，因此这里评估所有可产生预测目标的尾部；
    不用人工补零或伪造分数。VUS 的窗口长度仍完全由 TSB-AD 官方 ACF 规则确定。

    ``Standard-F1`` 与 ``R-based-F1`` 是 TSB-AD 评测器在分数上做 oracle 阈值扫描得到
    的 Point-F1 与 Range-F1；该过程不执行 Point Adjust。其余四项只依赖连续分数。
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    anomaly_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = np.asarray(reference_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("reference_values must have shape [time, channels]")
    if len(labels) != len(anomaly_scores) or len(labels) != len(values):
        raise ValueError("labels, scores and reference_values must have the same length")
    if not len(labels) or len(np.unique(labels)) < 2 or not np.isfinite(anomaly_scores).all():
        return {
            "protocol": "tsb_ad_official",
            "score_alignment": "forecast_target_tail_only",
            "sliding_window": None,
            "metrics": {
                "vus_pr": None,
                "vus_roc": None,
                "range_f1": None,
                "auc_pr": None,
                "auc_roc": None,
                "point_f1": None,
            },
        }

    find_length_rank, get_metrics = _load_tsb_ad_reference_implementations()
    # TSB-AD 使用第一通道的 ACF 自动选择 VUS 容忍窗口。
    sliding_window = int(find_length_rank(values[:, 0].reshape(-1, 1), rank=1))
    official = get_metrics(
        anomaly_scores,
        labels,
        slidingWindow=sliding_window,
        pred=None,
        version="opt",
        thre=250,
    )
    return {
        "protocol": "tsb_ad_official",
        "score_alignment": "forecast_target_tail_only",
        "sliding_window": sliding_window,
        "point_adjust_applied": False,
        "threshold_note": "Point-F1 and Range-F1 use TSB-AD evaluator oracle threshold scans; no Point Adjust.",
        "metrics": {
            "vus_pr": _finite_or_none(official.get("VUS-PR")),
            "vus_roc": _finite_or_none(official.get("VUS-ROC")),
            "range_f1": _finite_or_none(official.get("R-based-F1")),
            "auc_pr": _finite_or_none(official.get("AUC-PR")),
            "auc_roc": _finite_or_none(official.get("AUC-ROC")),
            "point_f1": _finite_or_none(official.get("Standard-F1")),
        },
    }


def nearest_rank_quantile(scores: np.ndarray, quantile: float = 0.99) -> float:
    """校准分数的 nearest-rank 分位数：S_(ceil(q*n))，不做线性插值。"""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a non-empty finite 1-D array")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("calibration quantile must be in (0, 1]")
    rank = int(np.ceil(quantile * len(values))) - 1
    return float(np.partition(values, rank)[rank])


def smooth_scores(
    scores: np.ndarray, window: int, mode: str = "causal", groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    """平滑窗口级异常分数，默认严格使用当前及历史分数。

    ``window=1`` 表示不平滑。``mode='causal'`` 是默认且可部署的 trailing
    moving average；``mode='offline_centered'`` 仅供明确标记的离线分析使用，
    因为它会读取未来分数。该操作只改变判别时的分数序列，不参与模型训练。
    """
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(scores):
            raise ValueError("smoothing groups must align with scores")
        boundaries = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(scores)]
        result = np.empty(len(scores), dtype=np.float64)
        for begin, end in zip(boundaries[:-1], boundaries[1:]):
            result[begin:end] = smooth_scores(scores[begin:end], window, mode)
        return result
    if window <= 1:
        return scores.astype(np.float64, copy=False)
    values = scores.astype(np.float64, copy=False)
    mode = str(mode).lower()
    if mode == "causal":
        cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        positions = np.arange(len(values))
        starts = np.maximum(0, positions - window + 1)
        totals = cumulative[positions + 1] - cumulative[starts]
        counts = positions - starts + 1
        return totals / counts
    if mode == "offline_centered":
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(values, kernel, mode="same")
    raise ValueError("score smoothing mode must be 'causal' or 'offline_centered'")


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """根据二值真实标签和预测标签计算逐点 Precision、Recall、F1 与 TP/FP/FN。"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _positive_ranges(values: np.ndarray) -> List[Tuple[int, int]]:
    """将二值序列转换为右开区间 ``[start, end)`` 的连续异常事件。"""
    binary = np.asarray(values, dtype=np.int64).reshape(-1)
    ranges: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for index, value in enumerate(binary):
        if value == 1 and start is None:
            start = index
        elif value == 0 and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(binary)))
    return ranges


@lru_cache(maxsize=1)
def _load_tab_reference_implementations():
    """延迟加载项目内保存的 TAB 官方评测实现。

    Range-P/R/F、Range-AUC/VUS 与 Affiliation 指标均直接从 ``benchmark/tab``
    导入。这样评测来源与 TAB 仓库保持一致，不维护容易产生语义偏差的本地重写版本。
    """
    project_root = Path(__file__).resolve().parents[1]
    tab_root = project_root / "benchmark" / "tab"
    if not tab_root.is_dir():
        raise RuntimeError("TAB 官方评测目录不存在: {}".format(tab_root))
    if str(tab_root) not in sys.path:
        sys.path.insert(0, str(tab_root))

    from ts_benchmark.evaluation.metrics.vus_metrics import generate_curve, metricor
    from ts_benchmark.evaluation.metrics.affiliation.generics import convert_vector_to_events
    from ts_benchmark.evaluation.metrics.affiliation.metrics import pr_from_events

    return metricor, generate_curve, convert_vector_to_events, pr_from_events


def _finite_or_none(value: object) -> Optional[float]:
    """将第三方评测器返回的 NumPy 标量或 NaN 统一为 JSON 友好的 Python 数值。"""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if np.isfinite(numeric_value) else None


def tab_label_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    """计算 TAB 图中全部标签型指标，预测必须是未经 Point Adjust 的原始标签。

    Point Adjust 是一项后处理协议，并不属于 TAB 表中的独立指标。为避免把一次命中整段
    异常的人为扩张误认为模型能力，这里固定使用 ``y_pred`` 的原始阈值结果。
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predictions = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if len(labels) != len(predictions):
        raise ValueError("y_true and y_pred must have the same length")

    point_metrics = precision_recall_f1(labels, predictions)
    true_negatives = int(((labels == 0) & (predictions == 0)).sum())
    accuracy = float((point_metrics["tp"] + true_negatives) / len(labels)) if len(labels) else None

    true_ranges = _positive_ranges(labels)
    if true_ranges:
        # TAB 引用 Tatbul et al. 的范围指标。这里直接调用 TAB 的 metric_new，
        # 而非维护一份容易产生语义偏差的本地重写版本。该实现采用 alpha=0.2
        # 和 flat positional bias，并返回 Range Recall/P/R-F1。
        metricor, _, _, _ = _load_tab_reference_implementations()
        range_values = metricor().metric_new(labels, predictions, plot_ROC=False, alpha=0.2)
        range_recall = _finite_or_none(range_values[4])
        range_precision = _finite_or_none(range_values[7])
        range_f1 = _finite_or_none(range_values[8])
    else:
        # 没有真实异常时，范围和关联指标均不可定义，宏平均时会被跳过。
        range_precision = None
        range_recall = None
        range_f1 = None

    affiliated_precision: Optional[float]
    affiliated_recall: Optional[float]
    affiliated_f1: Optional[float]
    true_events = true_ranges
    predicted_events = _positive_ranges(predictions)
    if not true_events:
        affiliated_precision = None
        affiliated_recall = None
        affiliated_f1 = None
    elif not predicted_events:
        affiliated_precision = 0.0
        affiliated_recall = 0.0
        affiliated_f1 = 0.0
    else:
        _, _, convert_vector_to_events, pr_from_events = _load_tab_reference_implementations()
        affiliation = pr_from_events(
            convert_vector_to_events(predictions.tolist()),
            convert_vector_to_events(labels.tolist()),
            (0, len(labels)),
        )
        affiliated_precision = _finite_or_none(affiliation["precision"])
        affiliated_recall = _finite_or_none(affiliation["recall"])
        affiliated_f1 = (
            2.0 * affiliated_precision * affiliated_recall / (affiliated_precision + affiliated_recall)
            if affiliated_precision is not None
            and affiliated_recall is not None
            and affiliated_precision + affiliated_recall > 0.0
            else 0.0
            if affiliated_precision is not None and affiliated_recall is not None
            else None
        )

    return {
        "accuracy": accuracy,
        "precision": float(point_metrics["precision"]),
        "recall": float(point_metrics["recall"]),
        "f1": float(point_metrics["f1"]),
        "range_precision": range_precision,
        "range_recall": range_recall,
        "range_f1": range_f1,
        "affiliated_precision": affiliated_precision,
        "affiliated_recall": affiliated_recall,
        "affiliated_f1": affiliated_f1,
    }


def _median_event_length(labels: np.ndarray) -> Optional[int]:
    """按 TSB-UAD 的默认策略，取真实异常事件长度的中位数作为范围缓冲宽度。"""
    event_ranges = _positive_ranges(labels)
    if not event_ranges:
        return None
    lengths = [end - start for start, end in event_ranges]
    return max(1, int(np.median(lengths)))


def tab_score_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, Optional[float]]:
    """计算 TAB 图中的 AUC、Range-AUC 和 VUS 分数型指标。

    这些指标直接使用连续异常分数，既不需要阈值，也不应用 Point Adjust。范围缓冲宽度
    使用当前独立序列真实异常事件长度的中位数，与 TSB-UAD/CATCH 评测器默认方式一致。
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    anomaly_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(anomaly_scores):
        raise ValueError("y_true and scores must have the same length")
    if not len(labels) or not np.isfinite(anomaly_scores).all() or len(np.unique(labels)) < 2:
        return {key: None for key in TAB_SCORE_METRIC_KEYS}

    auc_pr = float(average_precision_score(labels, anomaly_scores))
    auc_roc = float(roc_auc_score(labels, anomaly_scores))
    range_window = _median_event_length(labels)
    if range_window is None:
        return {
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "r_auc_pr": None,
            "r_auc_roc": None,
            "vus_pr": None,
            "vus_roc": None,
        }

    metricor, generate_curve, _, _ = _load_tab_reference_implementations()
    r_auc_roc, r_auc_pr, _, _, _ = metricor().RangeAUC(
        labels=labels,
        score=anomaly_scores,
        window=range_window,
        plot_ROC=True,
    )
    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(
        labels,
        anomaly_scores,
        2 * range_window,
    )
    return {
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "r_auc_pr": _finite_or_none(r_auc_pr),
        "r_auc_roc": _finite_or_none(r_auc_roc),
        "vus_pr": _finite_or_none(vus_pr),
        "vus_roc": _finite_or_none(vus_roc),
    }


def _mean_available_metrics(
    rows: Sequence[Dict[str, object]],
    keys: Sequence[str],
) -> Dict[str, Optional[float]]:
    """对分组结果作无权重宏平均，自动跳过无异常而不可定义的指标。"""
    averaged: Dict[str, Optional[float]] = {}
    for key in keys:
        values = [_finite_or_none(row.get(key)) for row in rows]
        finite_values = [value for value in values if value is not None]
        averaged[key] = float(np.mean(finite_values)) if finite_values else None
    return averaged


def tab_evaluation_metrics(
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    scores: np.ndarray,
    groups: Optional[Sequence[object]] = None,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """计算完整 TAB 指标，并保证多机器/多运行段不跨边界合并事件。

    单一序列直接输出所有指标。存在 ``groups`` 时，每个组独立计算范围、关联和 VUS 指标，
    主结果为无权重 Macro 均值；同时保留全体点的 Micro Accuracy/P/R/F1。返回的第二项可
    直接保存为 ``tab_group_metrics.csv``。
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predictions = np.asarray(y_pred_raw, dtype=np.int64).reshape(-1)
    anomaly_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(predictions) or len(labels) != len(anomaly_scores):
        raise ValueError("labels, predictions and scores must have the same length")

    if groups is None:
        label_metrics = tab_label_metrics(labels, predictions)
        score_metrics = tab_score_metrics(labels, anomaly_scores)
        return (
            {
                "aggregation": "single_sequence",
                "label_prediction_protocol": "thresholded_raw_predictions_without_point_adjust",
                "score_metric_protocol": "raw_anomaly_scores; range/VUS window=median_ground_truth_event_length",
                "label_metrics_raw": label_metrics,
                "score_metrics": score_metrics,
                "group_count": 1,
            },
            [],
        )

    group_values = np.asarray(groups, dtype=object).reshape(-1)
    if len(group_values) != len(labels):
        raise ValueError("groups must have the same length as labels")

    group_rows: List[Dict[str, object]] = []
    seen_groups = set()
    ordered_groups: List[object] = []
    for group in group_values:
        group_key = str(group)
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            ordered_groups.append(group)
    for group in ordered_groups:
        mask = group_values == group
        group_labels = labels[mask]
        group_predictions = predictions[mask]
        group_scores = anomaly_scores[mask]
        row: Dict[str, object] = {
            "group": str(group),
            "sample_count": int(len(group_labels)),
            "anomaly_count": int(group_labels.sum()),
            "vus_window": _median_event_length(group_labels),
        }
        row.update(tab_label_metrics(group_labels, group_predictions))
        row.update(tab_score_metrics(group_labels, group_scores))
        group_rows.append(row)

    point_micro = precision_recall_f1(labels, predictions)
    true_negatives = int(((labels == 0) & (predictions == 0)).sum())
    point_micro_with_accuracy: Dict[str, Optional[float]] = {
        "accuracy": float((point_micro["tp"] + true_negatives) / len(labels)) if len(labels) else None,
        "precision": float(point_micro["precision"]),
        "recall": float(point_micro["recall"]),
        "f1": float(point_micro["f1"]),
    }
    label_macro = _mean_available_metrics(group_rows, TAB_LABEL_METRIC_KEYS)
    score_macro = _mean_available_metrics(group_rows, TAB_SCORE_METRIC_KEYS)
    return (
        {
            "aggregation": "per_group_macro",
            "label_prediction_protocol": "thresholded_raw_predictions_without_point_adjust",
            "score_metric_protocol": "raw_anomaly_scores; each group uses median_ground_truth_event_length as range/VUS window",
            "label_metrics_raw": label_macro,
            "score_metrics": score_macro,
            "label_metrics_raw_macro": label_macro,
            "score_metrics_macro": score_macro,
            "point_metrics_raw_micro": point_micro_with_accuracy,
            "group_count": len(group_rows),
        },
        group_rows,
    )


def point_adjust_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[Sequence[object]] = None,
) -> np.ndarray:
    """执行 Point Adjust (PA) 后处理，并可按独立时间序列切分异常区间。

    对每一段连续真实异常，只要预测在其中任一点命中，就把整段预测改为异常。``groups``
    用于 SMD 的不同机器或 HAI 的不同运行段：组边界两侧即使标签都为异常，也必须视为
    两个独立事件，不能被 PA 合并。PA 会提高区间级召回率，因此训练脚本始终同时导出
    PA 和严格逐点两份结果。
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    adjusted = np.asarray(y_pred, dtype=np.int64).reshape(-1).copy()
    if len(labels) != len(adjusted):
        raise ValueError("y_true and y_pred must have the same length")

    if groups is None:
        group_values = None
    else:
        group_values = np.asarray(groups, dtype=object).reshape(-1)
        if len(group_values) != len(labels):
            raise ValueError("groups must have the same length as y_true")

    index = 0
    while index < len(labels):
        same_previous_group = (
            group_values is None or index == 0 or group_values[index] == group_values[index - 1]
        )
        if labels[index] != 1 or (index > 0 and labels[index - 1] == 1 and same_previous_group):
            index += 1
            continue

        start = index
        group = None if group_values is None else group_values[index]
        index += 1
        while index < len(labels):
            same_group = group_values is None or group_values[index] == group
            if labels[index] != 1 or not same_group:
                break
            index += 1
        if adjusted[start:index].any():
            adjusted[start:index] = 1
    return adjusted


def threshold_predictions(
    scores: np.ndarray,
    threshold: float,
    y_true: Optional[np.ndarray] = None,
    point_adjust: bool = False,
    groups: Optional[Sequence[object]] = None,
) -> np.ndarray:
    """以 ``score > threshold`` 产生原始预测，并按需执行按组切分的 PA。"""
    pred = (scores > threshold).astype(np.int64)
    if point_adjust and y_true is not None:
        pred = point_adjust_predictions(y_true, pred, groups)
    return pred


def search_best_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    grid_size: int = 1000,
    point_adjust: bool = False,
    groups: Optional[Sequence[object]] = None,
) -> Tuple[float, Dict[str, float]]:
    """在测试标签上搜索使 F1 最高的异常分数阈值。

    当 ``point_adjust=True`` 时，搜索目标是按 ``groups`` 切分事件后得到的 PA-F1；否则
    搜索严格逐点 F1。该评测协议会使用测试标签，适合复现采用 best-F1/PA 的异常检测
    文献，但不属于部署场景下的严格无标签泛化评估。
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if len(scores) != len(y_true):
        raise ValueError("scores and y_true must have the same length")
    if len(scores) == 0:
        raise ValueError("cannot search threshold on empty scores")

    unique_scores = np.unique(scores)
    if len(unique_scores) <= grid_size:
        candidate_scores = unique_scores
    else:
        qs = np.linspace(0.0, 100.0, grid_size)
        candidate_scores = np.unique(np.percentile(scores, qs))

    # 判别规则是 ``score > threshold``。对候选分数直接当作阈值会遗漏所有与它相等的
    # 样本；向负无穷方向移动一个浮点单位后，候选分数及更高分数都会被正确判为异常。
    # 额外保留“大于最大分数”的候选，覆盖全为正常预测这一边界情况。
    thresholds = np.concatenate(
        [
            np.nextafter(candidate_scores, -np.inf),
            np.asarray([np.nextafter(float(np.max(scores)), np.inf)]),
        ]
    )

    best_thr = float(thresholds[0])
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": -1.0}
    for thr in thresholds:
        # 相同 F1 时保持较早阈值，保证固定输入下结果稳定可复现。
        pred = threshold_predictions(scores, float(thr), y_true, point_adjust, groups)
        metrics = precision_recall_f1(y_true, pred)
        if metrics["f1"] > best_metrics["f1"]:
            best_thr = float(thr)
            best_metrics = metrics
    return best_thr, best_metrics
