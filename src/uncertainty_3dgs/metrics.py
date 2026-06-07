"""Uncertainty and failure-prediction metrics without heavy ML dependencies."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

try:  # Optional acceleration and array compatibility.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - depends on environment.
    _np = None


def evaluate_uncertainty(
    uncertainty: object,
    error: object,
    *,
    bad_threshold: float | None = None,
    mask: object | None = None,
    sparsification_fractions: Sequence[float] | None = None,
    num_reliability_bins: int = 10,
) -> dict[str, object]:
    """Return a compact metric summary for uncertainty-error alignment.

    ``uncertainty`` and ``error`` may be flat lists, nested lists, tuples, or
    numpy arrays when numpy is installed. Higher uncertainty is assumed to mean
    higher predicted error.
    """

    pairs = _finite_pairs(uncertainty, error, mask=mask)
    uncertainties = [pair[0] for pair in pairs]
    errors = [pair[1] for pair in pairs]
    summary: dict[str, object] = {
        "count": len(pairs),
        "mean_error": _mean(errors),
        "spearman": spearman_correlation(uncertainties, errors),
        "risk_coverage_auc": area_under_risk_coverage(
            risk_coverage_curve(errors, uncertainties)
        ),
        "sparsification": sparsification_summary(
            errors,
            uncertainties,
            fractions=sparsification_fractions,
        ),
        "uncertainty_bins": uncertainty_error_bins(
            uncertainties,
            errors,
            bad_threshold=bad_threshold,
            num_bins=num_reliability_bins,
        ),
    }

    if bad_threshold is not None:
        labels = [1 if err >= bad_threshold else 0 for err in errors]
        summary.update(
            {
                "bad_threshold": bad_threshold,
                "bad_fraction": _mean(labels),
                "auroc": roc_auc_score(uncertainties, labels),
                "auprc": average_precision_score(uncertainties, labels),
            }
        )

    return summary


def uncertainty_error_bins(
    uncertainty: object,
    error: object,
    *,
    bad_threshold: float | None = None,
    num_bins: int = 10,
    mask: object | None = None,
) -> list[dict[str, float | int]]:
    """Equal-count bins for reliability-style uncertainty/error plots.

    The input uncertainty does not need to be a calibrated probability. Bins are
    sorted by increasing uncertainty and report the observed mean error in each
    bin. When ``bad_threshold`` is provided, each bin also reports the empirical
    bad-sample fraction.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    pairs = sorted(_finite_pairs(uncertainty, error, mask=mask), key=lambda pair: pair[0])
    if not pairs:
        return []

    total = len(pairs)
    bins: list[dict[str, float | int]] = []
    for bin_index in range(num_bins):
        start = math.floor(bin_index * total / num_bins)
        end = math.floor((bin_index + 1) * total / num_bins)
        if start == end:
            continue

        values = pairs[start:end]
        uncertainties = [value[0] for value in values]
        errors = [value[1] for value in values]
        row: dict[str, float | int] = {
            "bin": bin_index,
            "count": len(values),
            "lower_quantile": start / total,
            "upper_quantile": end / total,
            "min_uncertainty": min(uncertainties),
            "max_uncertainty": max(uncertainties),
            "mean_uncertainty": _mean(uncertainties),
            "mean_error": _mean(errors),
        }
        if bad_threshold is not None:
            row["bad_fraction"] = _mean(1.0 if err >= bad_threshold else 0.0 for err in errors)
        bins.append(row)
    return bins


def spearman_correlation(
    uncertainty: object,
    error: object,
    *,
    mask: object | None = None,
) -> float:
    """Spearman rank correlation between uncertainty and observed error."""

    pairs = _finite_pairs(uncertainty, error, mask=mask)
    if len(pairs) < 2:
        return math.nan

    x_ranks = _average_ranks([pair[0] for pair in pairs])
    y_ranks = _average_ranks([pair[1] for pair in pairs])
    return _pearson(x_ranks, y_ranks)


def roc_auc_score(
    scores: object,
    labels: object,
    *,
    mask: object | None = None,
) -> float:
    """Rank-based ROC AUC for binary labels.

    Higher scores are treated as stronger predictions of the positive class.
    Returns ``nan`` when only one class is present.
    """

    pairs = _finite_pairs(scores, labels, mask=mask)
    if not pairs:
        return math.nan

    numeric_scores = [pair[0] for pair in pairs]
    binary_labels = [_as_binary_label(pair[1]) for pair in pairs]
    positives = sum(binary_labels)
    negatives = len(binary_labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    ranks = _average_ranks(numeric_scores)
    positive_rank_sum = sum(
        rank for rank, label in zip(ranks, binary_labels) if label == 1
    )
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def average_precision_score(
    scores: object,
    labels: object,
    *,
    mask: object | None = None,
) -> float:
    """Average precision for binary failure prediction."""

    pairs = _finite_pairs(scores, labels, mask=mask)
    if not pairs:
        return math.nan

    ordered = sorted(
        ((score, _as_binary_label(label)) for score, label in pairs),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(label for _, label in ordered)
    if positives == 0:
        return math.nan

    precision_area = 0.0
    true_positive = 0
    false_positive = 0
    previous_recall = 0.0
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        group_positive = 0
        group_negative = 0
        while index < len(ordered) and ordered[index][0] == score:
            if ordered[index][1] == 1:
                group_positive += 1
            else:
                group_negative += 1
            index += 1

        true_positive += group_positive
        false_positive += group_negative
        if group_positive == 0:
            continue

        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive)
        precision_area += precision * (recall - previous_recall)
        previous_recall = recall

    return precision_area


def risk_coverage_curve(
    error: object,
    uncertainty: object,
    *,
    mask: object | None = None,
) -> list[dict[str, float | int]]:
    """Risk-coverage curve when keeping lowest-uncertainty samples first."""

    pairs = _finite_pairs(uncertainty, error, mask=mask)
    if not pairs:
        return []

    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = len(ordered)
    cumulative_error = 0.0
    curve: list[dict[str, float | int]] = []
    for index, (_, observed_error) in enumerate(ordered, start=1):
        cumulative_error += observed_error
        curve.append(
            {
                "coverage": index / total,
                "risk": cumulative_error / index,
                "count": index,
            }
        )
    return curve


def area_under_risk_coverage(
    curve: Sequence[dict[str, float | int]],
) -> float:
    """Trapezoidal area under a risk-coverage curve."""

    if not curve:
        return math.nan
    if len(curve) == 1:
        return float(curve[0]["risk"])

    area = 0.0
    previous_coverage = 0.0
    previous_risk = float(curve[0]["risk"])
    for point in curve:
        coverage = float(point["coverage"])
        risk = float(point["risk"])
        area += (coverage - previous_coverage) * (previous_risk + risk) / 2
        previous_coverage = coverage
        previous_risk = risk
    return area


def sparsification_summary(
    error: object,
    uncertainty: object,
    *,
    mask: object | None = None,
    fractions: Sequence[float] | None = None,
) -> dict[str, object]:
    """Compare uncertainty-based sparsification against the oracle curve."""

    pairs = _finite_pairs(uncertainty, error, mask=mask)
    if not pairs:
        return {"ause": math.nan, "curve": []}

    if fractions is None:
        fractions = tuple(index / 20 for index in range(20))
    _validate_fractions(fractions)

    errors_by_uncertainty = [
        error for _, error in sorted(pairs, key=lambda pair: pair[0])
    ]
    oracle_errors = sorted((error for _, error in pairs))
    total = len(pairs)
    curve: list[dict[str, float | int]] = []
    gaps: list[tuple[float, float]] = []

    for fraction in fractions:
        remove_count = min(total - 1, int(math.floor(total * fraction)))
        keep_count = total - remove_count
        uncertainty_risk = _mean(errors_by_uncertainty[:keep_count])
        oracle_risk = _mean(oracle_errors[:keep_count])
        gap = abs(uncertainty_risk - oracle_risk)
        gaps.append((float(fraction), gap))
        curve.append(
            {
                "removed_fraction": float(fraction),
                "coverage": keep_count / total,
                "risk": uncertainty_risk,
                "oracle_risk": oracle_risk,
                "gap": gap,
                "count": keep_count,
            }
        )

    return {"ause": _trapezoid(gaps), "curve": curve}


def reliability_diagram(
    probabilities: object,
    labels: object,
    *,
    num_bins: int = 10,
    mask: object | None = None,
) -> dict[str, object]:
    """Binary reliability bins and expected calibration error.

    Probabilities are expected to be in ``[0, 1]`` and labels in ``{0, 1}``.
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    pairs = _finite_pairs(probabilities, labels, mask=mask)
    bins = [
        {
            "lower": index / num_bins,
            "upper": (index + 1) / num_bins,
            "count": 0,
            "confidence": 0.0,
            "accuracy": 0.0,
            "gap": math.nan,
        }
        for index in range(num_bins)
    ]
    if not pairs:
        return {"ece": math.nan, "bins": bins}

    bucket_values: list[list[tuple[float, int]]] = [[] for _ in range(num_bins)]
    for probability, label in pairs:
        if probability < 0.0 or probability > 1.0:
            raise ValueError("Probabilities must be in [0, 1].")
        index = min(num_bins - 1, int(probability * num_bins))
        bucket_values[index].append((probability, _as_binary_label(label)))

    total = len(pairs)
    ece = 0.0
    for index, values in enumerate(bucket_values):
        if not values:
            continue
        confidence = _mean(probability for probability, _ in values)
        accuracy = _mean(label for _, label in values)
        gap = abs(confidence - accuracy)
        bins[index].update(
            {
                "count": len(values),
                "confidence": confidence,
                "accuracy": accuracy,
                "gap": gap,
            }
        )
        ece += len(values) / total * gap

    return {"ece": ece, "bins": bins}


def gaussian_nll(
    target: object,
    mean: object,
    variance_or_stddev: object,
    *,
    variance: bool = False,
    mask: object | None = None,
    eps: float = 1e-12,
) -> float:
    """Mean Gaussian negative log likelihood for predictive distributions."""

    triplets = _finite_triplets(target, mean, variance_or_stddev, mask=mask)
    if not triplets:
        return math.nan

    losses = []
    for actual, predicted, scale in triplets:
        predictive_variance = scale if variance else scale * scale
        predictive_variance = max(predictive_variance, eps)
        residual = actual - predicted
        losses.append(
            0.5
            * (
                math.log(2.0 * math.pi * predictive_variance)
                + residual * residual / predictive_variance
            )
        )
    return _mean(losses)


def load_metric_json(path: str | Path) -> dict[str, object]:
    """Load metric inputs from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Metric JSON must contain an object.")
    return payload


def _finite_pairs(
    first: object,
    second: object,
    *,
    mask: object | None = None,
) -> list[tuple[float, float]]:
    first_values = _flatten_selected(first, mask=mask)
    second_values = _flatten_selected(second, mask=mask)
    count = min(len(first_values), len(second_values))
    pairs = []
    for left, right in zip(first_values[:count], second_values[:count]):
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            continue
        if math.isfinite(left_value) and math.isfinite(right_value):
            pairs.append((left_value, right_value))
    return pairs


def _finite_triplets(
    first: object,
    second: object,
    third: object,
    *,
    mask: object | None = None,
) -> list[tuple[float, float, float]]:
    first_values = _flatten_selected(first, mask=mask)
    second_values = _flatten_selected(second, mask=mask)
    third_values = _flatten_selected(third, mask=mask)
    count = min(len(first_values), len(second_values), len(third_values))
    triplets = []
    for left, middle, right in zip(
        first_values[:count],
        second_values[:count],
        third_values[:count],
    ):
        try:
            left_value = float(left)
            middle_value = float(middle)
            right_value = float(right)
        except (TypeError, ValueError):
            continue
        if (
            math.isfinite(left_value)
            and math.isfinite(middle_value)
            and math.isfinite(right_value)
        ):
            triplets.append((left_value, middle_value, right_value))
    return triplets


def _flatten_numeric(values: object, *, mask: object | None = None) -> list[float]:
    return [
        float(value)
        for value in _flatten_selected(values, mask=mask)
        if _is_finite_number(value)
    ]


def _flatten_selected(values: object, *, mask: object | None = None) -> list[object]:
    if _np is not None:
        try:
            array = _np.asarray(values)
            if mask is not None:
                array = array[_np.asarray(mask, dtype=bool)]
            array = array.reshape(-1)
            return list(array.tolist())
        except Exception:
            pass

    flat_values = list(_flatten(values))
    if mask is None:
        return flat_values

    flat_mask = list(_flatten(mask))
    return [value for value, keep in zip(flat_values, flat_mask) if bool(keep)]


def _flatten(values: object) -> Iterable[object]:
    if isinstance(values, (str, bytes)):
        yield values
        return
    if isinstance(values, Iterable):
        for value in values:
            yield from _flatten(value)
        return
    yield values


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        start = index
        value = ordered[index][1]
        while index < len(ordered) and ordered[index][1] == value:
            index += 1
        average_rank = (start + 1 + index) / 2
        for original_index, _ in ordered[start:index]:
            ranks[original_index] = average_rank
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_denominator = sum((value - left_mean) ** 2 for value in left)
    right_denominator = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_denominator * right_denominator)
    if denominator == 0.0:
        return math.nan
    return numerator / denominator


def _as_binary_label(value: float) -> int:
    if value in (0.0, 1.0):
        return int(value)
    raise ValueError(f"Binary labels must be 0 or 1, got {value!r}.")


def _validate_fractions(fractions: Sequence[float]) -> None:
    previous = -math.inf
    for fraction in fractions:
        if fraction < 0.0 or fraction >= 1.0:
            raise ValueError("Sparsification fractions must be in [0, 1).")
        if fraction < previous:
            raise ValueError("Sparsification fractions must be sorted ascending.")
        previous = fraction


def _trapezoid(points: Sequence[tuple[float, float]]) -> float:
    if not points:
        return math.nan
    if len(points) == 1:
        return points[0][1]
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def _mean(values: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += float(value)
        count += 1
    if count == 0:
        return math.nan
    return total / count


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
