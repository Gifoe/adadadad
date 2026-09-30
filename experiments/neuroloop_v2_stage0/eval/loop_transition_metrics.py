"""Metrics for the pre-registered same-checkpoint loop trajectory analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    prediction = probabilities.argmax(axis=1)
    value = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        mask = (confidence >= lower) & (confidence < lower + 0.1 if lower < 0.9 else confidence <= 1.0)
        if mask.any():
            value += mask.mean() * abs((prediction[mask] == labels[mask]).mean() - confidence[mask].mean())
    return float(value)


def loop_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    prediction = probabilities.argmax(axis=1)
    target_probability = np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
        "nll": float(-np.log(target_probability).mean()),
        "ece": expected_calibration_error(probabilities, labels),
    }


def transition_counts(labels: np.ndarray, probability_from: np.ndarray, probability_to: np.ndarray) -> dict[str, float | int]:
    previous = probability_from.argmax(axis=1) == labels
    following = probability_to.argmax(axis=1) == labels
    cc = int((previous & following).sum())
    cw = int((previous & ~following).sum())
    wc = int((~previous & following).sum())
    ww = int((~previous & ~following).sum())
    total = len(labels)
    return {
        "CC": cc,
        "CW": cw,
        "WC": wc,
        "WW": ww,
        "correction_rate": wc / total,
        "corruption_rate": cw / total,
        "correction_given_wrong": wc / max(1, wc + ww),
        "corruption_given_correct": cw / max(1, cc + cw),
    }


def confidence_group_metrics(labels: np.ndarray, first: np.ndarray, last: np.ndarray) -> list[dict[str, float | int | str]]:
    confidence = first.max(axis=1)
    groups = {
        "high": confidence >= 0.8,
        "medium": (confidence >= 0.6) & (confidence < 0.8),
        "low": confidence < 0.6,
    }
    rows: list[dict[str, float | int | str]] = []
    for name, mask in groups.items():
        if not mask.any():
            rows.append({"confidence_group": name, "n_trials": 0, "correction_rate": float("nan"), "corruption_rate": float("nan"), "accuracy_gain": float("nan")})
            continue
        transition = transition_counts(labels[mask], first[mask], last[mask])
        rows.append({
            "confidence_group": name,
            "n_trials": int(mask.sum()),
            "correction_rate": transition["correction_rate"],
            "corruption_rate": transition["corruption_rate"],
            "accuracy_gain": float((last[mask].argmax(axis=1) == labels[mask]).mean() - (first[mask].argmax(axis=1) == labels[mask]).mean()),
        })
    return rows
