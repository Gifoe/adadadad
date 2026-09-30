"""Shared cache, causal EA, online runners, metrics, and audit primitives."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import scipy.io
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from models import NeuroLoop, NeuroLoopSpec


LEFT, RIGHT = 1, 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inverse_sqrt(matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.power(np.maximum(values, eps), -0.5)) @ vectors.T


def trial_covariance(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    return centered @ centered.T / max(1, centered.shape[1] - 1)


def source_ea(source_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.mean([trial_covariance(trial) for trial in source_x], axis=0)
    transform = _inverse_sqrt(ref)
    # source_x is (trials, channels, time), so EA is left multiplication of
    # every channel-by-time trial matrix by the channel-by-channel transform.
    return np.einsum("ij,njt->nit", transform, source_x), transform


def target_ea(target_x: np.ndarray, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """EA that is either full-session paper compatible or strictly causal.

    Strict mode permits the present trial in its own covariance, but never a
    later signal.  There is deliberately no label argument.
    """
    if mode not in {"paper", "strict"}:
        raise ValueError(mode)
    if mode == "paper":
        ref = np.mean([trial_covariance(trial) for trial in target_x], axis=0)
        output = np.einsum("ij,njt->nit", _inverse_sqrt(ref), target_x)
        return output.astype(np.float32), {"mode": mode, "target_trials_seen_per_output": "all"}
    running = np.zeros((target_x.shape[1], target_x.shape[1]), dtype=np.float64)
    output = np.empty_like(target_x, dtype=np.float32)
    seen: list[int] = []
    for index, trial in enumerate(target_x):
        running += trial_covariance(trial)
        output[index] = _inverse_sqrt(running / (index + 1)) @ trial
        seen.append(index + 1)
    return output, {"mode": mode, "target_trials_seen_per_output": seen}


def _events_with_run_id(events: np.ndarray, event_id: dict[str, int]) -> tuple[dict[int, str], list[tuple[int, int]]]:
    reverse = {value: key for key, value in event_id.items()}
    run_starts = [(int(sample), ordinal) for ordinal, (sample, _, code) in enumerate(events) if reverse.get(int(code)) == "32766"]
    return reverse, run_starts


def _run_at(sample: int, starts: list[tuple[int, int]]) -> int:
    prior = [ordinal for start, ordinal in starts if start <= sample]
    return prior[-1] if prior else -1


def _load_session(gdf_path: Path, label_path: Path, session: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[str]]:
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose="ERROR")
    channel_names = [name for name in raw.ch_names if "EOG" not in name.upper()]
    if len(channel_names) != 22:
        raise RuntimeError(f"{gdf_path} has {len(channel_names)} non-EOG channels; expected 22")
    raw.pick(channel_names)
    raw.filter(1.0, 48.0, picks="all", verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    reverse, run_starts = _events_with_run_id(events, event_id)
    cue_events = [(int(sample), reverse[int(code)]) for sample, _, code in events if reverse[int(code)] in {"769", "770", "771", "772", "783"}]
    mat_labels = scipy.io.loadmat(label_path)["classlabel"].reshape(-1).astype(int)
    if session == "S1":
        cue_labels = [{"769": LEFT, "770": RIGHT, "771": 3, "772": 4}[code] for _, code in cue_events]
        labeled = list(zip(cue_events, cue_labels))
    else:
        if len(cue_events) != len(mat_labels):
            raise RuntimeError(f"target label count {len(mat_labels)} does not match cue count {len(cue_events)}")
        labeled = list(zip(cue_events, mat_labels.tolist()))
    signals: list[np.ndarray] = []
    labels: list[int] = []
    manifest: list[dict[str, Any]] = []
    data = raw.get_data().astype(np.float32)
    for original_index, ((sample, code), label) in enumerate(labeled):
        if label not in (LEFT, RIGHT):
            continue
        stop = sample + 1000
        if stop > data.shape[1]:
            raise RuntimeError(f"truncated 4 s epoch in {gdf_path.name} at {sample}")
        signals.append(data[:, sample:stop])
        labels.append(label - 1)
        manifest.append(
            {
                "session": session,
                "gdf": gdf_path.name,
                "original_trial_index": original_index,
                "cue_sample": sample,
                "run_id": _run_at(sample, run_starts),
                "event_code": code,
                "class_original": int(label),
                # A target manifest never exposes labels to an online runner.
                "source_label": int(label - 1) if session == "S1" else "WITHHELD",
            }
        )
    return np.stack(signals), np.asarray(labels, dtype=np.int64), manifest, channel_names


def build_or_load_stage0_cache(gdf_dir: Path, labels_dir: Path, cache_dir: Path) -> dict[str, Any]:
    signal_path = cache_dir / "signals.npy"
    source_label_path = cache_dir / "source_labels.npy"
    scoring_label_path = cache_dir / "target_labels_scoring_only.npy"
    manifest_path = cache_dir / "manifest.csv"
    if signal_path.exists() and source_label_path.exists() and scoring_label_path.exists() and manifest_path.exists():
        signals = np.load(signal_path, mmap_mode="r")
        source_labels = np.load(source_label_path, mmap_mode="r")
        rows = list(csv.DictReader(manifest_path.open(newline="")))
        return {"signals": signals, "source_labels": source_labels, "target_label_path": scoring_label_path, "manifest": rows, "cache_dir": cache_dir}
    cache_dir.mkdir(parents=True, exist_ok=True)
    s1_x, s1_y, s1_rows, names = _load_session(gdf_dir / "A01T.gdf", labels_dir / "A01T.mat", "S1")
    s2_x, s2_y, s2_rows, names2 = _load_session(gdf_dir / "A01E.gdf", labels_dir / "A01E.mat", "S2")
    if names != names2:
        raise RuntimeError("channel order differs between S1 and S2")
    signals = np.concatenate([s1_x, s2_x]).astype(np.float32)
    rows = s1_rows + s2_rows
    for index, row in enumerate(rows):
        row["cache_index"] = index
        row["channel_order"] = "|".join(names)
    if signals.shape != (288, 22, 1000):
        raise RuntimeError(f"bad Stage0 cache shape {signals.shape}; expected (288,22,1000)")
    for session, session_y in (("S1", s1_y), ("S2", s2_y)):
        if len(session_y) != 144 or np.bincount(session_y, minlength=2).tolist() != [72, 72]:
            raise RuntimeError(f"{session} left/right counts are invalid")
    np.save(signal_path, signals)
    np.save(source_label_path, s1_y)
    np.save(scoring_label_path, s2_y)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "shape": list(signals.shape),
        "sample_rate": 250,
        "channel_count": 22,
        "channel_order": names,
        "session_counts": {"S1": int(len(s1_y)), "S2": int(len(s2_y))},
        "class_counts": {"S1": np.bincount(s1_y, minlength=2).tolist(), "S2": np.bincount(s2_y, minlength=2).tolist()},
        "filter": "MNE FIR 1-48 Hz before 0-4 s epoch extraction",
        "signal_sha256": sha256(signal_path),
        "source_label_sha256": sha256(source_label_path),
        "target_labels_scoring_only_sha256": sha256(scoring_label_path),
        "target_labels_access_policy": "not opened until every model checkpoint is frozen",
        "gdf": {name: sha256(gdf_dir / name) for name in ("A01T.gdf", "A01E.gdf")},
        "official_labels": {name: sha256(labels_dir / name) for name in ("A01T.mat", "A01E.mat")},
    }
    (cache_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2))
    return {"signals": signals, "source_labels": s1_y, "target_label_path": scoring_label_path, "manifest": rows, "cache_dir": cache_dir}


def session_indices(manifest: list[dict[str, Any]], session: str) -> np.ndarray:
    return np.asarray([int(row["cache_index"]) for row in manifest if row["session"] == session], dtype=int)


def source_train_validation_indices(manifest: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    s1_rows = [row for row in manifest if row["session"] == "S1"]
    groups: dict[int, list[int]] = {}
    for row in s1_rows:
        groups.setdefault(int(row["run_id"]), []).append(int(row["cache_index"]))
    usable = [(group, idxs) for group, idxs in sorted(groups.items()) if len(idxs) >= 8]
    if len(usable) >= 2:
        val_group, val_indices = usable[-1]
        train_indices = [index for group, indices in usable[:-1] for index in indices]
        if len(train_indices) >= 16:
            return np.asarray(train_indices), np.asarray(val_indices), {"type": "run-level", "validation_run": val_group, "groups": {str(k): len(v) for k, v in groups.items()}}
    # The fallback is explicit rather than silently claiming a run-level split.
    ordered = np.asarray([int(row["cache_index"]) for row in s1_rows])
    cut = int(round(0.8 * len(ordered)))
    return ordered[:cut], ordered[cut:], {"type": "chronological-fallback", "groups": {str(k): len(v) for k, v in groups.items()}}


def probabilities_from_logits(logits: Tensor) -> np.ndarray:
    return logits.softmax(dim=-1).detach().cpu().numpy()


class SincSourceRunner:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model, self.device = model.eval(), device

    def reset(self) -> None:
        return None

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x[None, None]).to(self.device)
        return probabilities_from_logits(self.model(tensor).mean(dim=-1))[0]

    def update(self) -> None:
        return None


class BTTADGRunner:
    """Thin evaluator around the official source classes; no algorithm rewrite."""

    def __init__(self, model: nn.Module, device: torch.device, official_dir: Path):
        if str(official_dir) not in sys.path:
            sys.path.insert(0, str(official_dir))
        official = importlib.import_module("BTTA_DG")
        self.EEGModel = official.EEGModel
        self.estimator = official.DirichletEstimator()
        self.clusterer = official.OnlineClustererGMM(
            num_components=6, num_classes=2, conf_threshold=0.596, entropy_threshold=0.673
        )
        self.model, self.device = model.eval(), device
        self._pending: tuple[np.ndarray, int, np.ndarray] | None = None

    def reset(self) -> None:
        official = importlib.import_module("BTTA_DG")
        self.estimator = official.DirichletEstimator()
        self.clusterer = official.OnlineClustererGMM(
            num_components=6, num_classes=2, conf_threshold=0.596, entropy_threshold=0.673
        )
        self._pending = None

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x[None, None]).to(self.device)
        temporal, logits = self.EEGModel(self.model)(tensor)
        prob = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        feature = torch.softmax(temporal, dim=1).cpu().numpy().squeeze()
        pred = int(np.argmax(prob))
        alpha = self.estimator.estimate(feature)
        # This order exactly matches official run_online_adaptation().  It notably
        # does not call add_sample(); provenance records that source behavior.
        self._pending = (alpha, pred, prob)
        return prob

    def update(self) -> None:
        assert self._pending is not None
        alpha, pred, prob = self._pending
        self.clusterer.update(alpha, pred, prob)
        self._pending = None


class NeuroLoopRunner:
    def __init__(self, model: NeuroLoop, device: torch.device):
        self.model, self.device = model.eval(), device
        self.state = model.initial_state(1, device)
        self._final_h: Tensor | None = None
        self._probabilities: Tensor | None = None
        self.loop_predictions: list[int] = []

    def reset(self) -> None:
        self.state = self.model.initial_state(1, self.device)
        self._final_h = None
        self._probabilities = None
        self.loop_predictions = []

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, final_h, per_loop_logits = self.model.forward_with_state(torch.from_numpy(x[None]).to(self.device), self.state, return_intermediates=True)
        self._final_h = final_h
        self._probabilities = logits.softmax(dim=-1)
        self.loop_predictions = [int(value.argmax(dim=-1).item()) for value in per_loop_logits]
        return self._probabilities.detach().cpu().numpy()[0]

    def update(self) -> None:
        assert self._final_h is not None and self._probabilities is not None
        self.state = self.model.update_state(self._final_h, self.state, self._probabilities)
        self._final_h = None
        self._probabilities = None


def run_chronological(runner: Any, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Batch-1 predict-then-update with no labels admitted to the runner API."""
    runner.reset()
    probabilities: list[np.ndarray] = []
    loop_predictions: list[list[int]] = []
    for trial in x:
        probability = runner.predict(trial)
        probabilities.append(probability)
        loop_predictions.append(list(getattr(runner, "loop_predictions", [])))
        runner.update()
    probabilities_array = np.asarray(probabilities)
    return probabilities_array.argmax(axis=1), probabilities_array, loop_predictions


def ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidences, predictions = probabilities.max(axis=1), probabilities.argmax(axis=1)
    value = 0.0
    for lower in np.linspace(0.0, 0.9, bins):
        mask = (confidences >= lower) & (confidences < lower + 0.1 if lower < 0.9 else confidences <= 1.0)
        if mask.any():
            value += mask.mean() * abs((predictions[mask] == labels[mask]).mean() - confidences[mask].mean())
    return float(value)


def metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    safe = np.clip(probabilities[np.arange(len(labels)), labels], 1e-8, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "nll": float(-np.log(safe).mean()),
        "ece": ece(probabilities, labels),
    }


def train_sinc_source(
    official_dir: Path, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    device: torch.device, epochs: int, seed: int,
) -> tuple[nn.Module, list[dict[str, float]]]:
    if str(official_dir) not in sys.path:
        sys.path.insert(0, str(official_dir))
    source = importlib.import_module("SincAdaptNet")
    seed_everything(seed)
    model = source.SincAdaptNet(22, 2, 16, 24, 51, fs=250).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train[:, None]), torch.from_numpy(y_train)), batch_size=16, shuffle=True)
    best_state, best_score, history = None, -np.inf, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x).permute(0, 2, 1).reshape(-1, 2)
            loss = criterion(output, batch_y.repeat_interleave(batch_x.shape[-1]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            scores = model(torch.from_numpy(x_val[:, None]).to(device)).mean(dim=-1).argmax(dim=1).cpu().numpy()
        score = float(balanced_accuracy_score(y_val, scores))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "source_validation_ba": score})
        if score > best_score:
            best_score, best_state = score, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval(), history


def refit_sinc_source(official_dir: Path, x: np.ndarray, y: np.ndarray, device: torch.device, epochs: int, seed: int) -> nn.Module:
    """Refit after source-only selection using every permitted S1 trial."""
    if str(official_dir) not in sys.path:
        sys.path.insert(0, str(official_dir))
    source = importlib.import_module("SincAdaptNet")
    seed_everything(seed)
    model = source.SincAdaptNet(22, 2, 16, 24, 51, fs=250).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None]), torch.from_numpy(y)), batch_size=16, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(model(batch_x).permute(0, 2, 1).reshape(-1, 2), batch_y.repeat_interleave(batch_x.shape[-1]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval()


def _evaluate_neuroloop_sequence(model: NeuroLoop, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    """Validation follows source chronology and resets only at session start."""
    model.eval()
    state = model.initial_state(1, device)
    predictions: list[int] = []
    with torch.no_grad():
        for trial in x:
            logits, final_h = model.forward_with_state(torch.from_numpy(trial[None]).to(device), state)
            probabilities = logits.softmax(dim=-1)
            predictions.append(int(probabilities.argmax(dim=-1).item()))
            state = model.update_state(final_h, state, probabilities)
    return float(balanced_accuracy_score(y, predictions))


def train_neuroloop(
    spec: NeuroLoopSpec, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
    device: torch.device, epochs: int, seed: int,
) -> tuple[NeuroLoop, list[dict[str, float]]]:
    seed_everything(seed)
    model = NeuroLoop(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_score, history = None, -np.inf, []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        # Source training is explicit TBPTT over chronological chunks, never an
        # IID shuffled trial loader. Labels enter only this CE calculation.
        state = model.initial_state(1, device)
        for start in range(0, len(x_train), 16):
            stop = min(start + 16, len(x_train))
            loss = torch.zeros((), device=device)
            count = 0
            for trial, label in zip(x_train[start:stop], y_train[start:stop]):
                logits, final_h = model.forward_with_state(torch.from_numpy(trial[None]).to(device), state)
                probabilities = logits.softmax(dim=-1)
                loss = loss + criterion(logits, torch.tensor([int(label)], device=device))
                state = model.update_state(final_h, state, probabilities)
                count += 1
            loss = loss / count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            state = state.detach()
        score = _evaluate_neuroloop_sequence(model, x_val, y_val, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "source_validation_ba": score})
        if score > best_score:
            best_score, best_state = score, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval(), history


def refit_neuroloop(spec: NeuroLoopSpec, x: np.ndarray, y: np.ndarray, device: torch.device, epochs: int, seed: int) -> NeuroLoop:
    """Refit chronology-aware NeuroLoop on all allowed S1 trials after selection."""
    seed_everything(seed)
    model = NeuroLoop(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        state = model.initial_state(1, device)
        for start in range(0, len(x), 16):
            stop = min(start + 16, len(x))
            loss = torch.zeros((), device=device)
            count = 0
            for trial, label in zip(x[start:stop], y[start:stop]):
                logits, final_h = model.forward_with_state(torch.from_numpy(trial[None]).to(device), state)
                probabilities = logits.softmax(dim=-1)
                loss = loss + criterion(logits, torch.tensor([int(label)], device=device))
                state = model.update_state(final_h, state, probabilities)
                count += 1
            optimizer.zero_grad(set_to_none=True)
            (loss / count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            state = state.detach()
    return model.eval()


def model_audit(spec: NeuroLoopSpec, model: NeuroLoop) -> dict[str, Any]:
    total = model.parameter_count()
    if total >= 1_000_000:
        raise RuntimeError(f"NeuroLoop has {total} trainable parameters, violating <1M budget")
    return {
        "spec": asdict(spec),
        "trainable_parameters": total,
        "frequency_stem": f"learnable Sinc band-pass B={spec.bands}",
        "attention": "temporal attention then cross-frequency attention, followed by state cross-attention",
        "update_loop": "untied" if spec.untied else "shared",
        "stateful": spec.stateful,
    }


def causal_audit(factory: Callable[[], Any], x: np.ndarray) -> dict[str, Any]:
    cut = min(6, len(x) - 1)
    baseline_pred, baseline_prob, _ = run_chronological(factory(), x)
    changed_future = x.copy()
    changed_future[cut + 1 :] *= -3.0
    future_pred, future_prob, _ = run_chronological(factory(), changed_future)
    # The runner signature has no labels argument; an independent re-run verifies
    # determinism without opening target scoring labels.
    repeated_pred, repeated_prob, _ = run_chronological(factory(), x)
    return {
        "prefix_length": cut + 1,
        "future_signal_does_not_change_prefix": bool(np.array_equal(baseline_pred[: cut + 1], future_pred[: cut + 1]) and np.allclose(baseline_prob[: cut + 1], future_prob[: cut + 1])),
        "runner_api_has_no_target_label_parameter": True,
        "repeated_stream_is_deterministic": bool(np.array_equal(baseline_pred, repeated_pred) and np.allclose(baseline_prob, repeated_prob)),
    }

