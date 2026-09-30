#!/usr/bin/env python3
"""Pre-registered A01-only, seed-0, state-free NeuroLoop Stage-0 v2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval.evaluate_loop_trajectory import predict_loop_trajectory
from eval.loop_transition_metrics import confidence_group_metrics, entropy, loop_metrics, transition_counts
from models.neuroloop_v2 import NeuroLoopV2Spec, SharedNeuroLoopV2
from models.untied_control import UntiedDepthControl
from train.train_neuroloop_v2 import DEEP_SUPERVISION_WEIGHTS, seed_everything, select_then_refit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inverse_sqrt(matrix: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    return (vectors * np.power(np.maximum(values, epsilon), -0.5)) @ vectors.T


def covariance(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    return centered @ centered.T / max(1, x.shape[1] - 1)


def source_ea(x: np.ndarray) -> np.ndarray:
    transform = inverse_sqrt(np.mean([covariance(trial) for trial in x], axis=0))
    return np.einsum("ij,njt->nit", transform, x).astype(np.float32)


def target_ea(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "paper":
        transform = inverse_sqrt(np.mean([covariance(trial) for trial in x], axis=0))
        return np.einsum("ij,njt->nit", transform, x).astype(np.float32)
    if mode != "strict":
        raise ValueError(mode)
    running = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    out = np.empty_like(x, dtype=np.float32)
    for index, trial in enumerate(x):
        running += covariance(trial)
        out[index] = inverse_sqrt(running / (index + 1)) @ trial
    return out


def source_split(manifest: list[dict]) -> tuple[np.ndarray, np.ndarray, dict]:
    source = [row for row in manifest if row["session"] == "S1"]
    groups: dict[int, list[int]] = {}
    for row in source:
        groups.setdefault(int(row["run_id"]), []).append(int(row["cache_index"]))
    usable = [(group, indices) for group, indices in sorted(groups.items()) if len(indices) >= 8]
    if len(usable) < 2:
        raise RuntimeError("A01 source cache has no valid run-level split")
    validation_group, validation = usable[-1]
    training = [index for _, indices in usable[:-1] for index in indices]
    if len(training) != 120 or len(validation) != 24:
        raise RuntimeError(f"unexpected A01 source split: {len(training)} train, {len(validation)} validation")
    return np.asarray(training), np.asarray(validation), {"type": "run-level", "validation_run": validation_group, "groups": {str(key): len(value) for key, value in groups.items()}}


def source_sinc_select_refit(official_dir: Path, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, x_all: np.ndarray, y_all: np.ndarray, device: torch.device, seed: int, max_epochs: int) -> tuple[nn.Module, list[dict], int]:
    if str(official_dir) not in sys.path:
        sys.path.insert(0, str(official_dir))
    source = importlib.import_module("SincAdaptNet")

    def make() -> nn.Module:
        return source.SincAdaptNet(22, 2, 16, 24, 51, fs=250).to(device)

    def fit(model: nn.Module, x: np.ndarray, y: np.ndarray, epochs: int, validation: tuple[np.ndarray, np.ndarray] | None):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        history: list[dict] = []
        best, best_epoch = -np.inf, 1
        for epoch in range(1, epochs + 1):
            loader = DataLoader(TensorDataset(torch.from_numpy(x[:, None]), torch.from_numpy(y)), batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed + epoch))
            model.train()
            losses = []
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                temporal = model(batch_x).permute(0, 2, 1).reshape(-1, 2)
                loss = criterion(temporal, batch_y.repeat_interleave(batch_x.shape[-1]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            score = float("nan")
            if validation is not None:
                model.eval()
                with torch.no_grad():
                    prediction = model(torch.from_numpy(validation[0][:, None]).to(device)).mean(dim=-1).argmax(dim=1).cpu().numpy()
                score = float(balanced_accuracy_score(validation[1], prediction))
                if score > best:
                    best, best_epoch = score, epoch
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "source_validation_ba": score})
        return history, best_epoch

    seed_everything(seed)
    selection = make()
    history, selected_epoch = fit(selection, x_train, y_train, max_epochs, (x_val, y_val))
    seed_everything(seed)
    final = make()
    fit(final, x_all, y_all, selected_epoch, None)
    return final.eval(), history, selected_epoch


@torch.no_grad()
def sinc_probabilities(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    output = []
    for trial in x:
        logits = model(torch.from_numpy(trial[None, None]).to(device)).mean(dim=-1)
        output.append(logits.softmax(dim=-1).cpu().numpy()[0])
    return np.asarray(output)


def stem_diagnostic(stem, device: torch.device) -> dict:
    sample_rate, seconds = 250, 4
    time = torch.arange(sample_rate * seconds, device=device, dtype=torch.float32) / sample_rate
    def sine(amplitude: float, phase: float) -> torch.Tensor:
        signal = amplitude * torch.sin(2 * torch.pi * 10.0 * time + phase)
        return signal[None, None, :].repeat(1, stem.channels, 1)
    stem.eval()
    with torch.no_grad():
        power_a = stem.power_patches(sine(1.0, 0.0)).mean(dim=(0, 2, 3)).cpu().numpy()
        power_phase = stem.power_patches(sine(1.0, torch.pi / 2)).mean(dim=(0, 2, 3)).cpu().numpy()
        power_double = stem.power_patches(sine(2.0, 0.0)).mean(dim=(0, 2, 3)).cpu().numpy()
        tokens_a = stem(sine(1.0, 0.0))
        tokens_phase = stem(sine(1.0, torch.pi / 2))
        tokens_double = stem(sine(2.0, 0.0))
        signed = stem.signed_mean_patches(sine(1.0, 0.0)).abs().mean().item()
    low, high = stem.band_edges_hz()
    centers = ((low + high) / 2).detach().cpu().numpy()
    selected = int(np.argmin(np.abs(centers - 10.0)))
    phase_relative_error = float(abs(power_a[selected] - power_phase[selected]) / max(power_a[selected], 1e-8))
    amplitude_ratio = float(power_double[selected] / max(power_a[selected], 1e-8))
    token_phase_relative_error = float((tokens_a - tokens_phase).abs().mean().cpu() / tokens_a.abs().mean().clamp_min(1e-8).cpu())
    token_amplitude_change = float((tokens_double - tokens_a).abs().mean().cpu())
    return {
        "ten_hz_nearest_band": selected,
        "ten_hz_band_edges_hz": [float(low[selected].detach().cpu()), float(high[selected].detach().cpu())],
        "ten_hz_power": float(power_a[selected]),
        "ten_hz_phase_relative_error": phase_relative_error,
        "logpower_token_phase_relative_error": token_phase_relative_error,
        "amplitude_2x_power_ratio": amplitude_ratio,
        "logpower_token_2x_amplitude_mean_change": token_amplitude_change,
        "signed_waveform_mean_abs": float(signed),
        "checks": {
            "positive_power": bool(power_a[selected] > 0),
            "phase_stable": bool(phase_relative_error < 0.10),
            "logpower_token_phase_stable": bool(token_phase_relative_error < 0.10),
            "approximately_quadratic_amplitude": bool(3.2 <= amplitude_ratio <= 4.8),
            "signed_pooling_smaller_than_power": bool(signed < power_a[selected]),
        },
    }


def trajectory_rows(probabilities: list[np.ndarray], labels: np.ndarray, subject: str, session: str) -> list[dict]:
    predictions = [value.argmax(axis=1) for value in probabilities]
    entropies = [entropy(value) for value in probabilities]
    rows: list[dict] = []
    for index in range(len(labels)):
        correct_first, correct_last = predictions[0][index] == labels[index], predictions[3][index] == labels[index]
        transition = "CC" if correct_first and correct_last else "CW" if correct_first else "WC" if correct_last else "WW"
        row = {"subject": subject, "session": session, "trial_index": index, "true_label": int(labels[index]), "transition_1_4": transition}
        for loop in range(4):
            row[f"pred_{loop + 1}"] = int(predictions[loop][index])
            row[f"prob0_{loop + 1}"] = float(probabilities[loop][index, 0])
            row[f"prob1_{loop + 1}"] = float(probabilities[loop][index, 1])
            row[f"entropy_{loop + 1}"] = float(entropies[loop][index])
            row[f"correct_{loop + 1}"] = bool(predictions[loop][index] == labels[index])
        rows.append(row)
    return rows


def build_report(output: Path, metrics_rows: list[dict], transitions: list[dict], confidence_rows: list[dict], stem: dict, selected: dict, strict_shared: list[np.ndarray], strict_untied: list[np.ndarray], labels: np.ndarray) -> None:
    shared = [row for row in metrics_rows if row["model"] == "neuroloop_v2_shared" and row["ea_mode"] == "strict"]
    shared.sort(key=lambda row: row["loop"])
    untied = [row for row in metrics_rows if row["model"] == "neuroloop_v2_untied" and row["ea_mode"] == "strict" and row["loop"] == 4][0]
    transition14 = [row for row in transitions if row["model"] == "neuroloop_v2_shared" and row["from_loop"] == 1 and row["to_loop"] == 4][0]
    source_sinc = [row for row in metrics_rows if row["model"] == "source_only_sinc" and row["ea_mode"] == "strict"][0]
    confidence = {row["confidence_group"]: row for row in confidence_rows if row["model"] == "neuroloop_v2_shared" and row["ea_mode"] == "strict"}
    accuracy1, accuracy4 = shared[0]["accuracy"], shared[-1]["accuracy"]
    success = accuracy4 > accuracy1 and transition14["correction_rate"] > transition14["corruption_rate"]
    low = confidence["low"]
    if low["n_trials"] and low["correction_rate"] > low["corruption_rate"] and low["accuracy_gain"] > 0:
        ambiguous_conclusion = "The low-confidence group improves: the extra loops are consistent with correcting ambiguous trials."
    else:
        ambiguous_conclusion = "The low-confidence group does not improve: the extra loops are not correcting ambiguous EEG and instead show overthinking/corruption."
    lines = [
        "# A01 NeuroLoop Stage-0 v2 report",
        "",
        "This report is A01 / S1->S2 / seed 0 only. It is a falsification gate, not a multi-subject result.",
        "",
        "## Strict-causal same-checkpoint trajectory",
        "",
        "| loop | Accuracy | BA | Macro-F1 | NLL | ECE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {row['loop']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['nll']:.4f} | {row['ece']:.4f} |" for row in shared]
    lines += [
        "",
        "## Direct answers",
        "",
        f"1. **Log-power stem / v1 backbone weakness:** synthetic checks passed = `{all(stem['checks'].values())}`. Strict shared loop-4 accuracy is `{accuracy4:.4f}` versus source-only Sinc `{source_sinc['accuracy']:.4f}`. This {'does' if accuracy4 >= source_sinc['accuracy'] else 'does not'} repair the practical backbone deficit relative to Sinc on A01.",
        f"2. **Same checkpoint Acc@1 -> Acc@4:** `{accuracy1:.4f}` -> `{accuracy4:.4f}`; increase = `{accuracy4 - accuracy1:+.4f}`.",
        f"3. **Correction vs corruption (1->4):** `{transition14['correction_rate']:.4f}` vs `{transition14['corruption_rate']:.4f}`.",
        f"4. **Ambiguous trials:** low-confidence correction/corruption/gain = `{low['correction_rate']:.4f}` / `{low['corruption_rate']:.4f}` / `{low['accuracy_gain']:+.4f}`; medium = `{confidence['medium']['correction_rate']:.4f}` / `{confidence['medium']['corruption_rate']:.4f}` / `{confidence['medium']['accuracy_gain']:+.4f}`. {ambiguous_conclusion} Full bins are in `A01_CONFIDENCE_GROUPS.csv`.",
        f"5. **Shared vs untied K4:** shared `{accuracy4:.4f}`; untied `{untied['accuracy']:.4f}`.",
        f"6. **Decision:** `{'GO' if success else 'NO-GO'}`.",
        "",
        f"Selection epochs, source-only: Sinc={selected['sinc']}, shared={selected['shared']}, untied={selected['untied']}. Target scoring labels were opened only after all checkpoint hashes were recorded.",
    ]
    (output / "A01_STAGE0_V2_REPORT.md").write_text("\n".join(lines) + "\n")
    if not success:
        failure = [
            "# NeuroLoop v2 failure analysis",
            "",
            "**NO-GO.** The pre-registered A01 gate did not meet both required conditions: Acc@4 > Acc@1 and correction rate > corruption rate.",
            "",
            f"- Stem synthetic checks passed: `{all(stem['checks'].values())}`.",
            f"- Acc@1 -> Acc@4: `{accuracy1:.4f}` -> `{accuracy4:.4f}`.",
            f"- Correction / corruption: `{transition14['correction_rate']:.4f}` / `{transition14['corruption_rate']:.4f}`.",
            f"- Untied K4 accuracy: `{untied['accuracy']:.4f}`; shared K4: `{accuracy4:.4f}`.",
            "- Inspect `A01_LOOP_DIAGNOSTICS.csv`: cosine changes close to one indicate identity-like loops; large logit movement together with net accuracy loss indicates overthinking. `model_audit.json` stores learned gamma values.",
            "",
            "The v2 protocol forbids increasing width/depth, adding state, changing target preprocessing, or trying other datasets after this result. Do not expand to nine subjects or further seeds.",
        ]
        (output / "NEUROLOOP_V2_FAILURE_ANALYSIS.md").write_text("\n".join(failure) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--official-btta-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=40)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Stage-0 v2 is fixed to seed 0")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signals = np.load(args.cache / "signals.npy", mmap_mode="r")
    source_labels = np.load(args.cache / "source_labels.npy", mmap_mode="r")
    manifest = list(csv.DictReader((args.cache / "manifest.csv").open(newline="")))
    source_indices, validation_indices, split = source_split(manifest)
    target_indices = np.asarray([int(row["cache_index"]) for row in manifest if row["session"] == "S2"], dtype=int)
    if signals.shape != (288, 22, 1000) or len(source_labels) != 144 or len(target_indices) != 144:
        raise RuntimeError("shared A01 cache audit failed")
    source_x = source_ea(np.asarray(signals[:144]))
    source_y = np.asarray(source_labels, dtype=np.int64)
    if set(source_indices) & set(target_indices) or set(validation_indices) & set(target_indices):
        raise RuntimeError("target trials leaked into source selection")
    x_train, y_train = source_x[source_indices], source_y[source_indices]
    x_validation, y_validation = source_x[validation_indices], source_y[validation_indices]
    spec = NeuroLoopV2Spec()
    config = {
        "dataset": "BNCI2014001 / BCIC-IV-2a", "subject": "A01", "seed": 0,
        "source": "A01T/S1 left-right", "target": "A01E/S2 chronological left-right",
        "shared_preprocessing": "FIR 1-48 Hz cache, original 22 EEG channels, 0-4 seconds, 250 Hz",
        "ea_modes": ["paper", "strict"], "state": "disabled", "max_loops": 4,
        "deep_supervision_weights": DEEP_SUPERVISION_WEIGHTS, "max_epochs": args.max_epochs,
        "selection": split, "neuroloop_spec": asdict(spec),
    }
    try:
        config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        config["git_commit"] = "unavailable"
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "device": str(device)}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))

    sinc, sinc_history, sinc_epoch = source_sinc_select_refit(args.official_btta_dir, x_train, y_train, x_validation, y_validation, source_x, source_y, device, args.seed, args.max_epochs)
    shared, shared_history, shared_epoch = select_then_refit(lambda: SharedNeuroLoopV2(spec), x_train, y_train, x_validation, y_validation, source_x, source_y, device, args.seed, args.max_epochs)
    untied, untied_history, untied_epoch = select_then_refit(lambda: UntiedDepthControl(spec), x_train, y_train, x_validation, y_validation, source_x, source_y, device, args.seed, args.max_epochs)
    write_csv(args.output / "sinc_training.csv", sinc_history)
    write_csv(args.output / "shared_training.csv", shared_history)
    write_csv(args.output / "untied_training.csv", untied_history)

    # All model selection/refitting is complete. Freeze every checkpoint before
    # opening target scoring labels.
    checkpoint_rows = []
    for name, model in (("sinc_source", sinc), ("neuroloop_v2_shared", shared), ("neuroloop_v2_untied", untied)):
        path = args.output / f"{name}.pt"
        torch.save(model.state_dict(), path)
        checkpoint_rows.append({"model": name, "path": path.name, "sha256": sha256_file(path), "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)})
    write_csv(args.output / "checkpoint_audit.csv", checkpoint_rows)
    (args.output / "model_audit.json").write_text(json.dumps({
        "shared": shared.architecture_audit(),
        "untied": untied.architecture_audit(),
        "sinc_parameters": checkpoint_rows[0]["parameters"],
        "shared_gamma_values": [float(value) for value in shared.gamma.detach().cpu()],
        "untied_gamma_values": [float(value) for value in untied.gamma.detach().cpu()],
        "shared_block_parameter_ids": {name: id(value) for name, value in shared.shared_block.named_parameters()},
        "cross_trial_state": "absent",
    }, indent=2))

    target_labels = np.load(args.cache / "target_labels_scoring_only.npy")
    target_raw = np.asarray(signals[target_indices])
    metrics_rows: list[dict] = []
    transition_rows: list[dict] = []
    confidence_rows: list[dict] = []
    shared_strict = None
    untied_strict = None
    for ea_mode in ("paper", "strict"):
        target_x = target_ea(target_raw, ea_mode)
        digest = tensor_digest(target_x)
        sinc_probability = sinc_probabilities(sinc, target_x, device)
        metrics_rows.append({"model": "source_only_sinc", "ea_mode": ea_mode, "loop": 0, "input_tensor_sha256": digest, **loop_metrics(target_labels, sinc_probability)})
        for name, model in (("neuroloop_v2_shared", shared), ("neuroloop_v2_untied", untied)):
            trajectory = predict_loop_trajectory(model, target_x, device)
            probabilities = trajectory["probabilities"]
            for loop, probability in enumerate(probabilities, start=1):
                metrics_rows.append({"model": name, "ea_mode": ea_mode, "loop": loop, "input_tensor_sha256": digest, **loop_metrics(target_labels, probability)})
            for first, second in ((1, 2), (2, 3), (3, 4), (1, 4)):
                transition_rows.append({"model": name, "ea_mode": ea_mode, "from_loop": first, "to_loop": second, **transition_counts(target_labels, probabilities[first - 1], probabilities[second - 1])})
            if name == "neuroloop_v2_shared" and ea_mode == "strict":
                shared_strict = trajectory
                for row in confidence_group_metrics(target_labels, probabilities[0], probabilities[3]):
                    confidence_rows.append({"model": name, "ea_mode": ea_mode, **row})
                write_csv(args.output / "A01_LOOP_TRAJECTORY.csv", trajectory_rows(probabilities, target_labels, "A01", "S2"))
                diagnostic_rows = []
                for index in range(len(target_labels)):
                    diagnostic_rows.append({
                        "subject": "A01", "session": "S2", "trial_index": index,
                        **{f"feature_norm_{loop + 1}": float(trajectory["feature_norms"][index, loop]) for loop in range(4)},
                        **{f"cos_{loop + 1}_{loop + 2}": float(trajectory["cosine_changes"][index, loop]) for loop in range(3)},
                        **{f"logits_delta_{loop + 1}_{loop + 2}": float(trajectory["logits_changes"][index, loop]) for loop in range(3)},
                    })
                write_csv(args.output / "A01_LOOP_DIAGNOSTICS.csv", diagnostic_rows)
            if name == "neuroloop_v2_untied" and ea_mode == "strict":
                untied_strict = trajectory
    if shared_strict is None or untied_strict is None:
        raise RuntimeError("strict-causal trajectories missing")
    write_csv(args.output / "A01_METRICS.csv", metrics_rows)
    write_csv(args.output / "A01_TRANSITIONS.csv", transition_rows)
    write_csv(args.output / "A01_CONFIDENCE_GROUPS.csv", confidence_rows)
    diagnostics = stem_diagnostic(shared.stem, device)
    (args.output / "stem_diagnostic.json").write_text(json.dumps(diagnostics, indent=2))
    stem_markdown = ["# Log-power stem diagnostic", "", f"10 Hz nearest band: {diagnostics['ten_hz_nearest_band']} ({diagnostics['ten_hz_band_edges_hz'][0]:.2f}-{diagnostics['ten_hz_band_edges_hz'][1]:.2f} Hz)", f"- power phase relative error: {diagnostics['ten_hz_phase_relative_error']:.4f}", f"- token phase relative error: {diagnostics['logpower_token_phase_relative_error']:.4f}", f"- 2x amplitude power ratio: {diagnostics['amplitude_2x_power_ratio']:.4f}", f"- 2x-amplitude log-power token mean change: {diagnostics['logpower_token_2x_amplitude_mean_change']:.6f}", f"- signed-pooling mean absolute value: {diagnostics['signed_waveform_mean_abs']:.6f}", f"- checks: `{diagnostics['checks']}`"]
    (args.output / "STEM_DIAGNOSTIC.md").write_text("\n".join(stem_markdown) + "\n")
    btta_audit = """# BTTA-DG code audit\n\n## Examined source\n\nThe examined official snapshot is `/root/BTTA-DG-official` at the revision recorded by the surrounding repository metadata.  Its online route calls `clusterer.update(alpha, pred, prob_map)` but does not call `clusterer.add_sample(alpha, pred, prob_map)` before that update.\n\n## Consequence\n\nThe clustering/GMM buffer that the paper algorithm requires is not populated along this code path.  In addition, the local prior runner did not invoke a calibrated `predict_class()` posterior route.  Therefore the previous 80.56% number is only a source-only SincAdaptNet sanity result; it is not evidence for functioning BTTA-DG online adaptation.\n\n## Scope decision\n\nStage-0 v2 excludes BTTA-DG from efficacy comparisons.  It keeps Source-only SincAdaptNet as a fixed sanity baseline and evaluates only within-trial shared refinement.  A future repair must be explicitly separated into `BTTA-DG-official-code` and `BTTA-DG-paper-faithful`; silently adding the missing buffer write must not be called an official-code reproduction.\n"""
    (args.output / "BTTA_CODE_AUDIT.md").write_text(btta_audit)
    selected = {"sinc": sinc_epoch, "shared": shared_epoch, "untied": untied_epoch}
    (args.output / "selection_audit.json").write_text(json.dumps({"selected_epochs": selected, "target_labels_opened_after_checkpoint_freeze": True}, indent=2))
    build_report(args.output, metrics_rows, transition_rows, confidence_rows, diagnostics, selected, shared_strict["probabilities"], untied_strict["probabilities"], target_labels)
    print(args.output / "A01_STAGE0_V2_REPORT.md")


if __name__ == "__main__":
    main()
