#!/usr/bin/env python3
"""Run the one-subject, seed-0 BNCI2014001 online-adaptation Stage 0 gate."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from models import NeuroLoopSpec
from protocol import (
    BTTADGRunner, NeuroLoopRunner, SincSourceRunner, build_or_load_stage0_cache, causal_audit,
    metrics, model_audit, refit_neuroloop, refit_sinc_source, run_chronological, seed_everything, session_indices, sha256,
    source_ea, source_train_validation_indices, target_ea, train_neuroloop, train_sinc_source,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(model: torch.nn.Module, path: Path) -> str:
    torch.save(model.state_dict(), path)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--official-btta-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sinc-epochs", type=int, default=25)
    parser.add_argument("--neuro-epochs", type=int, default=20)
    args = parser.parse_args()
    if args.seed != 0:
        raise ValueError("Stage 0 is fixed to seed 0")
    seed_everything(args.seed)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = build_or_load_stage0_cache(args.gdf_dir, args.labels_dir, output / "shared_cache")
    # Only source labels are opened during selection/refitting. The separately
    # stored S2 scoring labels are opened after every model checkpoint is frozen.
    x, source_y, manifest = np.asarray(cache["signals"]), np.asarray(cache["source_labels"]), cache["manifest"]
    train_idx, val_idx, split_audit = source_train_validation_indices(manifest)
    target_idx = session_indices(manifest, "S2")
    if set(train_idx) & set(target_idx) or set(val_idx) & set(target_idx):
        raise RuntimeError("target samples entered source training or validation")
    x_source, source_transform = source_ea(x[session_indices(manifest, "S1")])
    source_index_to_ea = {int(index): position for position, index in enumerate(session_indices(manifest, "S1"))}
    x_train = np.asarray([x_source[source_index_to_ea[int(index)]] for index in train_idx], dtype=np.float32)
    x_val = np.asarray([x_source[source_index_to_ea[int(index)]] for index in val_idx], dtype=np.float32)
    y_train, y_val = source_y[train_idx], source_y[val_idx]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocol = {
        "dataset": "BNCI2014001 / BCI Competition IV 2a",
        "subject": "A01",
        "seed": 0,
        "source": "S1 / A01T: left vs right only",
        "target": "S2 / A01E chronological stream; labels unavailable to runners",
        "preprocessing": "original 22 EEG channels in GDF order; FIR 1-48 Hz; 0-4s at 250Hz",
        "ea_modes": ["paper", "strict"],
        "reset_policy": "reset each target session, no state leakage",
        "source_validation": split_audit,
        "official_btta_commit": "5932d026bbd8a7de106d31a6d264f4f4924537e4",
        "official_btta_files": {name: sha256(args.official_btta_dir / name) for name in ("BTTA_DG.py", "SincAdaptNet.py", "pretrain_SincAdaptNet.py")},
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    sinc_selected, sinc_history = train_sinc_source(args.official_btta_dir, x_train, y_train, x_val, y_val, device, args.sinc_epochs, args.seed)
    write_csv(output / "sinc_source_training.csv", sinc_history)
    sinc_selected_epoch = next(row["epoch"] for row in sinc_history if row["source_validation_ba"] == max(item["source_validation_ba"] for item in sinc_history))
    sinc = refit_sinc_source(args.official_btta_dir, x_source.astype(np.float32), source_y, device, sinc_selected_epoch, args.seed)
    sinc_hash = checkpoint(sinc, output / "sinc_source_best.pt")
    neural_variants = [
        ("neuroloop_static_k1", NeuroLoopSpec(steps=1, stateful=False)),
        ("neuroloop_static_k4", NeuroLoopSpec(steps=4, stateful=False)),
        ("neuroloop_shared_k1", NeuroLoopSpec(steps=1, stateful=True)),
        ("neuroloop_shared_k2", NeuroLoopSpec(steps=2, stateful=True)),
        ("neuroloop_shared_k4", NeuroLoopSpec(steps=4, stateful=True)),
        ("neuroloop_shared_k8", NeuroLoopSpec(steps=8, stateful=True)),
        ("neuroloop_untied_k4", NeuroLoopSpec(steps=4, stateful=True, untied=True)),
    ]
    trained: list[tuple[str, object, dict]] = []
    model_rows = [{"method": "source_only_sinc", "trainable_parameters": sum(p.numel() for p in sinc.parameters() if p.requires_grad), "checkpoint_sha256": sinc_hash, "architecture": "official SincAdaptNet"}]
    for offset, (name, spec) in enumerate(neural_variants):
        selected, history = train_neuroloop(spec, x_train, y_train, x_val, y_val, device, args.neuro_epochs, args.seed + offset)
        write_csv(output / f"{name}_training.csv", history)
        selected_epoch = next(row["epoch"] for row in history if row["source_validation_ba"] == max(item["source_validation_ba"] for item in history))
        model = refit_neuroloop(spec, x_source.astype(np.float32), source_y, device, selected_epoch, args.seed + offset)
        weight_hash = checkpoint(model, output / f"{name}_best.pt")
        audit = model_audit(spec, model)
        model_rows.append({"method": name, "trainable_parameters": audit["trainable_parameters"], "checkpoint_sha256": weight_hash, "architecture": json.dumps(audit)})
        trained.append((name, model, audit))
    write_csv(output / "model_audit.csv", model_rows)
    result_rows: list[dict] = []
    per_trial_rows: list[dict] = []
    audit: dict[str, object] = {}
    # Every checkpoint above is now frozen. This is the first permitted opening
    # of S2 labels; runners never receive this array.
    target_y = np.load(cache["target_label_path"])
    for ea_mode in ("paper", "strict"):
        target_x, ea_audit = target_ea(x[target_idx], ea_mode)
        runners = [
            ("source_only_sinc", lambda: SincSourceRunner(sinc, device)),
            ("btta_dg_official", lambda: BTTADGRunner(sinc, device, args.official_btta_dir)),
        ] + [(name, lambda model=model: NeuroLoopRunner(model, device)) for name, model, _ in trained]
        for method, factory in runners:
            predictions, probabilities, loop_predictions = run_chronological(factory(), target_x)
            score = metrics(target_y, predictions, probabilities)
            result_rows.append({"method": method, "ea_mode": ea_mode, **score, "n_trials": len(predictions)})
            for order, (prediction, probability, loop_prediction) in enumerate(zip(predictions, probabilities, loop_predictions)):
                per_trial_rows.append({"method": method, "ea_mode": ea_mode, "chronological_order": order, "prediction": int(prediction), "probability_0": float(probability[0]), "probability_1": float(probability[1]), "loop_predictions": json.dumps(loop_prediction)})
            if method in {"btta_dg_official", "neuroloop_shared_k4"} and ea_mode == "strict":
                audit[method] = causal_audit(factory, target_x)
        audit[f"target_ea_{ea_mode}"] = ea_audit
    write_csv(output / "stage0_metrics.csv", result_rows)
    write_csv(output / "stage0_per_trial_predictions.csv", per_trial_rows)
    audit["split"] = {"train_count": len(train_idx), "validation_count": len(val_idx), "target_count": len(target_idx), "overlap": False}
    audit["btta_semantics"] = "Official BTTA_DG.run_online_adaptation calls OnlineClustererGMM.update but never add_sample; Stage0 preserves that source order exactly."
    (output / "causal_audit.json").write_text(json.dumps(audit, indent=2))
    report = [
        "# BNCI2014001 Online Adaptation Stage 0",
        "",
        "This is a one-subject seed-0 gate only (A01 S1 -> chronological A01 S2). It is not a 9-subject benchmark result.",
        "",
        "## Integrity checks",
        "",
        f"- Shared cache: `{x.shape}` (expected 288 x 22 x 1000); S1/S2 retain 144 left/right trials each.",
        "- Target labels are separate from the runner interface and are scored only after a full stream is emitted.",
        "- Strict EA consumes x_1...x_t only; paper EA is separately labelled non-causal/full-session.",
        "- Every runner uses batch 1 and `predict()` followed by `update()`; its state resets for S2.",
        "- BTTA-DG is imported from the official source snapshot; its update order is preserved verbatim.",
        "",
        "## Metrics",
        "",
        "| method | EA | accuracy | balanced accuracy | macro-F1 | NLL | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    report += [f"| {row['method']} | {row['ea_mode']} | {row['accuracy']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['nll']:.4f} | {row['ece']:.4f} |" for row in result_rows]
    report += ["", "The published BTTA mean cannot be used as a quantitative replication check at n=1; the full 9-subject gate remains required before that comparison."]
    (output / "STAGE0_REPORT.md").write_text("\n".join(report) + "\n")
    (output / "STAGE0_COMPLETION.json").write_text(json.dumps({"completed": True, "stage": 0, "subject": "A01", "seed": 0, "all_causal_checks_pass": all(value is True for method in ("btta_dg_official", "neuroloop_shared_k4") for value in audit[method].values() if isinstance(value, bool))}, indent=2))
    print(output / "STAGE0_REPORT.md")


if __name__ == "__main__":
    main()

