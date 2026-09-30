#!/usr/bin/env python3
"""Measure frozen-checkpoint online latency and prove common target input use.

This tool intentionally opens no target scoring labels. It may be run after a
completed Stage 0 without altering training, checkpoint selection, or metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from models import NeuroLoop, NeuroLoopSpec
from protocol import BTTADGRunner, NeuroLoopRunner, SincSourceRunner, session_indices, target_ea


def tensor_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def time_stream(runner, x: np.ndarray) -> tuple[list[float], list[float]]:
    runner.reset()
    predict_ms, update_ms = [], []
    for trial in x:
        start = time.perf_counter()
        runner.predict(trial)
        middle = time.perf_counter()
        runner.update()
        finish = time.perf_counter()
        predict_ms.append((middle - start) * 1000)
        update_ms.append((finish - middle) * 1000)
    return predict_ms, update_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-output", type=Path, required=True)
    parser.add_argument("--official-btta-dir", type=Path, required=True)
    args = parser.parse_args()
    if str(args.official_btta_dir) not in sys.path:
        sys.path.insert(0, str(args.official_btta_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = args.stage_output / "shared_cache"
    x = np.load(cache / "signals.npy", mmap_mode="r")
    manifest = list(csv.DictReader((cache / "manifest.csv").open(newline="")))
    target_x, _ = target_ea(np.asarray(x[session_indices(manifest, "S2")]), "strict")
    target_digest = tensor_hash(target_x)
    source = importlib.import_module("SincAdaptNet")
    sinc = source.SincAdaptNet(22, 2, 16, 24, 51, fs=250).to(device)
    sinc.load_state_dict(torch.load(args.stage_output / "sinc_source_best.pt", map_location=device, weights_only=True))
    variants = [("source_only_sinc", lambda: SincSourceRunner(sinc, device)), ("btta_dg_official", lambda: BTTADGRunner(sinc, device, args.official_btta_dir))]
    for row in csv.DictReader((args.stage_output / "model_audit.csv").open(newline="")):
        if not row["method"].startswith("neuroloop_"):
            continue
        spec = NeuroLoopSpec(**json.loads(row["architecture"])["spec"])
        model = NeuroLoop(spec).to(device)
        model.load_state_dict(torch.load(args.stage_output / f"{row['method']}_best.pt", map_location=device, weights_only=True))
        variants.append((row["method"], lambda model=model: NeuroLoopRunner(model, device)))
    rows = []
    for name, factory in variants:
        # Every factory is handed the exact same ndarray object, and its digest
        # is recorded for an external audit of the common-input assertion.
        predict_ms, update_ms = time_stream(factory(), target_x)
        rows.append({
            "method": name,
            "mode": "strict",
            "n_trials": len(target_x),
            "target_tensor_sha256": target_digest,
            "predict_median_ms": float(np.median(predict_ms)),
            "predict_p95_ms": float(np.percentile(predict_ms, 95)),
            "update_median_ms": float(np.median(update_ms)),
            "update_p95_ms": float(np.percentile(update_ms, 95)),
        })
    with (args.stage_output / "online_latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "strict_target_shape": list(target_x.shape),
        "all_methods_share_one_target_tensor_digest": len({row["target_tensor_sha256"] for row in rows}) == 1,
        "target_tensor_sha256": target_digest,
        "labels_opened": False,
        "device": str(device),
        "methods": [row["method"] for row in rows],
    }
    (args.stage_output / "input_consistency_audit.json").write_text(json.dumps(audit, indent=2))
    print(args.stage_output / "online_latency.csv")


if __name__ == "__main__":
    main()
