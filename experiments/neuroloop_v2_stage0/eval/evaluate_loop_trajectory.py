"""Label-free forward pass plus post-freeze scoring for NeuroLoop v2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from models.neuroloop_v2 import _BaseNeuroLoopV2


@torch.no_grad()
def predict_loop_trajectory(model: _BaseNeuroLoopV2, x: np.ndarray, device: torch.device) -> dict[str, Any]:
    """Run only signals through the frozen checkpoint; labels are not accepted."""
    model.eval()
    loop_probabilities: list[list[np.ndarray]] = [[] for _ in range(model.spec.max_loops)]
    feature_norms: list[list[float]] = [[] for _ in range(model.spec.max_loops)]
    cosine_changes: list[list[float]] = [[] for _ in range(model.spec.max_loops - 1)]
    logits_changes: list[list[float]] = [[] for _ in range(model.spec.max_loops - 1)]
    for trial in x:
        logits, features = model(torch.from_numpy(trial[None]).to(device), return_features=True)
        probabilities = [value.softmax(dim=-1).cpu().numpy()[0] for value in logits]
        for index, probability in enumerate(probabilities):
            loop_probabilities[index].append(probability)
            feature_norms[index].append(float(features[index].flatten(1).norm(dim=1).item()))
        for index in range(model.spec.max_loops - 1):
            first = features[index].flatten(1)
            second = features[index + 1].flatten(1)
            cosine_changes[index].append(float(torch.nn.functional.cosine_similarity(first, second).item()))
            logits_changes[index].append(float((logits[index + 1] - logits[index]).norm(dim=1).item()))
    return {
        "probabilities": [np.asarray(items) for items in loop_probabilities],
        "feature_norms": np.asarray(feature_norms).T,
        "cosine_changes": np.asarray(cosine_changes).T,
        "logits_changes": np.asarray(logits_changes).T,
    }
