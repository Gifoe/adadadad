"""Source-only selection and refit for the state-free NeuroLoop v2 models."""

from __future__ import annotations

import copy
import random
from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.neuroloop_v2 import _BaseNeuroLoopV2


DEEP_SUPERVISION_WEIGHTS = (0.125, 0.125, 0.25, 0.5)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(x: np.ndarray, y: np.ndarray, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=16, shuffle=True, generator=generator)


def deep_supervision_loss(logits: list[torch.Tensor], labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    if len(logits) != len(DEEP_SUPERVISION_WEIGHTS):
        raise RuntimeError("v2 requires exactly four loop outputs")
    return sum(weight * criterion(output, labels) for weight, output in zip(DEEP_SUPERVISION_WEIGHTS, logits))


@torch.no_grad()
def validation_balanced_accuracy(model: _BaseNeuroLoopV2, x: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    logits = model(torch.from_numpy(x).to(device))[-1]
    return float(balanced_accuracy_score(y, logits.argmax(dim=1).cpu().numpy()))


def _run_epochs(model: _BaseNeuroLoopV2, x: np.ndarray, y: np.ndarray, epochs: int, device: torch.device, seed: int, validation: tuple[np.ndarray, np.ndarray] | None = None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    best_state, best_score = None, -np.inf
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_x, batch_y in _loader(x, y, seed + epoch):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = deep_supervision_loss(model(batch_x), batch_y, criterion)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = float("nan")
        if validation is not None:
            score = validation_balanced_accuracy(model, validation[0], validation[1], device)
            # Strictly earliest tie: only a strictly better score replaces it.
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "source_validation_ba": score})
    return history, best_state, best_score


def select_then_refit(
    factory: Callable[[], _BaseNeuroLoopV2],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_all_source: np.ndarray,
    y_all_source: np.ndarray,
    device: torch.device,
    seed: int = 0,
    max_epochs: int = 40,
) -> tuple[_BaseNeuroLoopV2, list[dict[str, float | int]], int]:
    """Choose an epoch on S1 validation, then refit that epoch count on all S1."""
    seed_everything(seed)
    selected_model = factory().to(device)
    history, best_state, _ = _run_epochs(selected_model, x_train, y_train, max_epochs, device, seed, (x_validation, y_validation))
    if best_state is None:
        raise RuntimeError("selection did not create a checkpoint")
    selected_epoch = next(int(row["epoch"]) for row in history if row["source_validation_ba"] == max(item["source_validation_ba"] for item in history))
    # Reinitialise at the same global seed; no target labels are loaded here.
    seed_everything(seed)
    final_model = factory().to(device)
    _run_epochs(final_model, x_all_source, y_all_source, selected_epoch, device, seed, validation=None)
    return final_model.eval(), history, selected_epoch
