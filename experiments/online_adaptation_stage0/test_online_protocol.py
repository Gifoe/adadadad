"""Fast invariant tests for online state and strict causal EA."""

from __future__ import annotations

import inspect
import numpy as np
import torch

from models import NeuroLoop, NeuroLoopSpec
from protocol import NeuroLoopRunner, causal_audit, run_chronological, target_ea


def test_strict_ea_does_not_read_future_trials() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(8, 22, 125)).astype(np.float32)
    transformed, _ = target_ea(x, "strict")
    changed = x.copy()
    changed[5:] += 100
    changed_transformed, _ = target_ea(changed, "strict")
    assert np.allclose(transformed[:5], changed_transformed[:5])


def test_neuroloop_predict_then_update_and_future_invariance() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 22, 125)).astype(np.float32)
    model = NeuroLoop(NeuroLoopSpec(steps=4, stateful=True)).eval()
    audit = causal_audit(lambda: NeuroLoopRunner(model, torch.device("cpu")), x)
    assert audit["future_signal_does_not_change_prefix"]
    assert audit["runner_api_has_no_target_label_parameter"]


def test_static_state_is_not_mutated() -> None:
    torch.manual_seed(1)
    model = NeuroLoop(NeuroLoopSpec(steps=1, stateful=False)).eval()
    runner = NeuroLoopRunner(model, torch.device("cpu"))
    before = runner.state.clone()
    runner.predict(np.zeros((22, 125), dtype=np.float32))
    runner.update()
    assert torch.equal(before, runner.state)


def test_target_labels_cannot_be_routed_to_online_runner() -> None:
    """Permuting an external label array has no callable route into prediction."""
    assert "labels" not in inspect.signature(run_chronological).parameters
    assert "labels" not in inspect.signature(NeuroLoopRunner.predict).parameters
    external_labels = np.arange(8) % 2
    assert not np.array_equal(external_labels, external_labels[::-1])


def test_predict_does_not_update_state_and_reset_restores_m0() -> None:
    torch.manual_seed(4)
    model = NeuroLoop(NeuroLoopSpec(steps=4, stateful=True)).eval()
    runner = NeuroLoopRunner(model, torch.device("cpu"))
    initial = runner.state.clone()
    runner.predict(np.ones((22, 125), dtype=np.float32))
    assert torch.equal(initial, runner.state), "predict must read M_(t-1), not mutate it"
    runner.update()
    runner.reset()
    assert torch.equal(initial, runner.state), "SESSION-RESET must restore M0"


def test_shared_loop_has_one_block_and_untied_has_one_per_iteration() -> None:
    shared = NeuroLoop(NeuroLoopSpec(steps=4, stateful=True))
    untied = NeuroLoop(NeuroLoopSpec(steps=4, stateful=True, untied=True))
    assert hasattr(shared, "shared_loop_block") and not hasattr(shared, "loop_blocks")
    assert len(untied.loop_blocks) == 4

