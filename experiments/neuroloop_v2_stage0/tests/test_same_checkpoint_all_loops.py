import io

import torch

from models.neuroloop_v2 import NeuroLoopV2Spec, SharedNeuroLoopV2


def test_all_loop_outputs_come_from_one_checkpoint():
    torch.manual_seed(0)
    spec = NeuroLoopV2Spec(dropout=0.)
    model = SharedNeuroLoopV2(spec).eval()
    x = torch.randn(3, 22, 1000)
    original = model(x)
    saved = io.BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = SharedNeuroLoopV2(spec).eval()
    restored.load_state_dict(torch.load(saved, weights_only=True))
    reproduced = restored(x)
    assert len(original) == len(reproduced) == 4
    assert all(torch.equal(first, second) for first, second in zip(original, reproduced))


if __name__ == "__main__":
    test_all_loop_outputs_come_from_one_checkpoint()
    print("PASS test_all_loop_outputs_come_from_one_checkpoint")
