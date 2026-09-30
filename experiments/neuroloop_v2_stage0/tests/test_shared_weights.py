import torch

from models.neuroloop_v2 import NeuroLoopV2Spec, SharedNeuroLoopV2


def test_shared_block_and_state_free_interface():
    model = SharedNeuroLoopV2(NeuroLoopV2Spec())
    assert model.block_for(0) is model.shared_block
    assert model.block_for(3) is model.shared_block
    names = [name for name, _ in model.named_parameters()]
    assert not any("blocks.1" in name or "blocks.2" in name for name in names)
    assert model.architecture_audit()["state_free"] is True
    assert len(model(torch.randn(2, 22, 1000))) == 4


if __name__ == "__main__":
    test_shared_block_and_state_free_interface()
    print("PASS test_shared_block_and_state_free_interface")
