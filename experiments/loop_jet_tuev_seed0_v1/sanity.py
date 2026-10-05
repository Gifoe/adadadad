"""CUDA BF16 and scientific invariants; synthetic tensors, never a trained result."""
import ctypes
import json
import pathlib
import sys
import time
from types import SimpleNamespace

from process_affinity import configure_process_affinity
AFFINITY = configure_process_affinity()
import torch
from loop_model import LoopDenoiser, LoopRawViTDiffusion, RawViTDiffusion

ROOT = pathlib.Path(__file__).resolve().parent

def main():
    (ROOT/'analysis').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    device = torch.device('cuda')
    report = {'torch': torch.__version__, 'cuda': torch.version.cuda,
              'python': sys.version, 'device': torch.cuda.get_device_name(),
              'capability': list(torch.cuda.get_device_capability()), 'arch_list': torch.cuda.get_arch_list(),
              'process_affinity': AFFINITY, 'synthetic_only': True}
    args = SimpleNamespace(**json.loads((ROOT/'configs'/'naive_k2.json').read_text())['official_args'])
    origin = RawViTDiffusion(model_name=args.model, num_channels=16, patch_size=200, target_length=1000, num_classes=6).to(device).eval()
    loop = LoopRawViTDiffusion(model_name=args.model, num_channels=16, patch_size=200, target_length=1000, num_classes=6, loop_count=1).to(device).eval()
    loop.load_state_dict(origin.state_dict(), strict=True)
    x = torch.randn(2, 16, 5, 200, device=device)
    t = torch.tensor([.2, .7], device=device)
    y = torch.tensor([0, 5], device=device)
    with torch.no_grad():
        a, b = origin(x, t, y), loop(x, t, y)
        error = float((a-b).abs().max())
        assert error < 1e-6
        with torch.autocast('cuda', dtype=torch.bfloat16):
            a, b = origin(x, t, y), loop(x, t, y)
        bf_error = float((a-b).abs().max())
        assert bf_error == 0
    report.update(k1_max_abs_float32=error, k1_max_abs_bfloat16=bf_error,
                  official_parameter_count=sum(p.numel() for p in origin.parameters()),
                  official_trainable_parameter_count=sum(p.numel() for p in origin.parameters() if p.requires_grad))
    loop.loop_count = 2
    counts = [0]*12
    hooks = []
    for i, block in enumerate(loop.blocks):
        def mark(module, inputs, output, index=i):
            counts[index] += 1
        hooks.append(block.register_forward_hook(mark))
    with torch.no_grad():
        p1, p2 = loop.forward_endpoints(x, t, y)
    for hook in hooks:
        hook.remove()
    assert counts == [1]*4+[2]*4+[2]*4
    report.update(k2_parameter_count=sum(p.numel() for p in loop.parameters()),
                  shared_decode_block_call_counts=counts,
                  untrained_endpoint_difference=float((p2-p1).abs().mean()))
    assert report['official_parameter_count'] == report['k2_parameter_count']
    del origin, loop, p1, p2, a, b
    torch.cuda.empty_cache()
    model = LoopDenoiser(args, deep_supervised=True, activation_checkpointing=True).to(device).train()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        objective, final, intermediate = model.training_losses(x, y)
    expected = (final+(1/3)*intermediate)/(1+1/3)
    assert torch.equal(objective, expected)
    objective.backward()
    torch.cuda.synchronize()
    grad = model.net.blocks[4].attn.in_proj_weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0
    report.update(bf16_forward_backward_pass=True, ds_formula_exact=True,
                  synthetic_batch_size=2, synthetic_loss=float(objective.detach()),
                  loop_block_gradient_norm=float(grad.norm()),
                  smoke_seconds=time.perf_counter()-start,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved())
    model.zero_grad(set_to_none=True)
    model.deep_supervised = False
    with torch.autocast('cuda', dtype=torch.bfloat16):
        naive, final, intermediate = model.training_losses(x, y)
    assert intermediate is None and torch.equal(naive, final)
    naive.backward()
    torch.cuda.synchronize()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    report['naive_bf16_forward_backward_pass'] = True
    (ROOT/'analysis'/'sanity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
