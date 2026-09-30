import pathlib,json,shutil,time,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from runtime import build,amp,sync
root=pathlib.Path(__file__).parent.resolve()
a=root/'artifacts/sanity'; b=root/'artifacts/sanity_v2_torch28_native_crash'
if a.exists():
    assert a.resolve().is_relative_to(root) and b.resolve().is_relative_to(root) and not b.exists()
    shutil.move(str(a),str(b))
c=json.loads((root/'configs/frozen.json').read_text()); c.update(micro_batch=8,activation_checkpointing=True,padding_bucket=64,eval_batch=8)
m=build(c,'step_conditioned_loop').cuda().train()
x=torch.randint(4,c['actual_vocab_size'],(8,c['max_length']),device='cuda')
torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
with amp(): loss=m(x,8).float().square().mean()
loss.backward(); sync()
r={'micro_batch':8,'max_length':c['max_length'],'activation_checkpointing':True,'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'seconds':time.perf_counter()-start}
print(r,flush=True)
assert r['peak_vram_mb']<8000
(root/'artifacts/checkpoint_memory_probe.json').write_text(json.dumps(r,indent=2))
(root/'configs/frozen.json').write_text(json.dumps(c,indent=2))
with (root/'EXPERIMENT_CHANGELOG.md').open('a',encoding='utf-8') as f:
    f.write('\n- PyTorch2.8/explicitCondaactivation/singleCPUthread retry also crashed at update~260 with0x80000003 in nvcuda64.dll. Windows event1000 identifies prior2.11 retry fault in cublasLt64_12.dll; system nvlddmkm event153 occurred at23:48:58. These are fault locations, not proven root cause. Preserve sanity_v2_torch28_native_crash. New engineering retry uses activation_checkpointing=True, microbatch8/effective128, evalbatch8, 64-token padding buckets for all models, unchanged weights/data/order/optimizer/budget/update counts. Enable faulthandler and CUDA_LAUNCH_BLOCKING=1 to obtain location evidence.\n')
