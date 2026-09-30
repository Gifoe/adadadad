import json,time,os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
from runtime import ROOT,build,amp,sync
c=json.loads((ROOT/'configs/base.json').read_text()); c['activation_checkpointing']=False
results=[]
for b in [16,32,64]:
    m=build(c,'step_conditioned_loop').cuda().train(); torch.cuda.reset_peak_memory_stats()
    x=torch.randint(4,8192,(b,256),device='cuda'); sync(); begin=time.perf_counter()
    try:
        for i in range(2):
            m.zero_grad(set_to_none=True)
            with amp(): loss=m(x,8).float().square().mean()
            loss.backward()
        sync()
        row={'micro_batch':b,'length':256,'budget':8,'checkpointing':False,'seconds_per_microbatch':(time.perf_counter()-begin)/2,'peak_mb':torch.cuda.max_memory_allocated()/2**20}
    except torch.cuda.OutOfMemoryError:
        row={'micro_batch':b,'oom':True}
    results.append(row); print(row,flush=True); del m,x; torch.cuda.empty_cache()
(ROOT/'artifacts/hardware_probe.json').write_text(json.dumps(results,indent=2))
