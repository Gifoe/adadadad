import json,time
import numpy as np
import torch
from runtime import *
from evaluate import configurations

@torch.no_grad()
def benchmark(name):
    m,c,_=checkpoint_model(name); data=load_data('iid'); bs=c['eval_batch']
    # Fixed bank of 100 IID batches, reused for every architecture and configuration.
    bank=[batch(data,np.arange(i*bs,(i+1)*bs),device())[0] for i in range(c['latency_batches'])]
    results=[]
    for budget,solver,sk in configurations(name,c):
        for i in range(c['latency_warmups']):
            with amp(): m(bank[i%len(bank)],budget,solver if budget else 'euler',sk if budget else 'uniform')
        sync(); torch.cuda.reset_peak_memory_stats(); measurements=[]
        for x in bank:
            sync(); begin=time.perf_counter()
            with amp(): m(x,budget,solver if budget else 'euler',sk if budget else 'uniform')
            sync(); measurements.append((time.perf_counter()-begin)*1000)
        r={'model':name,'budget':budget,'solver':solver,'schedule':sk,'batch_size':bs,'precision':'bf16',
            'warmup_runs':c['latency_warmups'],'timed_batches':len(bank),'latency_mean_ms':float(np.mean(measurements)),
            'latency_median_ms':float(np.median(measurements)),'latency_p95_ms':float(np.percentile(measurements,95)),
            'examples_per_sec':bs*1000/float(np.mean(measurements)),'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,
            'padded_length_distribution':[int(x.shape[1]) for x in bank]}
        results.append(r); print('latency',name,budget,solver,sk,r['latency_mean_ms'],flush=True)
        (ROOT/'artifacts'/name/'latency.json').write_text(json.dumps(results,indent=2))
    del bank,m; torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=NAMES,required=True); benchmark(p.parse_args().model)
