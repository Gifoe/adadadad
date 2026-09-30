import json
import numpy as np
import torch
from runtime import *
from solvers.fixed_step import integrate,schedule

@torch.no_grad()
def numerical_diagnostic():
    m,c,_=checkpoint_model('vector_field'); m.float(); data=load_data('iid')
    ids=np.sort(np.random.default_rng(9300).choice(len(data['labels']),c['diagnostic_count'],replace=False))
    jobs=[('rk4',32),('rk4',64)]+[(s,n) for s,steps in [('euler',[4,8,16,32,64]),('heun',[2,4,8,16,32]),('rk4',[1,2,4,8,16])] for n in steps]
    errors={f'{s}_{n}':[] for s,n in jobs}; drifts=[]; finest_acc=[]; reference_acc=[]
    # FP32 network evaluation is deliberate: BF16 rounding would hide numerical convergence.
    # This diagnostic has a different precision from task/latency measurements and is labeled.
    for start in range(0,len(ids),8):
        x,y=batch(data,ids[start:start+8],device()); h0,mask=m.encode(x)
        field=lambda h,t:m.core(h,t,mask)
        h32,_=integrate(field,h0,schedule(32),'rk4'); h64,_=integrate(field,h0,schedule(64),'rk4')
        ref=h64; den=ref.flatten(1).norm(dim=-1)+1e-8
        drift=(h32-h64).flatten(1).norm(dim=-1)/den; drifts.extend(drift.cpu().tolist())
        reference_acc.extend((m.readout(h64,mask).argmax(-1)==y).cpu().tolist())
        finest_acc.extend((m.readout(h32,mask).argmax(-1)==y).cpu().tolist())
        for sol,n in jobs:
            end=h32 if (sol,n)==('rk4',32) else h64 if (sol,n)==('rk4',64) else integrate(field,h0,schedule(n),sol)[0]
            err=(end-ref).flatten(1).norm(dim=-1)/den
            errors[f'{sol}_{n}'].extend(err.cpu().tolist())
        print('numerical',min(start+8,len(ids)),flush=True)
    rows=[]; slopes={}
    for sol,n in jobs:
        rows.append({'solver':sol,'steps':n,'nfe':n*{'euler':1,'heun':2,'rk4':4}[sol], 'dt':1/n,'relative_endpoint_error':float(np.mean(errors[f'{sol}_{n}'])),'precision':'fp32','examples':len(ids)})
    for sol in ['euler','heun','rk4']:
        r=[x for x in rows if x['solver']==sol and x['steps']<={'euler':64,'heun':32,'rk4':16}[sol] and x['relative_endpoint_error']>0]
        slopes[sol]=float(np.polyfit(np.log([x['dt'] for x in r]),np.log([x['relative_endpoint_error'] for x in r]),1)[0])
    drift=float(np.mean(drifts)); coarse=next(r['relative_endpoint_error'] for r in rows if r['solver']=='euler' and r['steps']==4)
    result={'rows':rows,'observed_slopes':slopes,'reference_32_vs_64':drift,'reference_drift_vs_euler4_error':drift/max(coarse,1e-12),
            'check_I_passed':bool(drift<coarse*.1),'reference_rk4_32_accuracy':float(np.mean(finest_acc)),
            'reference_rk4_64_accuracy':float(np.mean(reference_acc)), 'reference_nfe':256,
            'qualification':'Reference accuracy is diagnostic only. If Check I fails, do not claim a sufficiently resolved reference.',
            'warning':'numerical convergence itself does not establish improved reasoning.'}
    out=ROOT/'artifacts/vector_field'; np.savez_compressed(out/'convergence_per_example.npz',indices=ids,**{k:np.array(v) for k,v in errors.items()})
    (out/'convergence.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__': numerical_diagnostic()
