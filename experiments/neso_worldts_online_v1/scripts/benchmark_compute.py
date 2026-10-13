"""Incremental revision-time compute, with identical cached fixed-A information."""
import json,pathlib,sys,time
import numpy as np,torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import import_native,native_config,rescale_np,pred_decode,revision_features,LatentRevisionUpdater,set_seed
from analyze_replay import csv_write

def main():
    set_seed(2021);device=torch.device('cuda');cfg=json.loads((ROOT/'configs/worldts_neso.json').read_text());out=ROOT/'outputs/model'
    scaler=json.loads((out/'train_scaler.json').read_text());sd=float(scaler['std'][0]);mu=float(scaler['mean'][0])
    with np.load(ROOT/'outputs/dataset/val.npz') as z:raw={k:z[k] for k in z.files}
    data=rescale_np(raw,scaler);klass,_=import_native(str(ROOT/'upstream/WorldTS/DAG-bench'))
    model=klass(native_config(cfg,336,48)).to(device);model.load_state_dict(torch.load(out/'stage2_best.pt',weights_only=False,map_location=device)['model_state'],strict=True);model.configure_for_stage2();model.eval()
    weights=np.array(json.loads((out/'output_correction_weights.json').read_text())['ridge_weights'],dtype=np.float32)
    updater=None
    if (out/'updater_best.pt').exists():
        state=torch.load(out/'updater_best.pt',weights_only=False,map_location=device)
        updater=LatentRevisionUpdater(model.latent_dim,hidden=state['hidden']).to(device);updater.load_state_dict(state['model_state'],strict=True);updater.eval()
    rows=[]
    for batch in [1,16]:
        hist=torch.from_numpy(data['history'][:batch]).to(device);ca_np=data['cov_a'][:batch];cb_np=data['cov_b'][:batch]
        ca=torch.from_numpy(ca_np).to(device);cb=torch.from_numpy(cb_np).to(device);mask=torch.from_numpy(data['update_mask'][:batch]).to(device)
        with torch.inference_mode():oa=model(input=hist,future_target=None,x_future=ca);old=pred_decode(model,oa).cpu().numpy();old_mw=old*sd+mu
        seasonal=data['history'][:batch,:48,0:1]*sd+mu
        def call(method,scope):
            if method=='old':return old_mw.copy()
            if method=='seasonal_naive':return seasonal.copy()
            if method=='output_correction':
                revised=(raw['cov_b'][:batch]-np.array(scaler['mean'][1:],np.float32))/np.array(scaler['std'][1:],np.float32)
                return (old+np.einsum('bhd,d->bh',revision_features(ca_np,revised),weights)[...,None])*sd+mu
            revised=cb if scope=='gpu_ready' else torch.from_numpy(((raw['cov_b'][:batch]-np.array(scaler['mean'][1:],np.float32))/np.array(scaler['std'][1:],np.float32)).astype('float32')).to(device)
            if method=='recompute':
                ob=model(input=hist,future_target=None,x_future=revised);pred=pred_decode(model,ob)
            else:pred=pred_decode(model,oa,updater(oa['h_pred'],ca,revised,mask))
            return pred if scope=='gpu_ready' else pred.cpu().numpy()*sd+mu
        for method in ['old','recompute','seasonal_naive','output_correction']+(['updater'] if updater is not None else []):
            for scope in (['gpu_ready','host_output'] if method in ['recompute','updater'] else ['host_output']):
                with torch.inference_mode():
                    for _ in range(20):call(method,scope)
                    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();elapsed=[]
                    for _ in range(100):
                        torch.cuda.synchronize();start=time.perf_counter();result=call(method,scope);torch.cuda.synchronize();elapsed.append((time.perf_counter()-start)*1000)
                        del result
                    peak=torch.cuda.max_memory_allocated()
                row={'method':method,'batch':batch,'scope':scope,'repetitions':100,'warmups':20,
                     'mean_ms_per_batch':float(np.mean(elapsed)),'p50_ms_per_batch':float(np.quantile(elapsed,.5)),
                     'p95_ms_per_batch':float(np.quantile(elapsed,.95)),'mean_ms_amortized_per_window':float(np.mean(elapsed)/batch),
                     'resident_cuda_allocated_mib':baseline/1048576,'incremental_peak_cuda_allocated_mib':max(0,peak-baseline)/1048576,
                     'memory_measure':'Torch live tensor allocations; not whole-board nvidia-smi usage',
                     'cache_policy':'fixed-A history, latent, decoder statistics and old prediction cached; new-B covariates only'}
                rows.append(row);print(json.dumps(row),flush=True)
    csv_write(ROOT/'COMPUTE_RESULTS.csv',rows)
    result={'validation_only':True,'nvidia_gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'rows':rows,
            'cost_scope':'Incremental at B; A-time model run excluded. host_output includes normalization, H2D for new covariates, compute, D2H and inverse scaling. gpu_ready excludes transfers. CPU B0/B2 use cached result copies; B3 remains CPU.',
            'label_access':'No future targets loaded into benchmark inference; target arrays in prepared archive are ignored.',
            'not_measured':'Network delivery, public forecast availability, multi-tenant load, CPU-only deployment and 6-hour cache lifetime management.'}
    (ROOT/'outputs/compute_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
