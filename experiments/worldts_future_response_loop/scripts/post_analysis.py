from common import *
from response_model import build_model,SimpleNamespace
def predict(model,loader,k=None,mode='full'):
    if k is not None:model.predictor.response_k=k
    preds=[];targets=[];origins=[];gen=torch.Generator().manual_seed(719)
    with torch.no_grad():
        for yh,ch,cf,t,o in loader:
            if mode=='mask_first':cf=cf.clone();cf[:,:,0]=0
            if mode=='noise_0.25':cf=cf+.25*torch.randn(cf.shape,generator=gen)
            if mode=='no_future':model.allow_missing_future_covariate=True;cf=None
            p=model.forecast(torch.cat([yh,ch],-1).cuda(),x_future=cf.cuda() if cf is not None else None)
            preds.append(p.cpu().numpy());targets.append(t.numpy());origins.append(o.numpy())
    return np.concatenate(preds),np.concatenate(targets),np.concatenate(origins)
def run():
    costs=[];depth=[];robust=[];trajectories=[];sensitivity=[];verification=[];parameters=[]
    for path in sorted((ROOT/'runs').glob('*/metrics.json')):
        out=path.parent;m=json.loads(path.read_text());spec=json.loads((out/'research_spec.json').read_text());ds=m['dataset'];h=m['horizon'];_,y,c,tr,tv,sy,sc=load_data(ds);config=json.loads((out/'resolved_config.json').read_text());model=build_model(SimpleNamespace(**config),spec);model.load_state_dict(torch.load(m['checkpoint_path'],map_location='cpu',weights_only=False)['Model'],strict=True);model=model.cuda().eval();model.configure_for_stage2()
        assert sha(m['checkpoint_path'])==m['checkpoint_sha256'];arr=np.load(out/'predictions.npz');assert sha(out/'predictions.npz')==m['predictions_sha256'];origins=np.arange(tv,len(y)-h+1);assert np.array_equal(origins,arr['origins'])
        for start in range(0,len(origins),64):assert np.array_equal(arr['true'][start:start+64],y[origins[start:start+64,None]+np.arange(h)[None,:]].numpy())
        mse,mae=metrics(arr['pred'],arr['true']);assert abs(mse-m['reproduced_mse'])<1e-12 and abs(mae-m['reproduced_mae'])<1e-12
        sca=np.load(out/'scaler.npz')
        for name,value in [('mean_y',sy.mean_),('scale_y',sy.scale_),('mean_c',sc.mean_),('scale_c',sc.scale_)]:assert np.array_equal(sca[name],value)
        assert json.loads((out/'freeze_audit.json').read_text())['unchanged'];verification.append({'run_id':m['run_id'],'passed':True,'mse':mse,'mae':mae,'windows':len(origins),'all_raw_targets_exact':True,'scaler_train_only':True,'checkpoint_sha256':m['checkpoint_sha256'],'predictions_sha256':m['predictions_sha256']})
        if hasattr(model.predictor,'response_blocks'):
            eta=float(torch.sigmoid(model.predictor.response_eta_logit).detach())
            for j,block in enumerate(model.predictor.response_blocks):
                norm=float(block.out.weight.detach().norm());assert np.isfinite(norm) and np.isfinite(eta)
                parameters.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'variant':m['variant'],'seed':m['seed'],'block':j,'eta':eta,'output_projection_frobenius_norm':norm,'output_projection_initial_norm':0.,'output_projection_changed_from_zero_initialization':bool(norm>0),'historical_loop_effective_coefficient':None})
        if m['variant']=='C1':
            coefficient=float((.5*torch.tanh(model.predictor.patch_predictor.encoder.gate)).detach());assert np.isfinite(coefficient)
            parameters.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'variant':m['variant'],'seed':m['seed'],'block':'historical_encoder','eta':None,'output_projection_frobenius_norm':None,'output_projection_initial_norm':None,'output_projection_changed_from_zero_initialization':None,'historical_loop_effective_coefficient':coefficient})
        oo=np.linspace(tr,tv-h,32,dtype='int64');x=torch.stack([torch.cat([y[o-model.seq_len:o],c[o-model.seq_len:o]],-1) for o in oo]).cuda();cf=torch.stack([c[o:o+h] for o in oo]).cuda();target=torch.stack([y[o:o+h] for o in oo]).cuda();params=sum(p.numel() for p in model.parameters())
        timings={}
        with torch.no_grad():
            for batch in [1,32]:
                for _ in range(10):model.forecast(x[:batch],x_future=cf[:batch])
                torch.cuda.synchronize();values=[]
                for _ in range(30):
                    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();model.forecast(x[:batch],x_future=cf[:batch]);b.record();b.synchronize();values.append(a.elapsed_time(b))
                timings['batch'+str(batch)+'_ms_median']=float(np.median(values));timings['batch'+str(batch)+'_ms_p90']=float(np.quantile(values,.9))
            a=model(x,x_future=cf,future_target=target);b=model(x,x_future=cf,future_target=target+17);assert torch.equal(a['h_pred'],b['h_pred'])
            if hasattr(model.predictor,'response_trace'):
                model(x,x_future=cf);trace=model.predictor.response_trace;base=model.predictor.response_base
                for j,z in enumerate(trace):
                    yp=model.decode(base+z,a['target_revin_stats'],a['h_hist']);vm,va=metrics(yp.cpu().numpy(),target.cpu().numpy());trajectories.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'variant':m['variant'],'trained_k':m['k'],'iteration':j,'delta_l2_mean':float(z.flatten(1).norm(dim=1).mean()),'delta_rms':float(z.square().mean().sqrt()),'step_l2_mean':float((z-trace[j-1]).flatten(1).norm(dim=1).mean()) if j else 0.,'validation_sample_mse':vm,'validation_sample_mae':va,'validation_latent_l1':float((base+z-a['h_target']).abs().mean()),'origins_count':32})
        attempts=[json.loads(p.read_text()) for p in (ROOT/'logs').glob(m['run_id']+'_attempt*.json')];attempt_seconds=sum(v['end_time']-v['start_time'] for v in attempts) if attempts else m['end_time']-m['start_time']
        costs.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'variant':m['variant'],'k':m['k'],'seed':m['seed'],'parameters':params,'training_seconds':m['train_seconds'],'training_time_scope':'retained/resumed optimization including validation; failed partial epoch excluded','process_attempt_wall_seconds':attempt_seconds,'failed_attempts':sum(v['exit_code']!=0 for v in attempts),'peak_gpu_mb':m['peak_gpu_mb'],**timings})
        loader=torch.utils.data.DataLoader(Windows(y,c,model.seq_len,h,tv,len(y)),batch_size=64,shuffle=False)
        fresh,raw,oo=predict(model,loader);assert np.array_equal(oo,arr['origins']) and np.array_equal(raw,arr['true']);maxdiff=float(np.abs(fresh-arr['pred']).max());assert maxdiff<=1e-6
        verification[-1]['fresh_checkpoint_inference_max_abs_diff']=maxdiff
        per_h_mse=((fresh.astype('float64')-raw)**2).mean((0,2));per_h_mae=np.abs(fresh.astype('float64')-raw).mean((0,2));write_csv(out/'error_by_horizon.csv',[{'horizon_step':j+1,'mse':float(per_h_mse[j]),'mae':float(per_h_mae[j])} for j in range(h)])
        if m['variant']=='C5':
            for k in [1,2,4]:
                pred,true,oo=predict(model,loader,k);sm,sa=metrics(pred,true);np.savez_compressed(out/f'test_depth_K{k}.npz',pred=pred,true=true,origins=oo);depth.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'seed':m['seed'],'trained_k':m['k'],'test_k':k,'mse':sm,'mae':sa,'same_checkpoint_sha256':m['checkpoint_sha256']})
            model.predictor.response_k=m['k']
        if m['seed']==2021 and m['variant'] in ['B0','B1','C5']:
            for mode in ['full','mask_first','no_future','noise_0.25']:
                pred,true,oo=predict(model,loader,mode=mode);sm,sa=metrics(pred,true);robust.append({'run_id':m['run_id'],'dataset':ds,'horizon':h,'variant':m['variant'],'k':m['k'],'condition':mode,'mse':sm,'mae':sa,'scope':'evaluation sensitivity only, not a retrained ablation or causal intervention','noise_units':'0.25 train-scaled standard deviations' if mode=='noise_0.25' else ''})
            model.allow_missing_future_covariate=False
        if m['seed']==2021 and h==360:
            f=cf.detach().requires_grad_(True);p=model.forecast(x,x_future=f);r=config['state_patch_len']+config['decoder_patch_len']-1
            for j in [0,h//4,h//2,h-1]:
                g=torch.autograd.grad(p[:,j].sum(),f,retain_graph=True)[0].norm(dim=-1).detach().cpu().numpy().mean(0);distance=j-np.arange(h);off=(distance<0)|(distance>=r)
                sensitivity.append({'run_id':m['run_id'],'output_position':j,'mean_gradient_norm_outside_baseline_window':float(g[off].mean()),'max_gradient_norm_outside_baseline_window':float(g[off].max()),'n_origins':32})
        print('TRAINED_MODEL_POST_AUDIT',m['run_id'],flush=True)
    write_csv(ROOT/'results/model_costs.csv',costs);write_csv(ROOT/'results/response_parameter_audit.csv',parameters);write_csv(ROOT/'results/stage3_loop_depth.csv',depth);write_csv(ROOT/'results/stage4_robustness.csv',robust);write_csv(ROOT/'results/stage4_trajectory.csv',trajectories);write_csv(ROOT/'results/stage4_response_sensitivity.csv',sensitivity);write_json(ROOT/'results/independent_verification.json',{'all_passed':True,'results':verification,'dtype':'float64','raw_target_check':'every official origin, train-only scaler'})
if __name__=='__main__':run()
