from common import *
from response_model import WorldTSModel,SimpleNamespace
import platform,subprocess
def run():
    assert torch.cuda.is_available(),'GPU required for baseline reevaluation';device=torch.device('cuda')
    rows=[]
    for ds,h in [('NP',360),('PJM',360),('NP',24),('PJM',24)]:
        out=ROOT/'results/stage0'/f'{ds}_H{h}';out.mkdir(parents=True,exist_ok=True)
        src=baseline_dir(ds,h);old=json.loads((src/'metrics.json').read_text());checkpoint=pathlib.Path(old['checkpoint_path']);assert sha(checkpoint)==old['checkpoint_sha256'] and old['source_commit']==SOURCE_COMMIT
        model=WorldTSModel(SimpleNamespace(**cfg(ds,h))).to(device).eval();model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)['Model']);model.requires_grad_(False)
        df,y,c,tr,tv,sy,sc=load_data(ds);ss=np.load(src/'scaler.npz')
        for key,value in [('mean_y',sy.mean_),('scale_y',sy.scale_),('mean_c',sc.mean_),('scale_c',sc.scale_)]:assert np.array_equal(ss[key],value)
        windows=Windows(y,c,model.seq_len,h,tv,len(y));start=time.time();preds=[];targets=[];origins=[]
        with torch.no_grad():
            for yh,ch,cf,t,o in torch.utils.data.DataLoader(windows,batch_size=64):preds.append(model.forecast(torch.cat([yh,ch],-1).to(device),x_future=cf.to(device)).cpu().numpy());targets.append(t.numpy());origins.append(o.numpy())
        pred=np.concatenate(preds);true=np.concatenate(targets);oo=np.concatenate(origins);prior=np.load(src/'predictions.npz');assert sha(src/'predictions.npz')==old['predictions_sha256']
        assert np.array_equal(oo,prior['origins']) and np.array_equal(true,prior['true']);delta=float(np.max(abs(pred-prior['pred'])));assert delta<=1e-5
        mse,mae=metrics(pred,true);assert abs(mse-old['reproduced_mse'])<1e-6 and abs(mae-old['reproduced_mae'])<1e-6
        np.savez_compressed(out/'predictions.npz',pred=pred,true=true,origins=oo);np.savez(out/'scaler.npz',**{k:ss[k] for k in ss.files})
        row={'dataset':ds,'horizon':h,'run_type':'verified_reuse_with_fresh_inference','mse':mse,'mae':mae,'historical_mse':old['reproduced_mse'],'historical_mae':old['reproduced_mae'],'max_prediction_abs_diff':delta,'windows':len(oo),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'predictions_sha256':sha(out/'predictions.npz'),'data_sha256':DATA_SHA[ds],'elapsed_seconds':time.time()-start,'status':'VERIFIED'};rows.append(row);print('BASELINE_REVERIFIED',row,flush=True)
    a=torch.randn(16,16,device=device,requires_grad=True);a.square().mean().backward();assert torch.isfinite(a.grad).all()
    env={'device':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'platform':platform.platform(),'source_commit':SOURCE_COMMIT,'baseline_root':str(BASE),'cpu_threads':torch.get_num_threads(),'affinity':'CPU16-31 only when detected logical CPU count is at least32','cuda_backward_passed':True}
    write_json(ROOT/'environment.json',env);write_csv(ROOT/'results/stage0_baseline_verification.csv',rows)
    text='# Stage 0 — Baseline audit\n\n'
    sections={'Research Question':'Can the current four official checkpoints support controlled comparisons?','Pre-experiment Hypothesis':'All raw targets, scaler parameters, rolling origins and model predictions match the verified historical artifacts.','Experiment Design':'Read-only baseline artifacts; fresh GPU inference on every official test origin; fixed source '+SOURCE_COMMIT+'.','Actual Execution':'Four configurations freshly reevaluated; no historical baseline file or checkpoint modified.','Raw Results':pd.DataFrame(rows)[['dataset','horizon','mse','mae','max_prediction_abs_diff','windows']].to_markdown(index=False),'Interpretation':'Checkpoint reuse verified by source/data/checkpoint/prediction hashes, exact raw targets and all test origins. This does not establish bitwise reproducibility of retraining.','Scientific Contribution':'This is an audit prerequisite, not a methodological contribution.','Limitations and Alternative Explanations':'Single pretrained seed; efficient GPU determinism may not reproduce training trajectories. Future covariate vintages remain unknown. Original PJM24 training peak was unavailable.','Decision':'Proceed with isolated, matched two-stage continued-training controls; keep historical checkpoint results separate from newly trained controls.','Reproduction':'Activate the verified baseline private venv and execute python scripts/stage0.py. Detailed verified rows: results/stage0_baseline_verification.csv.'}
    for title,body in sections.items():text+='## '+title+'\n\n'+body+'\n\n'
    (ROOT/'reports/STAGE0_BASELINE_AUDIT.md').write_text(text,encoding='utf-8');write_json(ROOT/'STATUS.json',{'stage':0,'status':'STAGE0_VERIFIED','overall_complete':False,'verified_baselines':4})
if __name__=='__main__':run()
