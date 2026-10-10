"""Run the published DAG NP/PJM commands with observational instrumentation only."""
from common import *
import argparse, runpy, shlex, time
from ts_benchmark.baselines.dag.dag import DAG

def execute(name,h):
    out=ROOT/'runs'/f'{name}_{h}_B0';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():print('RESUME_SKIP',out.name,flush=True);return
    original=DAG.forecast_fit
    def observed_fit(self,*args,**kwargs):
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();history=[]
        validate=self.validate
        def observed_validate(*a,**kw):
            value=validate(*a,**kw);history.append(float(value));print('DAG_EPOCH',name,h,len(history),'VALID_MAE',value,flush=True);return value
        self.validate=observed_validate
        result=original(self,*args,**kwargs)
        torch.cuda.synchronize();cost=time.perf_counter()-start
        peak=torch.cuda.max_memory_allocated()
        self.model.load_state_dict(self.check_point['Model']);self.model.eval()
        torch.save({'model':self.check_point['Model'],'config':vars(self.config)},out/'checkpoint.pt')
        df,y,c,tr,tv,sy,sc=load_data(name);test=Windows(y,c,self.config.seq_len,h,tv,len(y))
        loader=torch.utils.data.DataLoader(test,batch_size=64,shuffle=False)
        predictions=[];errors=[];abserrors=[];starts=[]
        with torch.no_grad():
            for yh,ch,cf,target,o in loader:
                pred,_=self.model(torch.cat([yh,ch],-1).cuda(),cf.cuda());pred=pred[:,-h:,:1].cpu()
                errors.append((pred-target).square().mean((1,2)).numpy());abserrors.append((pred-target).abs().mean((1,2)).numpy())
                predictions.append(pred[:,:,0].numpy());starts.append(o.numpy())
        preds=np.concatenate(predictions);mse=np.concatenate(errors);mae=np.concatenate(abserrors)
        np.savez_compressed(out/'predictions.npz',pred=preds,origins=np.concatenate(starts),window_mse=mse,window_mae=mae)
        cfg={k:(v if isinstance(v,(str,bool,int,float,type(None))) else str(v)) for k,v in vars(self.config).items()}
        reference={('NP',24):(.202,.237),('NP',360):(.521,.451),('PJM',24):(.057,.143),('PJM',360):(.130,.218)}[name,h]
        json_write(out/'result.json',{'dataset':name,'horizon':h,'variant':'B0','seed':2021,'mse':float(mse.mean()),'mae':float(mae.mean()),'reference_mse':reference[0],'reference_mae':reference[1],
         'mse_relative_difference':float(mse.mean()/reference[0]-1),'epochs':len(history),'validation_mae':history,
         'train_seconds':cost,'peak_cuda_bytes':peak,'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),'configuration':cfg,'windows':len(test)})
        print('DAG_RESULT',out.name,float(mse.mean()),float(mae.mean()),flush=True)
        return result
    DAG.forecast_fit=observed_fit
    dagroot=ROOT/'upstream/DAG';os.chdir(dagroot)
    line=next(l for l in (dagroot/'scripts/w_future/DAG.sh').read_text().splitlines() if f'"{name}.csv"' in l and f'"horizon": {h},' in l)
    argv=shlex.split(line)[1:]
    argv += ['--eval-backend','sequential','--seed','2021','--deterministic','efficient','--save-true-pred','True','--report-method','csv']
    # Save exact published arguments plus environment-only execution choices.
    json_write(out/'official_command.json',{'published_line':line,'argv':argv,'architecture_modified':False,'execution_adjustments':['sequential evaluation backend (same strategy)','CPU affinity and four CPU threads','PyTorch 2.8 sm120 support instead of incompatible 2.4 build']})
    sys.argv=argv
    try:runpy.run_path(str(dagroot/'scripts/run_benchmark.py'),run_name='__main__')
    finally:DAG.forecast_fit=original
    assert (out/'result.json').exists(), 'Official runner failed; inspect log'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM']);p.add_argument('--horizon',type=int,choices=[24,360]);a=p.parse_args()
    audit_data();execute(a.dataset,a.horizon)
