from common import *
from response_model import WorldTSModel,SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def run():
    rows=[];horizon_rows=[];diagnostics=[];fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ds,h in [('NP',360),('PJM',360),('NP',24),('PJM',24)]:
        config=cfg(ds,h);model=WorldTSModel(SimpleNamespace(**config)).cuda().eval();model.load_state_dict(torch.load(baseline_dir(ds,h)/'stages/stage2_decoder_mix.pt',map_location='cuda',weights_only=False)['Model']);model.requires_grad_(False)
        _,y,c,tr,tv,_,_=load_data(ds);oo=np.linspace(tr,tv-h,32,dtype='int64');x=torch.stack([torch.cat([y[o-model.seq_len:o],c[o-model.seq_len:o]],-1) for o in oo]).cuda();future=torch.stack([c[o:o+h] for o in oo]).cuda().requires_grad_(True);target=torch.stack([y[o:o+h] for o in oo]).cuda()
        out=model.forecast(x,x_future=future);jac=[]
        for j in range(h):jac.append(torch.autograd.grad(out[:,j].sum(),future,retain_graph=True)[0].detach().cpu().numpy())
        jac=np.stack(jac,axis=1);s=np.linalg.norm(jac,axis=-1).mean(0);r=config['state_patch_len']+config['decoder_patch_len']-1;hs=np.arange(h)[:,None]-np.arange(h)[None,:];outside=(hs<0)|(hs>=r)
        outmax=float(np.max(s[outside]));assert outmax<=1e-9
        with torch.no_grad():
            reference=model.forecast(x,x_future=future);fd_max=0.
            for pos in sorted(set([0,h//3,2*h//3,h-1])):
                for ch in [0,1]:
                    a=future.detach().clone();b=a.clone();a[:,pos,ch]+=.001;b[:,pos,ch]-=.001;fd=((model.forecast(x,x_future=a)-model.forecast(x,x_future=b))/.002).cpu().numpy()[:,:,0]
                    fd_max=max(fd_max,float(np.max(abs(fd-jac[:,:,pos,ch]))))
            a=model(x,future_target=target,x_future=future);yp=model.decode(a['h_pred'],a['target_revin_stats'],a['h_hist']);oracle=model.decode(a['h_target'],a['target_revin_stats'],a['h_hist']);base=model.predictor.autonomous_trajectory(a['h_hist'],a['z_cond_hist']);delta=a['h_pred']-base
            pm,pa=metrics(yp.cpu().numpy(),target.cpu().numpy());om,oa=metrics(oracle.cpu().numpy(),target.cpu().numpy())
            diagnostics.append({'dataset':ds,'horizon':h,'origins':oo.tolist(),'pred_mse':pm,'pred_mae':pa,'encoded_future_diagnostic_mse':om,'encoded_future_diagnostic_mae':oa,'latent_l1':float((a['h_pred']-a['h_target']).abs().mean()),'correction_to_base_norm':float(delta.norm()/base.norm()),'receptive_positions':r,'maximum_time_displacement':r-1,'outside_support_max_sensitivity':outmax,'finite_difference_max_abs_discrepancy':fd_max,'unit':'train-standardized covariate; history RevIN fixed','oracle_is_deployable':False})
            assert fd_max<.02,'Finite difference discrepancy unexpectedly large'
        np.savez_compressed(ROOT/'results'/f'stage1_jacobian_{ds}_{h}.npz',jacobian=jac,origins=oo,sensitivity=s)
        for dest in range(h):
            for src in range(h):
                if s[dest,src]!=0:rows.append({'dataset':ds,'horizon':h,'output_position':dest,'input_position':src,'mean_gradient_norm':s[dest,src],'n_origins':32,'theoretical_range':r,'outside_theoretical_support':False})
        windows=Windows(y,c,model.seq_len,h,tr,tv);sum_sq=np.zeros(h);sum_abs=np.zeros(h);n=0
        with torch.no_grad():
            for yh,ch,cf,t,orig in torch.utils.data.DataLoader(windows,batch_size=64):
                p=model.forecast(torch.cat([yh,ch],-1).cuda(),x_future=cf.cuda()).cpu().numpy();e=p.astype('float64')-t.numpy().astype('float64');sum_sq+=np.sum(e[:,:,0]**2,0);sum_abs+=np.sum(abs(e[:,:,0]),0);n+=len(p)
        for j in range(h):horizon_rows.append({'dataset':ds,'horizon':h,'position':j,'validation_mse':sum_sq[j]/n,'validation_mae':sum_abs[j]/n,'validation_windows':n})
        if h==360:ax=axes[0 if ds=='NP' else 1];im=ax.imshow(np.log10(s+1e-12),origin='lower',aspect='auto',vmin=-12);ax.set(title=f'{ds}: R={r}',xlabel='Future covariate position',ylabel='Output position');fig.colorbar(im,ax=ax)
        print('STRUCTURAL_RESPONSE_VERIFIED',ds,h,r,outmax,fd_max,flush=True)
    fig.tight_layout();fig.savefig(ROOT/'figures/stage1_response_heatmap.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for ds in ['NP','PJM']:
        for h in [24,360]:rr=[r for r in horizon_rows if r['dataset']==ds and r['horizon']==h];ax.plot([r['position'] for r in rr],[r['validation_mse'] for r in rr],label=f'{ds} H{h}')
    ax.set(xlabel='Forecast position',ylabel='Validation MSE');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/stage1_error_by_horizon.png',dpi=180);plt.close(fig)
    write_csv(ROOT/'results/stage1_sensitivity.csv',rows);write_csv(ROOT/'results/stage1_error_by_horizon.csv',horizon_rows);write_json(ROOT/'results/stage1_response_diagnostics.json',diagnostics)
    print('STAGE1_REAL_DATA_DIAGNOSIS_COMPLETE',flush=True)
if __name__=='__main__':run()
