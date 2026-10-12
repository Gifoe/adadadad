from common import *
from response_model import build_model,SimpleNamespace
from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
def run_one(a,noise,variant):
    rid=f'a{a}_noise{noise}_{variant}';out=ROOT/'results/synthetic'/rid;out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():return json.loads((out/'metrics.json').read_text())
    gen=np.random.default_rng(419);u=gen.normal(size=(4096,1));eps=gen.normal(scale=noise,size=(4096,1));v=np.zeros_like(u)
    for j in range(1,len(v)):v[j]=a*v[j-1]+u[j]+eps[j]
    tr=int(len(v)*.7);tv=int(len(v)*.8);ym=v[:tr].mean(0);ys=v[:tr].std(0);cm=u[:tr].mean(0);cs=u[:tr].std(0)
    y=torch.tensor((v-ym)/ys,dtype=torch.float32);c=torch.tensor((u-cm)/cs,dtype=torch.float32)
    config={'seq_len':24,'horizon':48,'pred_len':48,'latent_dim':16,'condition_dim':8,'future_x_dim':1,'output_dim':1,'series_dim':1,'state_patch_len':3,'state_hidden_multiplier':2,'decoder_patch_len':3,'decoder_hidden_multiplier':2,'decoder_type':'causal_patch','state_encoder':'causal_patch','predictor_patch_len':8,'predictor_patch_stride':4,'predictor_d_model':8,'predictor_d_ff':16,'predictor_n_heads':2,'predictor_e_layers':1,'predictor_dropout':0.,'use_revin':True,'use_future_x_condition':True,'latent_regularizer_type':'legacy_vc'}
    seed(2021);spec={'variant':variant,'k':4 if variant=='C5' else 1,'width':16,'seed':2021};model=build_model(SimpleNamespace(**config),spec).cuda();crit=LatentStateLoss(latent_mse_weight=10,latent_cosine_weight=15,latent_sigreg_weight=.1,latent_alignment_loss='mae',latent_detach_target=True,sigreg_std_weight=1.,sigreg_cov_weight=.04,sigreg_eps=1e-4)
    loaders={name:torch.utils.data.DataLoader(Windows(y,c,24,48,start,end),batch_size=32,shuffle=name=='train',drop_last=name=='train') for name,start,end in [('train',24,tr),('val',tr,tv),('test',tv,len(y))]};history=[];start=time.time()
    def objective(x,cf,target,stage):
        z=model(x,x_future=cf,future_target=target)
        if stage==1:return crit(z['h_pred'],z['h_target'])['loss']
        pred=model.decode(z['h_pred'],z['target_revin_stats'],z['h_hist']);warm=model.decode(z['h_target'],z['target_revin_stats'],z['h_hist']);return (pred-target).abs().mean()+.3*(warm-target).abs().mean()
    yh,ch,cf,target,_=next(iter(loaders['val']));audit_x=torch.cat([yh,ch],-1).cuda();audit_f=cf.cuda();audit_t=target.cuda();model.eval()
    with torch.no_grad():
        z=model(audit_x,x_future=audit_f,future_target=audit_t);z2=model(audit_x,x_future=audit_f,future_target=audit_t+17)
        assert torch.equal(z['h_pred'],z2['h_pred']);assert model.forecast(audit_x,x_future=audit_f).shape==audit_t.shape
    gradient_checks=[];frozen_before=None
    for stage in [1,2]:
        model.configure_for_stage1() if stage==1 else model.configure_for_stage2();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001 if stage==1 else .0001);best=float('inf');wait=0;state=None
        if stage==2:frozen_before=[tensor_sha(m) for m in model._stage1_modules()]
        for epoch in range(20):
            model.train()
            if stage==2:model.set_stage1_eval()
            for yh,ch,cf,target,o in loaders['train']:
                opt.zero_grad();loss=objective(torch.cat([yh,ch],-1).cuda(),cf.cuda(),target.cuda(),stage);assert torch.isfinite(loss);loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if stage==1 and epoch in [0,1] and len(gradient_checks)<4 and hasattr(model.predictor,'response_blocks'):
                    gradient_checks.append({n:float(p.grad.norm()) for n,p in model.predictor.response_blocks.named_parameters() if p.grad is not None})
                opt.step()
            model.eval();valid=[]
            with torch.no_grad():
                for yh,ch,cf,target,o in loaders['val']:
                    x=torch.cat([yh,ch],-1).cuda();cf=cf.cuda();target=target.cuda()
                    loss=objective(x,cf,target,stage) if stage==1 else (model.forecast(x,x_future=cf)-target).abs().mean();valid.append(float(loss))
            vl=float(np.mean(valid));history.append({'stage':stage,'epoch':epoch+1,'validation_loss':vl})
            if vl<best:best=vl;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
            else:wait+=1
            if wait>=5:break
            if epoch>=2:
                for g in opt.param_groups:g['lr']*=.9
        model.load_state_dict(state)
        if stage==2:assert frozen_before==[tensor_sha(m) for m in model._stage1_modules()]
    if gradient_checks:assert any(g>0 for row in gradient_checks for name,g in row.items() if '.out.' in name)
    write_json(out/'implementation_audit.json',{'shape_checked':True,'future_target_invariance':True,'finite_gradients':True,'stage2_frozen':True,'sampled_early_gradients':gradient_checks})
    model.eval();preds=[];targets=[]
    with torch.no_grad():
        for yh,ch,cf,target,o in loaders['test']:preds.append(model.forecast(torch.cat([yh,ch],-1).cuda(),x_future=cf.cuda()).cpu().numpy());targets.append(target.numpy())
    pred=np.concatenate(preds);true=np.concatenate(targets);mse,mae=metrics(pred,true);np.savez_compressed(out/'predictions.npz',pred=pred,true=true);torch.save({'Model':model.state_dict(),'config':config,'spec':spec},out/'checkpoint.pt');write_csv(out/'history.csv',history)
    restored=build_model(SimpleNamespace(**config),spec).cuda();restored.load_state_dict(torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)['Model'],strict=True);restored.configure_for_stage2();restored.eval();assert tensor_sha(restored)==tensor_sha(model)
    with torch.no_grad():assert torch.equal(restored.forecast(audit_x,x_future=audit_f),model.forecast(audit_x,x_future=audit_f))
    result={'system_a':a,'noise_std':noise,'variant':variant,'k':spec['k'],'mse':mse,'mae':mae,'training_seconds':time.time()-start,'status':'COMPLETE','training_family':'synthetic_from_scratch','fixed_seed':2021,'physical_response_above_5_steps':float(a**5),'source_type':'pinned_WorldTS_small_controlled_configuration','n_train_points':tr,'n_validation_points':tv-tr,'n_test_points':len(y)-tv,'checkpoint_sha256':sha(out/'checkpoint.pt'),'predictions_sha256':sha(out/'predictions.npz'),'checkpoint_reloaded_equal':True};write_json(out/'metrics.json',result);print('SYNTHETIC_RESULT',result,flush=True);return result
def main():
    rows=[run_one(a,noise,v) for a in [.2,.95] for noise in [.05,.5] for v in ['B0','B1','C5']];write_csv(ROOT/'results/stage1_synthetic.csv',rows)
    print('CONTROLLED_SYNTHETIC_EXPERIMENTS_COMPLETE',len(rows),flush=True)
if __name__=='__main__':main()
