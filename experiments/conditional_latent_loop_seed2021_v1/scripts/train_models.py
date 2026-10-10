from common import *
from models.conditional_loop import ConditionalLoop,latent_loss
import time,argparse,copy
from torch.utils.data import DataLoader

def build(name,t,h,v,k):
    seed_all();ae=official_ae();ae.load_state_dict(torch.load(ROOT/'ae'/name/'checkpoint.pt',weights_only=True))
    return ConditionalLoop(ae,t,h,v,k,128 if name=='NP' else 256).cuda()

def model_forward(model,batch,mask=None):
    yh,ch,cf,target,o=batch;yh=yh.cuda();ch=ch.cuda();cf=cf.cuda();target=target.cuda()
    if mask=='zero':cf=torch.zeros_like(cf)
    if mask=='shuffle':
        # Fixed time permutation within each window; invariant to evaluation batch size.
        order=torch.randperm(cf.shape[1],generator=torch.Generator().manual_seed(2021)).cuda();cf=cf[:,order]
    zs,ys=model(yh,ch,cf)
    return zs,ys,target

@torch.no_grad()
def validation(model,loader):
    model.eval();total=0;count=0
    for batch in loader:
        zs,ys,y=model_forward(model,batch);total+=float((ys[-1]-y).square().sum());count+=y.numel()
    return total/count

def evaluate(model,ds,out):
    model.eval();preds=[];zs_mse=[];zs_cos=[];latent_recon=[];origins=[];windows=[];maes=[];horizon=[];iteration_zm=[];iteration_zc=[]
    masked={'zero':[],'shuffle':[]}
    loader=DataLoader(ds,batch_size=128,shuffle=False)
    with torch.no_grad():
        for batch in loader:
            zs,ys,y=model_forward(model,batch)
            errs=torch.stack([(v-y).square()[:,:,0] for v in ys],1)
            windows.append(errs.mean(-1).cpu().numpy());horizon.append(errs.sum(0).cpu().numpy())
            maes.append(torch.stack([(v-y).abs().mean((1,2)) for v in ys],1).cpu().numpy())
            preds.append(ys[-1][:,:,0].cpu().numpy());origins.append(batch[-1].numpy())
            if model.variant!='B1':
                zy=model.ae.encode(y);zs_mse.append((zs[-1]-zy).square().mean((1,2)).cpu().numpy())
                zs_cos.append((1-torch.nn.functional.cosine_similarity(zs[-1],zy,dim=-1)).mean(1).cpu().numpy())
                iteration_zm.append(torch.stack([(v-zy).square().mean((1,2)) for v in zs],1).cpu().numpy())
                iteration_zc.append(torch.stack([(1-torch.nn.functional.cosine_similarity(v,zy,dim=-1)).mean(1) for v in zs],1).cpu().numpy())
                latent_recon.append((model.ae.decode(zy)-y).square().mean((1,2)).cpu().numpy())
            for mask in masked:
                _,pm,ym=model_forward(model,batch,mask);masked[mask].append((pm[-1]-ym).square().mean((1,2)).cpu().numpy())
    wm=np.concatenate(windows);wa=np.concatenate(maes);pr=np.concatenate(preds)
    payload={'pred':pr,'origins':np.concatenate(origins),'iteration_window_mse':wm,'iteration_window_mae':wa,'horizon_mse':np.sum(horizon,axis=0)/len(ds)}
    for mask in masked:payload[mask+'_window_mse']=np.concatenate(masked[mask])
    if zs_mse:payload.update(latent_window_mse=np.concatenate(zs_mse),latent_window_cosine=np.concatenate(zs_cos),reconstruction_window_mse=np.concatenate(latent_recon),iteration_latent_window_mse=np.concatenate(iteration_zm),iteration_latent_window_cosine=np.concatenate(iteration_zc))
    np.savez_compressed(out/'predictions.npz',**payload)
    return {'mse':float(wm[:,-1].mean()),'mae':float(wa[:,-1].mean()),'per_iteration_mse':wm.mean(0).tolist(),
        'mask_mse':{m:float(payload[m+'_window_mse'].mean()) for m in masked},'latent_mse':float(np.concatenate(zs_mse).mean()) if zs_mse else None,
        'latent_cosine':float(np.concatenate(zs_cos).mean()) if zs_cos else None,'decoded_true_latent_mse':float(np.concatenate(latent_recon).mean()) if latent_recon else None,'windows':len(ds)}

def profile(model,ds):
    yh,ch,cf,_,_=next(iter(DataLoader(ds,batch_size=1)));args=(yh.cuda(),ch.cuda(),cf.cuda());model.eval()
    with torch.no_grad():
        for _ in range(15):model(*args,return_all=False)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):model(*args,return_all=False)
        torch.cuda.synchronize();latency=(time.perf_counter()-start)/100
        fastpath=torch.backends.mha.get_fastpath_enabled();torch.backends.mha.set_fastpath_enabled(False)
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:model(*args,return_all=False)
        finally:torch.backends.mha.set_fastpath_enabled(fastpath)
        flops=sum(e.flops for e in prof.key_averages())
    # Analytical multiply-add count includes attention products missing from fused profiler counters.
    t,h,d,ff=model.t,model.h,64,model.ff;ph=t//24;pf=h//24
    block=lambda n:8*n*d*d+4*n*d*ff+4*n*n*d
    common=2*(t+h)*(2*32+32*32)+4*t*d*d+2*d*ph*pf+2*h*d*32+2*h*(64*64+64*32)+2*block(ph)
    if model.variant=='B1':common+=2*t*32+2*h*32
    else:common+=2*t*(64+64*32)+2*h*(32*64+64)
    if model.variant in ('B3','B4','B5'):common+=model.k*(6*h*32*d+block(pf))
    return {'inference_ms_batch1':latency*1000,'profiled_flops_batch1':int(flops),'flop_scope':'PyTorch with_flops: supported matmul/conv only; excludes some ops',
        'analytic_dense_attention_flops_batch1':int(common),'analytic_flop_scope':'multiply/add FLOPs of linear and dense attention products; excludes bias, activation, normalization, scalar residuals; final-only decoder',
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'all_parameters':sum(p.numel() for p in model.parameters())}

def train(name,t,h,v,k):
    suffix=f'{name}_{h}_{v}_K{k}' if v in ('B3','B4','B5') else f'{name}_{h}_{v}'
    out=ROOT/'runs'/suffix;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():print('RESUME_SKIP',suffix,flush=True);return
    model=build(name,t,h,v,k);train_ds,val_ds,test_ds=datasets(name,t,h)
    # Every model sees the same complete chronological training-window set.
    train_loader=DataLoader(train_ds,batch_size=64,shuffle=True,drop_last=False)
    val_loader=DataLoader(val_ds,batch_size=128,shuffle=False)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    best=float('inf');wait=0;history=[];start=time.perf_counter();updates=0;torch.cuda.reset_peak_memory_stats()
    config={'dataset':name,'history':t,'horizon':h,'variant':v,'K':k,'seed':2021,'patch_len':24,'d_model':64,'heads':4,'initial_layers':2,
        'd_ff':128 if name=='NP' else 256,'dropout':0,'batch_size':64,'optimizer':'Adam','lr':.001,'max_epochs':50,'patience':5,
        'lradj':'type3' if name=='NP' else 'type1','checkpoint_selection':'decoded validation MSE','drop_last':False,
        'loss':'observation MSE' if v=='B1' else '10 * latent MSE + 15 * per-timestamp cosine loss','intermediate_supervision':False,
        'residual_refinement_scale':.1,'AE_frozen':True,'future_conditions_initial':True,'future_conditions_refinement':v in ('B3','B5')}
    json_write(out/'config.json',config)
    for epoch in range(50):
        model.train();total=0;count=0
        for batch in train_loader:
            opt.zero_grad();zs,ys,y=model_forward(model,batch)
            if v=='B1':loss=(ys[-1]-y).square().mean()
            else:
                with torch.no_grad():zy=model.ae.encode(y)
                loss=latent_loss(zs[-1],zy)
            assert torch.isfinite(loss),'Nonfinite training loss'
            loss.backward();opt.step();updates+=1;total+=float(loss.detach())*len(y);count+=len(y)
        vm=validation(model,val_loader);history.append({'epoch':epoch+1,'train_loss':total/count,'val_mse':vm,'lr':opt.param_groups[0]['lr']})
        if vm<best:
            best=vm;wait=0;best_epoch=epoch+1;torch.save(model.state_dict(),out/'checkpoint.pt')
        else:wait+=1
        print('MODEL_EPOCH',suffix,epoch+1,'TRAIN',total/count,'VAL_MSE',vm,'seconds',time.perf_counter()-start,flush=True)
        json_write(out/'progress.json',{'state':'training','epochs':len(history),'best_val_mse':best,'updates':updates,'elapsed_seconds':time.perf_counter()-start})
        if wait>=5:break
        next_epoch=epoch+1
        lr=.001 if name=='NP' and next_epoch<3 else (.001*.9**(next_epoch-3) if name=='NP' else .001*.5**(next_epoch-1))
        for pg in opt.param_groups:pg['lr']=lr
    torch.cuda.synchronize();cost=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()
    model.load_state_dict(torch.load(out/'checkpoint.pt',weights_only=True));model.eval()
    metrics=evaluate(model,test_ds,out);budget=profile(model,test_ds)
    result={**config,**metrics,**budget,'best_val_mse':best,'best_epoch':best_epoch,'epochs':len(history),'history_log':history,
        'train_seconds':cost,'peak_cuda_bytes':peak,'optimizer_updates':updates}
    json_write(out/'result.json',result);json_write(out/'progress.json',{'state':'complete','mse':metrics['mse'],'mae':metrics['mae']})
    print('MODEL_RESULT',suffix,metrics,flush=True)

def sanity():
    name='NP';t=168;h=24
    b1=build(name,t,h,'B1',2);b2=build(name,t,h,'B2',2)
    common_parts=('cov_encoder','hist_fusion','patch_embed','hist_position','predictor','time_projection','latent_head','future_fusion')
    assert all(torch.equal(value,b2.state_dict()[key]) for key,value in b1.state_dict().items() if key.startswith(common_parts))
    b4=build(name,t,h,'B4',2);b5=build(name,t,h,'B5',2)
    assert all(torch.equal(a,b) for a,b in zip(b4.state_dict().values(),b5.state_dict().values()))
    assert sum(p.numel() for p in b4.parameters())==sum(p.numel() for p in b5.parameters())
    test=datasets(name,t,h)[1];batch=next(iter(DataLoader(test,batch_size=4)))
    yh,ch,cf,y,_=batch; assert yh.shape==(4,t,1) and cf.shape==(4,h,2)
    # Inference signature has no future target; changing labels cannot change forecast.
    b5.eval();batch2=(*batch[:3],batch[3]+123456,batch[4])
    with torch.no_grad():
        z1,p1,_=model_forward(b5,batch);z2,p2,_=model_forward(b5,batch2)
    assert torch.equal(p1[-1],p2[-1]);assert p1[-1].shape==(4,h,1)
    b3=build(name,t,h,'B3',2); assert b3.refiners[0] is not b3.refiners[1]
    assert not ({id(p) for p in b3.refiners[0].parameters()} & {id(p) for p in b3.refiners[1].parameters()})
    for v in ('B1','B2','B3','B4','B5'):
        m=build(name,t,h,v,2);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
        frozen={n:p.clone() for n,p in m.ae.named_parameters()}
        for _ in range(3):
            opt.zero_grad();zs,ys,target=model_forward(m,batch);loss=(ys[-1]-target).square().mean() if v=='B1' else latent_loss(zs[-1],m.ae.encode(target));loss.backward();opt.step()
        assert all(p.grad is None and torch.equal(p,frozen[n]) for n,p in m.ae.named_parameters())
    json_write(ROOT/'sanity.json',{'passed':True,'checks':['B1/B2 identical common backbone initialization','B4/B5 identical initial parameters and architecture','no shared B3 block parameters','future label perturbation leaves prediction identical','AE frozen through backward and optimizer','all five variants finite real-data backward']})
    print('MODEL_SANITY_PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sanity',action='store_true');p.add_argument('--dataset');p.add_argument('--horizon',type=int);p.add_argument('--variant');p.add_argument('--k',type=int,default=2);a=p.parse_args()
    if a.sanity:sanity()
    else:train(a.dataset,168 if a.horizon==24 else 720,a.horizon,a.variant,a.k)
