"""One real TRAIN batch, official source, each changed critical training path."""
import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import *

def main():
    set_seed(2021);data=read_arrays(ROOT/'outputs/dataset',('train',))
    scaler=train_scaler(data['train']);d=rescale_np(data['train'],scaler)
    cfg=json.loads((ROOT/'configs/worldts_neso.json').read_text())
    core,loss_class=import_native(str(ROOT/'upstream/WorldTS/DAG-bench'))
    config=native_config(cfg,336,48);model=core(config).cuda()
    x=torch.from_numpy(d['history'][:16]).cuda();y=torch.from_numpy(d['target'][:16]).cuda();ca=torch.from_numpy(d['cov_a'][:16]).cuda()
    criterion=loss_class(**{k:cfg[k] for k in ['latent_alignment_loss','latent_mse_weight','latent_cosine_weight',
                       'latent_sigreg_weight','latent_detach_target','sigreg_std_weight','sigreg_cov_weight','sigreg_eps']}).cuda()
    checks={}
    model.configure_for_stage1();model.train();decoder_before=tensor_hash(model,'decoder')
    out=model(input=x,future_target=y,x_future=ca);loss=stage1_objective(out,criterion);loss.backward()
    for prefix in ['target_encoder.','condition_encoder.','predictor.']:
        grads=[p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.norm()) for g in grads)>0
        checks[prefix+'finite_nonzero_gradient']=True
    torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['stage1_lr']).step()
    assert tensor_hash(model,'decoder')==decoder_before;checks['stage1_decoder_frozen']=True
    model.zero_grad(set_to_none=True);model.configure_for_stage2();model.train();model.set_stage1_eval()
    frozen=tensor_hash(model,'stage1');before=tensor_hash(model,'decoder')
    with torch.no_grad():out=model(input=x,future_target=y,x_future=ca)
    value=stage2_objective(model,out,y);value.backward()
    torch.optim.Adam(model.decoder.parameters(),lr=cfg['stage2_lr']).step()
    assert tensor_hash(model,'stage1')==frozen and tensor_hash(model,'decoder')!=before
    checks['stage2_updates_only_decoder']=True
    model.eval()
    with torch.no_grad():
        oa=model(input=x,future_target=None,x_future=ca);p=pred_decode(model,oa)
        ob=model(input=x,future_target=y.flip(0),x_future=ca)
        assert torch.equal(oa['h_pred'],ob['h_pred']) and p.shape==y.shape and torch.isfinite(p).all()
        repeated=pred_decode(model,model(input=x,future_target=None,x_future=ca));assert torch.equal(p,repeated)
    checks['label_foil_repeat_shape_finite']=True
    saved={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    fresh=core(config).cuda().eval();fresh.load_state_dict(saved,strict=True);fresh.configure_for_stage2()
    assert tensor_hash(model)==tensor_hash(fresh)
    with torch.no_grad():reload=pred_decode(fresh,fresh(input=x,future_target=None,x_future=ca))
    reload_diff=float((p-reload).abs().max());print('FRESH_RESTORE_DIFFERENCE',reload_diff,'bitwise_equal',torch.equal(p,reload),flush=True)
    assert torch.allclose(p,reload,rtol=1e-5,atol=1e-6)
    checks['strict_fresh_restore_prediction_max_diff']=reload_diff
    checks['strict_state_dict_tensor_hash_equal']=True
    checks['restore_prediction_tolerance']='FP32 allclose rtol1e-5 atol1e-6; observed max recorded, never silently rounded'
    report={'evidence_scope':'real TRAIN batch smoke; not formal scores','n_samples':len(x),
            'class':str(core),'stage1_loss':float(loss.detach()),'stage2_loss':float(value.detach()),
            'checks':checks,'parameters':sum(q.numel() for q in model.parameters()),
            'predictive_output_shape':list(p.shape),'source_commit':subprocess.check_output(['git','-C',str(ROOT/'upstream/WorldTS'),'rev-parse','HEAD'],text=True).strip()}
    (ROOT/'outputs/native_smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REAL_NATIVE_SMOKE_PASS',json.dumps(report),flush=True)
if __name__=='__main__':main()
