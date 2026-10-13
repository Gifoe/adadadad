import json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import *
set_seed(2021);cfg=json.loads((ROOT/'configs/worldts_neso.json').read_text());out=ROOT/'outputs/model'
scaler=json.loads((out/'train_scaler.json').read_text());d=rescale_np(read_arrays(ROOT/'outputs/dataset',('val',))['val'],scaler)
core,_=import_native(str(ROOT/'upstream/WorldTS/DAG-bench'));model=core(native_config(cfg,336,48)).cuda().eval();model.configure_for_stage2()
model.load_state_dict(torch.load(out/'stage2_best.pt',weights_only=False,map_location='cuda')['model_state'],strict=True)
hpred=[];htarget=[];maximum_foil=0.;max_reload=0.;cursor=0;max_saved_val=0.
with np.load(out/'validation_predictions.npz') as z:saved_val=z['pred_old_mw']
fresh=core(native_config(cfg,336,48)).cuda().eval();fresh.configure_for_stage2();fresh.load_state_dict(torch.load(out/'stage2_best.pt',weights_only=False,map_location='cuda')['model_state'],strict=True)
assert tensor_hash(model)==tensor_hash(fresh)
for x,y,ca,cb,mask in as_loader(d,16,False,2021):
    x,y,ca=x.cuda(),y.cuda(),ca.cuda()
    with torch.no_grad():
        plain=model(input=x,future_target=None,x_future=ca);labelled=model(input=x,future_target=y,x_future=ca)
        p=pred_decode(model,plain);r=pred_decode(fresh,fresh(input=x,future_target=None,x_future=ca))
    assert p.shape==(len(x),48,1) and torch.isfinite(p).all()
    maximum_foil=max(maximum_foil,float((plain['h_pred']-labelled['h_pred']).abs().max()))
    max_reload=max(max_reload,float((p-r).abs().max()))
    assert torch.allclose(p,r,rtol=1e-5,atol=1e-6)
    mw=p.cpu().numpy()*scaler['std'][0]+scaler['mean'][0]
    max_saved_val=max(max_saved_val,float(np.abs(mw-saved_val[cursor:cursor+len(x)]).max()));cursor+=len(x)
    hpred.append(plain['h_pred'].cpu().numpy());htarget.append(labelled['h_target'].cpu().numpy())
hp=np.concatenate(hpred);ht=np.concatenate(htarget)
assert maximum_foil==0
stats={'h_pred_shape':list(hp.shape),'h_pred_finite':bool(np.isfinite(hp).all()),'h_target_finite':bool(np.isfinite(ht).all()),
       'mean_h_pred_feature_std_across_windows_and_time':float(hp.reshape(-1,hp.shape[-1]).std(axis=0).mean()),
       'mean_h_target_feature_std_across_windows_and_time':float(ht.reshape(-1,ht.shape[-1]).std(axis=0).mean()),
       'mean_h_pred_temporal_std':float(hp.std(axis=1).mean()),'latent_mae':float(np.abs(hp-ht).mean()),
       'future_label_foil_h_pred_max_difference':maximum_foil,'fresh_checkpoint_restore_prediction_max_difference':max_reload,
       'fresh_restored_saved_validation_prediction_max_diff_mw':max_saved_val,
       'stage2_selected_stage1_tensor_hash':tensor_hash(model,'stage1')}
stage1=core(native_config(cfg,336,48)).cuda();stage1.load_state_dict(torch.load(out/'stage1_best.pt',weights_only=False,map_location='cuda')['model_state'],strict=True)
assert tensor_hash(stage1,'stage1')==stats['stage2_selected_stage1_tensor_hash']
stats['selected_stage1_preserved_in_stage2']=True
assert stats['h_pred_finite'] and stats['h_target_finite'] and stats['mean_h_pred_feature_std_across_windows_and_time']>1e-8
assert max_saved_val<.1,'Fresh model failed to reproduce saved Validation predictions within0.1MW'
(ROOT/'outputs/selected_model_diagnostics.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print('SELECTED_MODEL_DIAGNOSTICS_PASS',json.dumps(stats),flush=True)
