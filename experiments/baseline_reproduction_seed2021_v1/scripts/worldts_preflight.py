"""Meaningful checks on the actual official WorldTS modules and objective."""
from common import *
from worldts_config import *
import argparse
sys.path.insert(0,str(ROOT/'upstream/WorldTS/DAG-bench'))
from ts_benchmark.baselines.worldts.worldts import WorldTS
from ts_benchmark.baselines.worldts.models.worldts_model import WorldTSModel
from ts_benchmark.baselines.worldts.losses.latent_state_loss import LatentStateLoss
def test(dataset,horizon):
 _,_,hp=published_config(dataset,horizon);wrapper=WorldTS(**hp);cfg=wrapper.config;cfg.series_dim=1;cfg.input_dim=3;cfg.output_dim=1;cfg.future_x_dim=2;cfg.pred_len=horizon
 torch.manual_seed(2021);model=WorldTSModel(cfg).float();model.eval();b=2;t=hp['seq_len'];h=horizon;d=hp['latent_dim'];se=hp['state_patch_len'];sd=hp['decoder_patch_len']
 x=torch.randn(b,t,3);future=torch.randn(b,h,1);cov=torch.randn(b,h,2)
 with torch.no_grad():
  a=model(x,future_target=future,x_future=cov);changed=model(x,future_target=future*13+7,x_future=cov);inference=model(x,x_future=cov)
  assert torch.equal(a['h_hist'],changed['h_hist']);assert torch.equal(a['h_pred'],changed['h_pred']);assert torch.equal(a['h_pred'],inference['h_pred'])
  enc=model.target_encoder;raw=torch.randn(b,t+h,1);encoded=enc(raw);perturbed=raw.clone();perturbed[:,t:]+=19;assert torch.equal(encoded[:,:t],enc(perturbed)[:,:t])
  assert torch.allclose(model.encode_future_target(raw[:,t:],raw[:,:t]),encoded[:,t:],atol=1e-6)
  enc_cov=model.condition_encoder;rawcov=torch.randn(b,t+h,2);assert torch.allclose(model.encode_future_condition(rawcov[:,t:],rawcov[:,:t]),enc_cov(rawcov)[:,t:],atol=1e-6)
  assert a['h_hist'].shape==(b,t,d) and a['h_pred'].shape==(b,h,d) and a['h_target'].shape==(b,h,d)
 # Only regularization may update the target branch.
 hp1=torch.randn(b,h,d,requires_grad=True);ht=torch.randn(b,h,d,requires_grad=True)
 match=LatentStateLoss(latent_mse_weight=10,latent_cosine_weight=15,latent_sigreg_weight=0,latent_alignment_loss='mae',latent_detach_target=True)
 match(hp1,ht)['loss'].backward();assert hp1.grad is not None and ht.grad is None
 hp1.grad=None;criterion=wrapper.latent_state_loss;values=criterion(hp1,ht);values['loss'].backward();assert ht.grad is not None and ht.grad.abs().sum()>0
 manualcos=1-torch.nn.functional.cosine_similarity(hp1.flatten(1),ht.detach().flatten(1),dim=-1).mean();assert torch.allclose(values['cosine_loss'],manualcos.detach())
 model.configure_for_stage1();assert not any(p.requires_grad for p in model.decoder.parameters());assert all(p.requires_grad for m in model._stage1_modules() for p in m.parameters())
 model.configure_for_stage2();assert all(not p.requires_grad for m in model._stage1_modules() for p in m.parameters());assert all(p.requires_grad for p in model.decoder.parameters())
 before=[tensor_sha(m) for m in model._stage1_modules()];model.set_stage1_eval();opt=torch.optim.Adam(model.decoder.parameters(),lr=hp['stage2_lr'])
 with torch.no_grad():a=model(x,future_target=future,x_future=cov)
 y=model.decode(a['h_pred'],target_revin_stats=a['target_revin_stats'],history_tokens=a['h_hist']);recon=model.decode(a['h_target'],target_revin_stats=a['target_revin_stats'],history_tokens=a['h_hist']);loss=(y-future).abs().mean()+hp['decoder_warmup_weight']*(recon-future).abs().mean();loss.backward();opt.step();assert before==[tensor_sha(m) for m in model._stage1_modules()]
 predictor=model.predictor.patch_predictor
 result={'dataset':dataset,'horizon':horizon,'passed':True,'source_commit':COMMITS['WorldTS'],'future_label_invariance':True,'history_encoder_causal':True,'future_target_historical_context':True,'future_covariate_historical_context':True,'latent_target_stopgrad_for_match':True,'vicreg_target_branch_receives_gradient':True,'trajectory_cosine_verified':True,'stage1_decoder_frozen':True,'stage2_encoder_predictor_unchanged_after_step':True,'encoder_patch':se,'decoder_patch':sd,'latent_dim':d,'predictor':{'input':[b,t,d],'patch_projection':[hp['predictor_patch_len'],hp['predictor_d_model']],'patch_count':predictor.patch_num,'embedded_shape':[b*d,predictor.patch_num,hp['predictor_d_model']],'attention_heads':hp['predictor_n_heads'],'encoder_layers':hp['predictor_e_layers'],'ffn_width':hp['predictor_d_ff'],'sinusoidal_position':True,'head_projection':[hp['predictor_d_model']*predictor.patch_num,h],'output':[b,h,d]},'latent_regularizer_type':hp.get('latent_regularizer_type','legacy_vc'),'configuration_source':'OFFICIAL_SOURCE','test_device':'CPU; same module equations'}
 write_json(ROOT/'reports'/f'worldts_preflight_{dataset}_{horizon}.json',result);print('WORLDTS_PREFLIGHT_PASSED',json.dumps(result),flush=True)
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['NP','PJM'],required=True);p.add_argument('--horizon',type=int,choices=[24,360],required=True);a=p.parse_args();test(a.dataset,a.horizon)
