"""Numerical and protocol checks for the fixed six-cell experiment."""
import hashlib, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from ops import *
from decoder_fit import tensor_hash,nondecoder_hash,get_model,eval_select

checks=[]
def check(name,condition,detail=''):
    assert bool(condition),name+': '+str(detail)
    checks.append({'check':name,'status':'PASS','detail':str(detail)})
    print('PASS '+name+(' '+str(detail) if detail else ''),flush=True)

def audit():
    all_states={};decoder_hashes={}
    for ds in ['NP','PJM']:
        cfg,dr,_,_=config(ds)
        replay=[]
        for ep in range(1,7):
            saved=torch.load(ROOT/'outputs/checkpoints'/ds/f'replay_stage1_epoch{ep}.pt',
                             map_location='cpu',weights_only=False)
            model=WorldTSModel(cfg);model.load_state_dict(saved['Model'],strict=True)
            check(f'{ds}_snapshot_restore_ep{ep}',saved['epoch']==ep and
                  saved['dataset']==ds and saved['identity'].startswith('raw completed epoch') and
                  saved['optimizer'] and all(k in saved['rng'] for k in ['python','numpy','torch','cuda']) and
                  saved['source_sha256']==sha(ROOT/'upstream/WorldTS/DAG-bench/ts_benchmark/baselines/worldts/worldts.py'))
            replay.append(saved)
        all_states[ds]=replay
        epoch_keys=[]
        for ep in [1,3,6]:
            state=replay[ep-1]['Model']
            epoch_keys.append((tensor_hash({k:v for k,v in state.items() if k.startswith('target_encoder.')}),
                               tensor_hash({k:v for k,v in state.items() if k.startswith('predictor.')}),
                               tensor_hash({k:v for k,v in state.items() if k.startswith('condition_encoder.')})))
        check(f'{ds}_encoder_predictor_condition_distinct',len(set(x[0] for x in epoch_keys))==3 and
              len(set(x[1] for x in epoch_keys))==3 and len(set(x[2] for x in epoch_keys))==3)
        original=torch.load(dr/'resume_stage1_latent.pt',map_location='cpu',weights_only=False)
        original_best=torch.load(dr/'stages/stage1_latent.pt',map_location='cpu',weights_only=False)['Model']
        check(f'{ds}_best_restore_not_epoch6',tensor_hash(original_best)==tensor_hash(original['best_state']) and
              tensor_hash(original_best)!=tensor_hash(original['model']) and
              tensor_hash(replay[0]['Model'])!=tensor_hash(replay[5]['Model']) and
              replay[5]['run_records'][-1]['epoch']==6)
        common=torch.load(ROOT/'outputs/checkpoints'/ds/'untrained_decoder.pt',
                          map_location='cpu',weights_only=False)['Decoder']
        decoder_hashes[ds]=tensor_hash(common)
        in_replay=[tensor_hash({k[len('decoder.'):]:v for k,v in p['Model'].items()
                                if k.startswith('decoder.')}) for p in replay]
        check(f'{ds}_decoder_frozen_stage1',len(set(in_replay))==1 and in_replay[0]==decoder_hashes[ds])
        original_train=torch.load(ROOT/'outputs/checkpoints'/ds/'stage1_initial.pt',
                                  map_location='cpu',weights_only=False)
        check(f'{ds}_original_untrained_decoder_match',original_train['decoder_tensor_hash']==decoder_hashes[ds])
        d=data(ds);tr=origins(d,'train');va=origins(d,'val');select=PROTOCOL['V_select_origins'];diag=PROTOCOL['V_diagnostic_origins']
        check(f'{ds}_common_origins',len(tr)==35611 and len(va)==4883 and
              tr[0]==720 and tr[-1]==36330 and va[0]==36690 and va[-1]==41572 and
              select==[36690,38051] and diag==[40210,41572])
        check(f'{ds}_raw_support_embargo',diag[0]-720-(select[1]+359)-1==1079 and
              diag[1]+359==41931 and diag[0]>=36690)
        model,cfg,init_hash,_=get_model(ds,1)
        check(f'{ds}_common_decoder_init',init_hash==decoder_hashes[ds])
        origin=np.asarray([int(tr[0]),int(tr[100]),int(tr[300])])
        x,cf,y=batch(d,origin)
        model.eval()
        with torch.no_grad():
            deployed=model.forecast(x,x_future=cf)
            out1=model(x,x_future=cf,future_target=y)
            foil=torch.flip(y,(0,))
            out2=model(x,x_future=cf,future_target=foil)
            reconstructed=WorldTS._decode_model_states(model,out1['h_pred'],out1)
            repeated=model.forecast(x,x_future=cf)
            stats=out1['target_revin_stats']
            hist=x[:,:,:1]
            expected_mean=hist.mean(1,keepdim=True)
            expected_std=torch.sqrt(hist.var(1,keepdim=True,unbiased=False)+cfg.revin_eps)
        check(f'{ds}_future_label_foil',torch.equal(out1['h_pred'],out2['h_pred']) and
              torch.equal(deployed,reconstructed))
        check(f'{ds}_revin_inverse',stats is not None and torch.allclose(stats['mean'],expected_mean,atol=1e-6) and
              torch.allclose(stats['stdev'],expected_std,atol=1e-6) and torch.equal(deployed,reconstructed))
        check(f'{ds}_prediction_repeat',torch.equal(deployed,repeated))
        # One official Stage-2 objective update must leave every Stage-1 tensor intact.
        model.train();model.set_stage1_eval()
        frozen_before=nondecoder_hash(model);decoder_before=tensor_hash(model.decoder.state_dict())
        wrapper=WorldTS(**{k:v for k,v in vars(cfg).items() if k!='criterion'});wrapper.model=model
        wrapper.config.training_stage='stage2_decoder_mix'
        target=torch.cat([y,cf],dim=-1)
        out=wrapper._process(x,target,None,None,cf)
        loss=torch.nn.functional.l1_loss(out['output'],y)+out['additional_loss']
        manual=torch.nn.functional.l1_loss(out['output'],y)+.3*out['decoder_warm_loss']
        check(f'{ds}_official_decoder_objective',torch.allclose(loss,manual,atol=1e-6))
        optimizer=torch.optim.Adam(model.decoder.parameters(),lr=float(cfg.stage2_lr));optimizer.zero_grad()
        loss.backward();optimizer.step()
        check(f'{ds}_stage2_updates_only_decoder',nondecoder_hash(model)==frozen_before and
              tensor_hash(model.decoder.state_dict())!=decoder_before and
              all(p.requires_grad==k.startswith('decoder.') for k,p in model.named_parameters()))
        # The primary estimator is a mean of fixed batches; all permutations keep windows.
        for partition_seed in PROTOCOL['partition_seeds']:
            shuffled=np.random.default_rng(partition_seed).permutation(va)
            check(f'{ds}_partition_{partition_seed}',len(shuffled)==len(va) and
                  np.array_equal(np.sort(shuffled),va))
        sample=np.asarray([0.1,-0.2,0.3,0.4],dtype='float32')
        target_values=np.asarray([0.0,-0.1,0.2,0.8],dtype='float32')
        mse,mae=metrics(sample,target_values)
        check(f'{ds}_independent_forecast_metrics',abs(mae-.175)<1e-6 and
              abs(mse-.0475)<1e-6)
        for ep in [1,3,6]:
            p=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'/'complete.json'
            if not p.is_file():continue
            info=json.loads(p.read_text())
            for decoder_epoch in [0,info['best_decoder_epoch'],20]:
                checkpoint=torch.load(p.parent/f'decoder_epoch{decoder_epoch}.pt',
                                      map_location='cpu',weights_only=False)
                restored=WorldTSModel(cfg);restored.load_state_dict(checkpoint['Model'],strict=True)
                check(f'{ds}_decoder_checkpoint_{ep}_{decoder_epoch}',
                      checkpoint['decoder_epoch']==decoder_epoch and
                      checkpoint['stage1_epoch']==ep and checkpoint['optimizer'] is not None and
                      checkpoint['protocol_sha256']==sha(ROOT/'configs/protocol.json'))
    write_json(ROOT/'outputs/test_results.json',{'checks':checks,'passed':len(checks),'failed':0})
    print('ALL_CHECKS_PASS',len(checks),flush=True)

if __name__=='__main__':audit()
