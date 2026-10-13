"""Post-lock diagnostic forecasts and continuous-window block summaries."""
import json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
from ops import *
from decoder_fit import eval_select,nondecoder_hash,tensor_hash

def read_complete():
    matrix=json.loads((ROOT/'outputs/formal_matrix_status.json').read_text())
    assert matrix['all_six_locked'] and matrix['stage']=='complete'
    summaries={}
    for ds in ['NP','PJM']:
        for ep in [1,3,6]:
            path=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'/'complete.json'
            assert path.is_file(),'All six fits must finish before diagnostic evaluation: '+str(path)
            value=json.loads(path.read_text())
            assert value['decoder_epochs']==20 and value['optimizer_steps']==22240
            assert value['protocol_sha256']==sha(ROOT/'configs/protocol.json')
            summaries[(ds,ep)]=value
    hashes_by_dataset={ds:{summaries[(ds,ep)]['decoder_init_tensor_hash'] for ep in [1,3,6]}
                       for ds in ['NP','PJM']}
    assert all(len(v)==1 for v in hashes_by_dataset.values())
    assert (ROOT/'outputs/test_results.json').is_file()
    assert json.loads((ROOT/'outputs/test_results.json').read_text())['failed']==0
    return summaries

def load_candidate(ds,ep,decoder_epoch):
    cfg,_,_,_=config(ds)
    path=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'/f'decoder_epoch{decoder_epoch}.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['dataset']==ds and saved['stage1_epoch']==ep and saved['decoder_epoch']==decoder_epoch
    model=WorldTSModel(cfg);model.load_state_dict(saved['Model'],strict=True)
    model=model.cuda().eval().requires_grad_(False)
    assert nondecoder_hash(model)==saved['frozen_stage1_tensor_hash']
    return model,saved,path

def reconstruction(model,d,origin_range):
    oo=np.arange(origin_range[0],origin_range[1]+1,dtype=np.int64)
    reconstructed=[];actual=[]
    with torch.no_grad():
        for start in range(0,len(oo),32):
            x,cf,y=batch(d,oo[start:start+32]);out=model(x,x_future=cf,future_target=y)
            decoded=WorldTS._decode_model_states(model,out['h_target'],out)
            reconstructed.append(decoded.cpu().numpy());actual.append(y.cpu().numpy())
    return metrics(np.concatenate(reconstructed),np.concatenate(actual))

def paired_blocks(error_candidate,error_reference,ds,ep):
    difference=np.asarray(error_candidate,dtype=np.float64)-np.asarray(error_reference,dtype=np.float64)
    n=len(difference);assert n==1363
    rows=[]
    for block in PROTOCOL['bootstrap_block_lengths']:
        rng=np.random.default_rng(PROTOCOL['bootstrap_seed']+block+(0 if ds=='NP' else 100000)+ep*1000)
        reps=PROTOCOL['bootstrap_repetitions'];take=int(np.ceil(n/block))
        means=np.empty(reps,dtype=np.float64)
        for j in range(reps):
            starts=rng.integers(0,n,size=take)
            indices=np.concatenate([(np.arange(block)+s)%n for s in starts])[:n]
            means[j]=difference[indices].mean()
        simple_blocks=[difference[k:min(n,k+block)].mean() for k in range(0,n,block)]
        rows.append({'dataset':ds,'stage1_epoch':ep,'reference_stage1_epoch':1,
                     'block_length':block,'n_origins':n,'n_nonoverlap_blocks':len(simple_blocks),
                     'observed_delta_mae_candidate_minus_epoch1':float(difference.mean()),
                     'bootstrap_mean':float(means.mean()),
                     'bootstrap_ci025':float(np.quantile(means,.025)),
                     'bootstrap_ci975':float(np.quantile(means,.975)),
                     'fraction_bootstrap_better':float(np.mean(means<0)),
                     'nonoverlap_block_delta_json':json.dumps(simple_blocks),
                     'interpretation':'descriptive only; overlapping rolling forecast windows'})
    return rows

def run():
    completions=read_complete()
    all_results=[];curves=[];blocks=[];pred_store={};paired_time=[]
    for ds in ['NP','PJM']:
        d=data(ds)
        for ep in [1,3,6]:
            directory=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'
            summary=completions[(ds,ep)]
            rows=list(csv.DictReader((directory/'curves.csv').open(newline='',encoding='utf-8')))
            assert len(rows)==21
            curves.extend(rows)
            for label,decoder_epoch in [('best',summary['best_decoder_epoch']),('last',20)]:
                model,saved,path=load_candidate(ds,ep,decoder_epoch)
                select_score,_,_,_=eval_select(model,d,PROTOCOL['V_select_origins'],True)
                assert abs(select_score['selection_mae_batch_mean']-
                           float(rows[decoder_epoch]['val_select_selection_mae_batch_mean']))<1e-5
                diagnostic,pred,true,oo=eval_select(model,d,PROTOCOL['V_diagnostic_origins'],True)
                select_recon_mse,select_recon_mae=reconstruction(model,d,PROTOCOL['V_select_origins'])
                diag_recon_mse,diag_recon_mae=reconstruction(model,d,PROTOCOL['V_diagnostic_origins'])
                assert len(oo)==1363 and np.array_equal(oo,np.arange(40210,41573))
                row={'dataset':ds,'stage1_epoch':ep,'decoder_checkpoint':label,
                     'decoder_epoch':decoder_epoch,'n_optimizer_steps':decoder_epoch*1112,
                     'val_select_forecast_mae':select_score['mae'],
                     'val_select_forecast_mse':select_score['mse'],
                     'val_select_selection_mae_batch_mean':select_score['selection_mae_batch_mean'],
                     'val_diagnostic_forecast_mae':diagnostic['mae'],
                     'val_diagnostic_forecast_mse':diagnostic['mse'],
                     'val_select_reconstruction_mae':select_recon_mae,
                     'val_select_reconstruction_mse':select_recon_mse,
                     'val_diagnostic_reconstruction_mae':diag_recon_mae,
                     'val_diagnostic_reconstruction_mse':diag_recon_mse,
                     'training_mae_last_epoch':float(rows[-1]['train_pred_mae']),
                     'training_objective_last_epoch':float(rows[-1]['train_objective']),
                     'training_seconds':summary['elapsed_seconds'],
                     'peak_cuda_mib':summary['peak_cuda_mib'],
                     'decoder_parameters':sum(p.numel() for p in model.decoder.parameters()),
                     'checkpoint_sha256':sha(path),
                     'decoder_init_tensor_hash':summary['decoder_init_tensor_hash'],
                     'frozen_stage1_tensor_hash':summary['frozen_stage1_tensor_hash'],
                     'data_order_hash_epoch20':rows[-1]['data_order_sha256'],
                     'checkpoint_reloaded':True}
                all_results.append(row)
                if label=='best':
                    pred_store[(ds,ep)]=(pred,true,oo)
                    np.savez_compressed(ROOT/'outputs/checkpoints'/ds/
                                        f'diagnostic_predictions_stage1_epoch{ep}.npz',
                                        pred=pred,true=true,origins=oo)
                print('DIAGNOSTIC_DONE',ds,ep,label,diagnostic['mae'],flush=True)
        reference_pred,reference_true,reference_origins=pred_store[(ds,1)]
        reference_error=np.mean(np.abs(reference_pred.astype('float64')-reference_true.astype('float64')),axis=(1,2))
        for ep in [3,6]:
            pred,true,oo=pred_store[(ds,ep)]
            assert np.array_equal(oo,reference_origins) and np.array_equal(true,reference_true)
            current_error=np.mean(np.abs(pred.astype('float64')-true.astype('float64')),axis=(1,2))
            blocks.extend(paired_blocks(current_error,reference_error,ds,ep))
            for k in range(0,len(oo),360):
                paired_time.append({'dataset':ds,'stage1_epoch':ep,'reference_stage1_epoch':1,
                                    'origin_first':int(oo[k]),'origin_last':int(oo[min(len(oo),k+360)-1]),
                                    'n_origins':len(oo[k:k+360]),
                                    'candidate_mae':float(current_error[k:k+360].mean()),
                                    'reference_mae':float(reference_error[k:k+360].mean()),
                                    'delta_candidate_minus_epoch1':float((current_error-reference_error)[k:k+360].mean())})
    write_csv(ROOT/'outputs/decoder_training_results.csv',all_results)
    write_csv(ROOT/'outputs/decoder_training_curves.csv',curves)
    write_csv(ROOT/'outputs/paired_block_statistics.csv',blocks)
    write_csv(ROOT/'outputs/paired_time_blocks.csv',paired_time)
    write_json(ROOT/'outputs/diagnostic_lock.json',
               {'all_six_fits_completed_before_first_diagnostic':True,
                'selection_locked':{f'{ds}_{ep}':completions[(ds,ep)]['best_decoder_epoch']
                                    for ds in ['NP','PJM'] for ep in [1,3,6]},
                'results_sha256':sha(ROOT/'outputs/decoder_training_results.csv'),
                'diagnostic_is_not_historically_unseen':True})
    print('FINAL_DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':run()
