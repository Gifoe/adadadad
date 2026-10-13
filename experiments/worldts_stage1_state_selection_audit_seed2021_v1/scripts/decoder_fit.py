"""Matched twenty-epoch decoder fits for the fixed Stage-1 epoch matrix."""
import argparse, hashlib, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
from ops import *
from torch.utils.data import DataLoader


def tensor_hash(state):
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def nondecoder_hash(model):
    return tensor_hash({k: v for k, v in model.state_dict().items() if not k.startswith('decoder.')})


def get_model(ds, ep):
    cfg, _, _, _ = config(ds)
    saved = torch.load(ROOT / 'outputs/checkpoints' / ds / f'replay_stage1_epoch{ep}.pt',
                       map_location='cpu', weights_only=False)
    assert saved['epoch'] == ep and saved['dataset'] == ds and saved['identity'].startswith('raw completed epoch')
    model = WorldTSModel(cfg)
    model.load_state_dict(saved['Model'], strict=True)
    common = torch.load(ROOT / 'outputs/checkpoints' / ds / 'untrained_decoder.pt',
                        map_location='cpu', weights_only=False)['Decoder']
    model.decoder.load_state_dict(common, strict=True)
    model = model.cuda()
    model.configure_for_stage2()
    model.set_stage1_eval()
    assert all(p.requires_grad == k.startswith('decoder.') for k,p in model.named_parameters())
    assert not any(m.training for m in model._stage1_modules())
    return model,cfg,tensor_hash(common),sha(ROOT / 'outputs/checkpoints' / ds / f'replay_stage1_epoch{ep}.pt')


def eval_select(model, d, origin_range, return_arrays=False):
    model.eval()
    origins_fixed = np.arange(origin_range[0], origin_range[1] + 1, dtype=np.int64)
    p_list=[]; y_list=[]; batch_mae=[]
    with torch.no_grad():
        for start in range(0, len(origins_fixed), 32):
            x,cf,y=batch(d,origins_fixed[start:start+32])
            p=model.forecast(x,x_future=cf)
            assert p.shape==y.shape and bool(torch.isfinite(p).all())
            p_list.append(p.cpu().numpy()); y_list.append(y.cpu().numpy())
            batch_mae.append(float(torch.nn.functional.l1_loss(p,y).cpu()))
    pred=np.concatenate(p_list);true=np.concatenate(y_list)
    mse,mae=metrics(pred,true)
    result={'mse':mse,'mae':mae,'selection_mae_batch_mean':float(np.mean(batch_mae)),
            'n_origins':len(origins_fixed),'n_batches':len(batch_mae)}
    return (result,pred,true,origins_fixed) if return_arrays else result


def save_epoch(path, ds, stage1_epoch, decoder_epoch, model, optimizer, generator,
               records, best_epoch, best_value, frozen_hash, origin_hash, elapsed):
    payload={'dataset':ds,'stage1_epoch':stage1_epoch,'decoder_epoch':decoder_epoch,
             'Model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             'optimizer':optimizer.state_dict(),'rng':rng(),
             'data_order_rng':generator.get_state(),'records':list(records),
             'best_decoder_epoch':best_epoch,'best_selection_value':best_value,
             'frozen_stage1_tensor_hash':frozen_hash,'origin_order_hash':origin_hash,
             'elapsed_seconds':elapsed,'protocol_sha256':sha(ROOT/'configs/protocol.json')}
    temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);temporary.replace(path)


def execute(ds, ep):
    destination=ROOT/'outputs/checkpoints'/ds/f'decoder_from_stage1_epoch{ep}'
    destination.mkdir(parents=True,exist_ok=True)
    completion=destination/'complete.json'
    if completion.is_file():
        info=json.loads(completion.read_text());assert info['decoder_epochs']==20
        print('DECODER_ALREADY_COMPLETE',ds,ep,flush=True);return
    model,cfg,decoder_init_hash,stage1_file_sha=get_model(ds,ep)
    frozen_hash=nondecoder_hash(model)
    d=data(ds)
    n=len(origins(d,'train'))
    assert n==35611 and len(origins(d,'val'))==4883
    selector=PROTOCOL['V_select_origins']
    assert selector==[36690,38051]
    kwargs=vars(cfg).copy();kwargs.pop('criterion',None)
    kwargs['training_stage']='stage2_decoder_mix'
    wrapper=WorldTS(**kwargs);wrapper.model=model
    optimizer=wrapper._init_optimizer(lr=float(cfg.stage2_lr))
    lr_config=wrapper._stage_lr_config(float(cfg.stage2_lr))
    generator=torch.Generator();generator.manual_seed(2021)
    seed(2021)
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter(); records=[]; best_epoch=None;best_value=float('inf');first_epoch=0
    latest=destination/'latest.pt'
    if latest.is_file():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['dataset']==ds and saved['stage1_epoch']==ep and saved['protocol_sha256']==sha(ROOT/'configs/protocol.json')
        model.load_state_dict(saved['Model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['data_order_rng']);restore_rng(saved['rng'])
        records=saved['records'];best_epoch=saved['best_decoder_epoch'];best_value=saved['best_selection_value']
        first_epoch=saved['decoder_epoch']+1
        started=time.perf_counter()-saved['elapsed_seconds']
        assert nondecoder_hash(model)==frozen_hash
        print('DECODER_RESUME',ds,ep,first_epoch,flush=True)
    else:
        initial=eval_select(model,d,selector)
        best_epoch=0;best_value=initial['selection_mae_batch_mean']
        records.append({'dataset':ds,'stage1_epoch':ep,'decoder_epoch':0,
                        **{'val_select_'+k:v for k,v in initial.items()},
                        'train_objective':None,'train_pred_mae':None,'train_recon_mae':None,
                        'optimizer_steps':0,'learning_rate':float(cfg.stage2_lr),
                        'decoder_init_tensor_hash':decoder_init_hash,
                        'frozen_stage1_tensor_hash':frozen_hash,
                        'elapsed_seconds':time.perf_counter()-started})
        save_epoch(destination/'decoder_epoch0.pt',ds,ep,0,model,optimizer,generator,
                   records,best_epoch,best_value,frozen_hash,hashlib.sha256(b'').hexdigest(),time.perf_counter()-started)
        save_epoch(latest,ds,ep,0,model,optimizer,generator,records,best_epoch,best_value,
                   frozen_hash,hashlib.sha256(b'').hexdigest(),time.perf_counter()-started)
        first_epoch=1
    assert first_epoch<=20
    order_cumulative=hashlib.sha256()
    if first_epoch>1:
        # The saved epoch's order digest documents the prefix. Per-epoch hashes
        # permit exact comparison among the three candidate states.
        order_cumulative.update(str(first_epoch-1).encode())
    for decoder_epoch in range(first_epoch,21):
        model.train();model.set_stage1_eval()
        loader=DataLoader(torch.arange(n,dtype=torch.long),batch_size=32,shuffle=True,
                          drop_last=True,num_workers=0,generator=generator)
        train_total=[];train_pred=[];train_recon=[]; order=hashlib.sha256()
        for index in loader:
            origin=720+index.numpy()
            order.update(origin.astype('<i8').tobytes())
            x,cf,y=batch(d,origin)
            target=torch.cat([y,cf],dim=-1)
            optimizer.zero_grad(set_to_none=True)
            out=wrapper._process(x,target,None,None,cf)
            yp=out['output']
            main=torch.nn.functional.l1_loss(yp,y)
            loss=main+out['additional_loss']
            assert torch.isfinite(loss)
            loss.backward();optimizer.step()
            train_total.append(float(loss.detach().cpu()))
            train_pred.append(float(main.detach().cpu()))
            train_recon.append(float(out['decoder_warm_loss'].detach().cpu()))
        assert len(train_total)==1112 and nondecoder_hash(model)==frozen_hash
        selection=eval_select(model,d,selector)
        value=selection['selection_mae_batch_mean']
        if value<best_value:
            best_value=value;best_epoch=decoder_epoch
        lr_used=float(optimizer.param_groups[0]['lr'])
        wrapper._adjust_lr(optimizer,decoder_epoch,lr_config)
        row={'dataset':ds,'stage1_epoch':ep,'decoder_epoch':decoder_epoch,
             **{'val_select_'+k:v for k,v in selection.items()},
             'train_objective':float(np.mean(train_total)),
             'train_pred_mae':float(np.mean(train_pred)),
             'train_recon_mae':float(np.mean(train_recon)),
             'optimizer_steps':decoder_epoch*1112,'learning_rate':lr_used,
             'decoder_init_tensor_hash':decoder_init_hash,
             'frozen_stage1_tensor_hash':frozen_hash,
             'data_order_sha256':order.hexdigest(),
             'best_decoder_epoch_to_date':best_epoch,
             'elapsed_seconds':time.perf_counter()-started}
        records.append(row)
        save_epoch(destination/f'decoder_epoch{decoder_epoch}.pt',ds,ep,decoder_epoch,
                   model,optimizer,generator,records,best_epoch,best_value,frozen_hash,
                   order.hexdigest(),row['elapsed_seconds'])
        save_epoch(latest,ds,ep,decoder_epoch,model,optimizer,generator,records,best_epoch,
                   best_value,frozen_hash,order.hexdigest(),row['elapsed_seconds'])
        write_csv(destination/'curves.csv',records)
        print('DECODER_EPOCH_DONE',ds,'stage1',ep,'decoder',decoder_epoch,
              'train',row['train_objective'],'val',value,'best',best_epoch,
              'order',order.hexdigest(),flush=True)
    assert len(records)==21 and records[-1]['optimizer_steps']==22240
    if ep in (1,3,6):
        assert all(row['decoder_init_tensor_hash']==decoder_init_hash for row in records)
    summary={'dataset':ds,'stage1_epoch':ep,'decoder_epochs':20,'best_decoder_epoch':best_epoch,
             'best_val_select_mae_batch_mean':best_value,'last_val_select_mae_batch_mean':records[-1]['val_select_selection_mae_batch_mean'],
             'decoder_init_tensor_hash':decoder_init_hash,'frozen_stage1_tensor_hash':frozen_hash,
             'stage1_file_sha256':stage1_file_sha,'optimizer_steps':22240,
             'elapsed_seconds':time.perf_counter()-started,
             'peak_cuda_mib':torch.cuda.max_memory_allocated()/1048576,
             'protocol_sha256':sha(ROOT/'configs/protocol.json')}
    write_json(completion,summary)
    print('DECODER_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['NP','PJM'],required=True)
    parser.add_argument('--stage1-epoch',type=int,choices=[1,3,6],required=True)
    args=parser.parse_args();execute(args.dataset,args.stage1_epoch)
