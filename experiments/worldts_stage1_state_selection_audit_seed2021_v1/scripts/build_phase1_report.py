"""Build source-backed preflight and fixed Stage-1 audit reports."""
import csv, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'

def read_csv(path):
    with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def f(v,n=4):return f'{float(v):.{n}f}'

def run():
    inventory=json.loads((OUT/'checkpoint_hashes.json').read_text())
    checks=json.loads((OUT/'test_results.json').read_text())
    assert checks['failed']==0 and checks['passed']>=48
    pre=['# Phase 0 — Checkpoint provenance and preflight','',
         'The original baseline saves the restored best Stage-1 weights in `stages/stage1_latent.pt`. '
         'The raw model at the end of epoch 6 is in `resume_stage1_latent.pt`. '
         'The historical epoch-3 file was absent, so both tasks were replayed for six full epochs '
         'with the pinned upstream method. The only runtime method changes recorded in '
         '`stage1_instrumentation.json` add snapshot/logging hooks and prevent early termination before epoch 6.','',
         '| Dataset | Original best epoch | Original raw epoch 6 | Replayed epoch 1/6 tensors identical | Replay validation curve identical | Untrained Decoder frozen |',
         '|---|---:|---:|---|---|---|']
    for ds in ['NP','PJM']:
        task=next(x for x in inventory['tasks'] if x['dataset']==ds)
        hist=json.loads((OUT/'checkpoints'/ds/'stage1_history.json').read_text())
        assert len(hist)==6 and len(task['history'])==6
        identical=all(abs(float(a['validation_loss'])-float(b['validation_official']))<1e-9
                      for a,b in zip(task['history'],hist))
        records=[x for x in inventory['inventory'] if x['dataset']==ds and x['status']=='RECOVERED_ORIGINAL']
        same=all(x['tensor_hash']==hist[x['epoch']-1]['raw_model_tensor_hash'] for x in records)
        assert identical and same
        init=hist[0]['epoch'];best=min(hist,key=lambda x:x['validation_official'])['epoch']
        assert init==1 and best==1
        pre.append(f'| {ds}-H360 | {best} | 6 | {same} | {identical} | PASS |')
    pre+=['',
          'The replay preserved Adam, type-3 schedule, batch 32, shuffled training and full validation, '
          'the original model and loss, seed 2021, and the original scaler fit on Train only. '
          'The six raw snapshots each contain the complete model plus optimizer, scheduler LR state, '
          'Python/NumPy/Torch/CUDA RNG states, prior best state and validation history. '
          'The saved identity explicitly says raw completed epoch before best restoration. '
          'All original source files were checked against 312 SHA-256 manifest entries.','',
          f'Preformal checks: **{checks["passed"]} passed, {checks["failed"]} failed**. '
          'They cover strict snapshot restore, nonidentical encoder/predictor weights, raw-versus-restored '
          'identity, frozen/common Decoder initialization, one-step Decoder-only gradients, '
          'future-label foil, origin equality and 1,079-hour raw-support embargo, RevIN inversion, '
          'repeated predictions, identical shuffled window sets, and independent MAE/MSE arithmetic.','',
          'Protected original data/config/checkpoint/source SHA-256 hashes are recorded in '
          '`checkpoint_hashes.json`; all new checkpoint files remain under the experiment directory. '
          'No Test forecast was evaluated.']
    (OUT/'PREFLIGHT_REPORT.md').write_text('\n'.join(pre)+'\n',encoding='utf-8')
    metrics=[];partitions=[];geometry=[];align=[]
    for ds in ['NP','PJM']:
        metrics+=read_csv(OUT/f'stage1_epoch_metrics_{ds}.csv')
        partitions+=read_csv(OUT/f'batch_partition_sensitivity_{ds}.csv')
        geometry+=read_csv(OUT/f'latent_geometry_{ds}.csv')
        align+=read_csv(OUT/f'latent_geometry_alignment_{ds}.csv')
    write_csv(OUT/'stage1_epoch_metrics.csv',metrics)
    write_csv(OUT/'batch_partition_sensitivity.csv',partitions)
    write_csv(OUT/'latent_geometry.csv',geometry)
    write_csv(OUT/'latent_geometry_alignment.csv',align)
    lines=['# Phase 1 — Why Stage 1 peaks at epoch 1','',
           'Both tasks use fixed Train and Validation origins, `eval()`, 32-window batches and '
           'the same original latent objective. The Train Probe is 4,883 evenly spaced Train origins; '
           'the Validation set is all 4,883 original Validation origins. '
           'The gap below is their matched snapshot evaluation difference. '
           'The historical online Train loss is displayed separately and never used as that gap.','']
    for ds in ['NP','PJM']:
        val=[r for r in metrics if r['dataset']==ds and r['domain']=='validation_chronological']
        train={int(r['stage1_epoch']):r for r in metrics if r['dataset']==ds and r['domain']=='train_probe_chronological'}
        hist=json.loads((OUT/'checkpoints'/ds/'stage1_history.json').read_text())
        val.sort(key=lambda r:int(r['stage1_epoch']))
        assert len(val)==6
        lines += [f'## {ds}-H360','',
                  '| Stage-1 epoch | Online Train loss | Fixed Train Probe | Fixed Validation | Matched gap | 10×latent L1 | 15×trajectory cosine | 0.1×regularizer |',
                  '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for h,row in zip(hist,val):
            ep=int(row['stage1_epoch']);assert h['epoch']==ep
            lines.append(f'| {ep} | {f(h["train_online"]["additional_loss"])} | '
                         f'{f(train[ep]["loss"])} | {f(row["loss"])} | '
                         f'{f(row["matched_train_probe_gap"])} | '
                         f'{f(10*float(row["mse_loss"]))} | '
                         f'{f(15*float(row["cosine_loss"]))} | '
                         f'{f(.1*float(row["sigreg_loss"]))} |')
        first,last=val[0],val[-1]
        change=float(last['loss'])-float(first['loss'])
        l1=10*(float(last['mse_loss'])-float(first['mse_loss']))
        cos=15*(float(last['cosine_loss'])-float(first['cosine_loss']))
        reg=.1*(float(last['sigreg_loss'])-float(first['sigreg_loss']))
        assert abs(change-(l1+cos+reg))<5e-5
        lines += ['',f'From epoch 1 to 6, Validation objective rises **{change:.4f}**: '
                  f'L1 contributes {l1:.4f}, trajectory cosine {cos:.4f}, '
                  f'and the batch regularizer {reg:.4f}. '
                  f'The matched Train–Validation gap increases from '
                  f'{f(first["matched_train_probe_gap"])} to {f(last["matched_train_probe_gap"])}.','']
        part=[r for r in partitions if r['dataset']==ds]
        assert len(part)==30
        lines += ['| Epoch | 5 shuffled full-objective mean [min, max] | Fixed chronological | Shuffled no-regularizer mean |',
                  '|---:|---:|---:|---:|']
        for row in val:
            e=row['stage1_epoch'];p=[x for x in part if x['stage1_epoch']==e]
            vals=[float(x['loss']) for x in p]
            without=[float(x['no_regularizer']) for x in p]
            lines.append(f'| {e} | {statistics.mean(vals):.4f} '
                         f'[{min(vals):.4f}, {max(vals):.4f}] | {f(row["loss"])} | '
                         f'{statistics.mean(without):.4f} |')
        groups={r['partition_seed'] for r in part}
        assert len(groups)==5
        stable=all(min((r for r in part if r['partition_seed']==seed_value),
                       key=lambda r:float(r['loss']))['stage1_epoch']=='1' for seed_value in groups)
        noregul=all(min((r for r in part if r['partition_seed']==seed_value),
                        key=lambda r:float(r['no_regularizer']))['stage1_epoch']=='1' for seed_value in groups)
        assert stable and noregul and min(val,key=lambda r:float(r['loss']))['stage1_epoch']=='1'
        lines += ['',f'Epoch 1 ranks first under all five fixed shuffle partitions '
                  f'and under every no-regularizer ranking. Chronological and shuffled '
                  f'absolute values differ because batch grouping and the 19-window tail '
                  f'change the original unweighted batch-mean estimator; neither change '
                  f'the winner. These are the same Validation origins in every partition.','']
        g=[r for r in geometry if r['dataset']==ds and r['scope']=='val']
        lines += ['| Epoch | State | Standard deviation | Effective rank | Mean token L2 |',
                  '|---:|---|---:|---:|---:|']
        for row in g:
            lines.append(f'| {row["stage1_epoch"]} | {row["state"]} | '
                         f'{f(row["std"])} | {f(row["effective_rank_singular_entropy"],2)} | '
                         f'{f(row["mean_token_l2"])} |')
        a=[r for r in align if r['dataset']==ds]
        lines += ['', '| Later epoch | State | Centered CKA on Validation | Uncentered kernel alignment | Train-fitted Procrustes Validation MSE / epoch-1 variance |',
                  '|---:|---|---:|---:|---:|']
        for row in a:
            lines.append(f'| {row["stage1_epoch"]} | {row["state"]} | '
                         f'{f(row["centered_linear_cka_val"])} | '
                         f'{f(row["uncentered_kernel_alignment_val"])} | '
                         f'{f(row["registered_val_mse_over_epoch1_val_variance"])} |')
        lines += ['', 'The geometry uses the same 16 predefined Train and 16 Validation origins at each epoch. '
                  'CKA is centered linear kernel alignment over token rows; the separate uncentered '
                  'score is shown explicitly. Effective rank is the exponential entropy of centered '
                  'singular values. Orthogonal Procrustes is fitted on Train tokens and evaluated '
                  'only on Validation tokens. Its coordinate error is descriptive and the fitted '
                  'rotation is never applied at deployment. A geometry change alone does not establish '
                  'information loss or representation collapse.','']
    lines += ['## Interpretation limits','',
              'The consistent rise in matched Validation loss and widening matched gap establishes '
              'a genuine train–validation divergence for the original latent objective. '
              'The trajectory cosine term dominates the rise, while the regularizer contribution '
              'is small. The moving encoder changes target geometry, so the loss itself cannot '
              'identify whether forecasting information was lost. The fixed six-cell Decoder matrix '
              'tests that downstream consequence. No Test score enters this analysis.','']
    (OUT/'STAGE1_DYNAMICS_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('PHASE1_REPORT_BUILT',len(metrics),len(partitions),len(geometry),len(align))

if __name__=='__main__':run()
