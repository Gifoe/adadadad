"""Produce evidence-linked reports after the locked Test; never select or refit."""
import csv,datetime,difflib,hashlib,json,pathlib,subprocess,sys,zipfile
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from neso_online.native_worldts import file_sha

def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
def write(name,text):(ROOT/name).write_text(text.strip()+'\n',encoding='utf-8')
def metrics_table(rows):
    lines=['| Method | MAE MW | MSE MW² | Normalized MAE | Normalized MSE | MAE gain vs B0 | Improved windows | Worsened windows |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["method"]} | {r["mae_mw"]:.6f} | {r["mse_mw2"]:.6f} | {r["normalized_mae"]:.8f} | {r["normalized_mse"]:.8f} | {r["relative_mae_gain_vs_old"]:.4%} | {r["fraction_windows_improved"]:.2%} | {r["fraction_windows_worsened"]:.2%} |')
    return '\n'.join(lines)

def figures(history,val,test,compute):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    dest=ROOT/'outputs/figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(dest/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3))
    for stage,ax in zip([1,2],axes):
        h=[r for r in history if r['stage']==stage]
        ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],label='Train' if stage==1 else 'Train: prediction +0.3reconstruction')
        ax.plot([r['epoch'] for r in h],[r['val_loss'] for r in h],label='Validation' if stage==1 else 'Validation: prediction only')
        best=min(h,key=lambda r:r['val_loss']);ax.axvline(best['epoch'],ls='--',color='0.4',label=f'Selected epoch {best["epoch"]}')
        ax.set(xlabel='Epoch',ylabel='Latent objective' if stage==1 else 'Normalized decoded L1',title=f'Official WorldTS Stage {stage}');ax.legend(fontsize=8)
    fig.tight_layout();save(fig,'training_curves')
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for data,ax,label in [(val,axes[0],'Validation'),(test,axes[1],'Test: eligible April-July only')]:
        methods=[r['method'] for r in data['metrics']];x=np.arange(len(methods))
        ax.bar(x,[r['mae_mw'] for r in data['metrics']],color=['#718096','#2b6cb0','#b7791f','#a0aec0'][:len(methods)])
        ax.set_xticks(x,['B0 old','B1 recompute','B3 linear','B2 seasonal'][:len(methods)],rotation=20,ha='right');ax.set(ylabel='MAE (MW)',title=label)
    fig.tight_layout();save(fig,'matched_accuracy')
    fig,axes=plt.subplots(1,2,figsize=(10,3.3))
    for split,ax,label in [('val',axes[0],'Validation'),('test',axes[1],'Eligible Test')]:
        with open(ROOT/'outputs'/f'{split}_per_window.csv',encoding='utf-8') as f:rows=[r for r in csv.DictReader(f) if r['method']=='recompute']
        ax.scatter([float(r['revision_rms_mw']) for r in rows],[float(r['delta_mae_vs_old_mw']) for r in rows],s=10,alpha=.55,color='#2b6cb0')
        ax.axhline(0,color='0.4',lw=1);ax.set(xlabel='Wind/solar revision RMS (MW)',ylabel='B1 - B0 window MAE (MW)',title=label)
    fig.tight_layout();save(fig,'revision_improves_and_worsens')
    rows=[r for r in compute['rows'] if r['scope']=='host_output' and r['batch']==1]
    fig,ax=plt.subplots(figsize=(6,3.4));x=np.arange(len(rows));ax.bar(x,[r['p95_ms_per_batch'] for r in rows],color='#2b6cb0')
    ax.set_xticks(x,[r['method'] for r in rows],rotation=15,ha='right');ax.set_yscale('log');ax.set(ylabel='p95 incremental latency (ms, log scale)',title='Cached A; one new B window; host output')
    fig.tight_layout();save(fig,'incremental_compute')
    # Predefined illustrative cases: largest B1 improvement/worsening on each
    # split, reporting only. No sample filtering or training follows these.
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for split,data,rowaxes in [('val',val,axes[0]),('test',test,axes[1])]:
        filename='validation_predictions.npz' if split=='val' else 'test_predictions.npz'
        with np.load(ROOT/'outputs/model'/filename) as z:
            for kind,ax in zip(['largest_improvements','largest_degradations'],rowaxes):
                case=next(c for c in data['cases'] if c['method']=='recompute' and c['kind']==kind);i=case['window'];mask=z['update_mask'][i]
                lead=np.arange(1,49)/2
                for key,label,color in [('target_mw','Observed ND','black'),('pred_old_mw','B0 old','#a0aec0'),('pred_recompute_mw','B1 recompute','#2b6cb0'),('pred_output_correction_mw','B3 linear','#b7791f')]:
                    ax.plot(lead[mask],z[key][i,:,0][mask],label=label,color=color,lw=1.2)
                ax.set(xlabel='Hours after A-history endpoint',ylabel='ND (MW)',title=f'{split}: {kind.replace("largest_", "")} | {case["issue_a"][:16]}')
                ax.legend(fontsize=7)
    fig.tight_layout();save(fig,'illustrative_cases')

def main():
    history=read('outputs/model/train_history.json');audit=read('outputs/data_audit.json');env=read('outputs/environment.json')
    val=read('outputs/val_analysis.json');test=read('outputs/test_analysis.json');compute=read('outputs/compute_results.json')
    gate=read('outputs/model/updater_gate.json');lock=read('outputs/model/validation_decision_lock.json');diag=read('outputs/selected_model_diagnostics.json');gap=read('outputs/coverage_gap_audit.json')
    assert not gate['allow_updater'],'This report generator is for the measured negative-gate path; never fabricate updater outcomes'
    for path,expected in lock['sha256'].items():assert file_sha(path)==expected,'Frozen artifact changed: '+path
    v={r['method']:r for r in val['metrics']};t={r['method']:r for r in test['metrics']}
    b1=next(r for r in compute['rows'] if r['method']=='recompute' and r['batch']==1 and r['scope']=='host_output')
    b3=next(r for r in compute['rows'] if r['method']=='output_correction' and r['batch']==1 and r['scope']=='host_output')
    capture=(v['old']['mae_mw']-v['output_correction']['mae_mw'])/(v['old']['mae_mw']-v['recompute']['mae_mw'])
    write('ENVIRONMENT.md',f'''# Executed server environment

Client date:2026-10-14 Asia/Shanghai; environment snapshot UTC:{env['utc']}. Server work directory:`D:\\neso_worldts_online_v1`. SSH credentials are not stored in this artifact.

OS:`{env['platform']}`. Python:`{env['python']}`. Interpreter:`{env['executable']}`. Torch:`{env['package_versions']['torch']}`; CUDA runtime:{env['torch_cuda']}; GPU:{env['gpu']}; driver616.56; reported board memory32607MiB. Initial free GPU memory~29701MiB, RAMavailable{env['ram']['available']/2**30:.2f}GiB, diskfree{env['disk_free_bytes']/2**30:.2f}GiB. Resource snapshots and package versions are in `outputs/environment.json`.

Activate `E:\\Anaconda\\envs\\persist_stable_251` through the conda batch entrypoint before invoking the existing CUDA-compatible venv launcher. No Torch downgrade or original DAG requirements installation was performed. Only supplied lightweight requirements were checked. CPUthreads4; Windows affinity cores16-31. Single RTX5090; no distributed training or AMP.

Requirements check and original7-test package suite passed; repaired13-test suite passed. Original dummy-model/synthetic tests establish engineering behavior only. The separate `outputs/native_smoke.json` contains a real16-window TRAIN batch, actual imported WorldTS class, nonzero finite E/C/P gradients, Decoder-only Stage2 updates, label foil, shape `[16,48,1]`, and strict checkpoint tensor equality. Formal training and held-out results are separate artifacts.

Official source HEAD:`269426ea6217d7152a244da0e209d922acd46d4f`. A real Git checkout is retained under `upstream/WorldTS`; its tracked source was never patched.''')
    write('ISSUE_LOG.md','''# Execution issues and minimal repairs

1. Original2025 datastore dump stalled at zero bytes and was cancelled. Resolve the current official `resource_show` metadata and use the official static CSV URL when provided; retain dump fallback, retries, download progress and SHA validation. Logs:`download_attempt1.log`, `download_attempt2.log`.
2. Requested Jan-Jun2026 ID `d6375700-69c2-4c25-8bde-883a205d742e` returns404 and is absent from current official package resources. Save its unavailable record. Use the current resource31861619, named full-year2026, only after verifying actual raw coverage; no replacement Jan-JunCSV was fabricated.
3. Actual2026 `TIME_GMT` also uses HH:MM. Original timedelta coercion silently fell back to settlement keys. Explicit HH:MM parsing now validates GMT independently; mixed ISO production formats are parsed explicitly.
4. `prepare_attempt1.log`: Already tz-aware localization error from2025 civil settlement dates serialized with Z. Strip the ISO timestamp to its civil calendar date before London midnight localization; never shift the date using a timezone guess.
5. Actual2025 Historic Demand has no A/F column. Only the exact resource b2bde559 with verified Historic Demand2025 outturn metadata may use the whole archive.2026 retains explicit A-row filtering. Unknown archives with absent indicator remain rejected.
6. `prepare_attempt2.log`: strict GMT/settlement conflict. Full source inspection found1268 conflicting2026 future rows (0.03% of2026 archive). Explicit `--exclude-settlement-mismatch` drops these ambiguous rows; neither endpoint is chosen or repaired. Strict failure remains the default. Source conflict examples and counts are retained.
7. Original metric flag incorrectly labelled scaled MW errors as normalized. Now report MW/MW² and normalized metrics separately, restore the train mean for saved absolute predictions, and independently recalculate all scored metrics in float64. Preserve predictions, timestamps and masks. Ridge fitting uses float64; weights remain float32.
8. Original adapter lacked persistent per-epoch raw resume states and explicit Stage2 label-free validation. Save optimizer/RNG/generator states and selected/raw checkpoints; validation predictive path takes no target. Preserve original network, losses, constant learning rates, clipping and training sample schedule.
9. `native_smoke_attempt1.log`: fresh restore with different stage trainability flags failed bitwise prediction assertion. Configure the restored model for the same Stage2 state; strict state hashes and predictions then match exactly (0.0 observed difference) in the real smoke and complete Validation diagnostics. FP32 allclose tolerance is declared separately and observed differences are never rounded to zero.
10. Post-June source vintages often omit the first still-unfinished settlement slot. All124 August A candidates lack the full original A24h horizon. Keep the supplied adjacent-history/immediate-horizon definition, reject incomplete windows, and report only eligible April-July Test. Do not backfill current-period covariates or move A's target horizon after observing the results.
11. Lightweight updater was not trained: the predeclared Validation gate failed because linear output correction is within0.19% of B1 MAE. This is the task's required conditional branch, not an unexecuted training obligation. No extra architecture or objective was added.

All meaningful modifications are in `outputs/package_patch.diff` against the immutable supplied ZIP; new execution/audit scripts are separately retained. Original error logs remain unchanged.''')
    write('CONFIG_PROVENANCE.md',f'''# Frozen configuration and experimental definition

Official numerical `WorldTSModel` and `LatentStateLoss` are imported directly from the pinned Git checkout. The supplied `configs/worldts_neso.json` is a NESO-specific adaptation, not the paper's NP/PJM protocol. Model495473parameters: target latent128, covariate latent64; causal patch state/decoder patch5; PatchTST transition patch16/stride8, d_model64,4heads,1layer,d_ff256,dropout0.1; RevIN enabled, numerical future covariates enabled; no text/time modalities or loop. The source default regularizer is `legacy_vc`, matching the supplied loss settings. No upstream file was changed.

Lookback336, horizon48,30-minute UTC grid, seed2021, batch16, at most one real A issue per6h bin; B is the earliest issue6-8h after A. Hold all history fixed at A. Replace covariates only where period end>B; earlier covariates remain A. Both methods predict the same48targets and score the same unfinished portion (32-36slots).

Split by whole future target support:2025 TRAIN;2026Jan-Mar VAL;2026Apr-Aug requested TEST. Crossing windows dropped; actual eligible Test ends July19. All future target supports are mutually disjoint. Scaler fit on TRAIN history only, including repeated histories from sampled windows; no test/val fitting. Train-only revision quartile edges are frozen for both holdouts. Seasonal naive is a fixed336UTC-slot lag, not local-clock matching across DST.

Stage1: native latent MAE weight10, cosine15, legacy VCregularization0.1; detach target enabled. Adam0.001, clip5, max20epochs, patience5; select sample-weighted Validation latent loss. Stage2: E/C/P frozen and eval, Adam0.0001, predicted-state L1 +0.3target-state reconstruction L1; select predictive decoded Validation L1. Constant LR, no drop_last, no scheduler, no AMP are supplied adapter choices.

B3 uses TRAIN-only ridge0.01, no intercept; features normalized wind/solar deltas multiplied by lead0/1/2powers. No future labels enter its inference. B0 is cached A, B1 full recompute B, B2 seasonal, B3 linear correction. Prespecified gate and sensitivity settings are in `configs/execution_protocol.json`, written before formal training and validation metrics. The gate rejected the updater because B3-B1 gap is only{(v['output_correction']['mae_mw']/v['recompute']['mae_mw']-1):.4%}; the required gap was>1%.

Final decision lock UTC:{lock['frozen_utc']}. Exact configuration, selected models, calibration, scaler, dataset hashes, revision bins, analysis and inference code are SHA-locked in `outputs/model/validation_decision_lock.json`. Replay refuses changed artifacts and refuses overwriting a completed final Test. Test results were not used for selection.''')
    stage_rows=[]
    for stage in [1,2]:
        h=[r for r in history if r['stage']==stage];best=min(h,key=lambda r:r['val_loss']);last=h[-1]
        stage_rows.append(f'| {stage} | {len(h)} | {best["epoch"]} | {best["val_loss"]:.9f} | {last["optimizer_steps"]} | {last["stage_seconds"]:.3f} | {last["trainable_parameters"]} | {last["peak_cuda_allocated_mib"]:.3f} |')
    write('TRAINING_REPORT.md','''# Actual native WorldTS training

| Stage | Executed epochs | Selected epoch | Selected Validation objective | Optimizer steps | Stage seconds | Trainable parameters | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+'\n'.join(stage_rows)+f'''

TRAIN1406windows →88optimizer steps/epoch; batch16 with final14-window batch retained. Stage1 early-stopped at11 after5non-improving epochs; Stage2 completed its20epoch budget and selected20. Its curve was still improving at the boundary; this fixed-budget result does not establish convergence or SOTA. Combined measured stage time{sum([next(r for r in reversed(history) if r['stage']==s)['stage_seconds'] for s in [1,2]]):.3f}s excludes import/download/preparation and later calibration/replay.

The exact native losses, gradients and frozen tensor checks passed. Selected Stage1 modules are preserved in selected Stage2 checkpoint (SHA:{diag['stage2_selected_stage1_tensor_hash']}). Optimizer/RNG/generator resume state is saved after every completed epoch; retained checkpoints are the raw last state per stage, latest resume state, initial model and best state per stage. Earlier overwritten raw epoch states are not retained. Every objective remains in the full training history.

Complete Validation diagnostics: latent predicted feature std={diag['mean_h_pred_feature_std_across_windows_and_time']:.6f}, target feature std={diag['mean_h_target_feature_std_across_windows_and_time']:.6f}, predicted temporal std={diag['mean_h_pred_temporal_std']:.6f}. Latents are finite and not identically constant under these checks; these diagnostics do not prove representation quality. Future-label foil difference={diag['future_label_foil_h_pred_max_difference']}; fresh checkpoint prediction difference={diag['fresh_checkpoint_restore_prediction_max_difference']}; fresh restored saved Validation MW prediction difference={diag['fresh_restored_saved_validation_prediction_max_diff_mw']}. Checkpoint tensors load strictly with matching stage flags.

Updater training: skipped by predeclared Validation gate; no updater score or checkpoint is invented. Original7engineering tests and repaired13tests passed; real-model forward/backward evidence is independent. Curves:`outputs/figures/training_curves.png`. Raw evidence:`train_history.json`, training log, native smoke, selected model diagnostics and checkpoint SHA manifest.''')
    compute_lines=['| Method | Batch | Scope | Mean ms/batch | p95 ms/batch | Peak incremental MiB |','|---|---:|---|---:|---:|---:|']
    for r in compute['rows']:compute_lines.append(f'| {r["method"]} | {r["batch"]} | {r["scope"]} | {r["mean_ms_per_batch"]:.6f} | {r["p95_ms_per_batch"]:.6f} | {r["incremental_peak_cuda_allocated_mib"]:.6f} |')
    blocks=[]
    for split,data in [('Validation',val),('Test',test)]:
        for comparison in ['recompute_vs_old','output_correction_vs_old','output_correction_vs_recompute']:
            for length in ['1','7','14']:
                r=data['block_comparisons'][comparison][length]
                blocks.append(f'| {split} | {comparison} | {length} | {r["point_delta_mae_mw"]:.6f} | [{r["ci95_low_mw"]:.6f}, {r["ci95_high_mw"]:.6f}] | {r["fraction_disjoint_blocks_favor_candidate"]:.2%} |')
    case_lines=['| Split | Case | A UTC | B UTC | B1-B0 MAE MW | Revision RMS MW |','|---|---|---|---|---:|---:|']
    for split,data in [('Validation',val),('Test',test)]:
        for kind in ['largest_improvements','largest_degradations']:
            c=next(x for x in data['cases'] if x['method']=='recompute' and x['kind']==kind)
            case_lines.append(f'| {split} | {kind} | {c["issue_a"]} | {c["issue_b"]} | {c["delta_mae_vs_old_mw"]:.6f} | {c["revision_rms_mw"]:.6f} |')
    write('FINAL_REPORT.md',f'''# NESO Online WorldTS real execution

**Decision: NO-GO for a new persistent latent-state updater under this fixed V1 protocol.** The real revisions alter WorldTS forecasts and modestly improve mean accuracy, but a six-coefficient linear correction captures {capture:.2%} of the Validation MAE benefit. Full recomputation is already cheap on the measured RTX 5090: single-window host-output p95 {b1['p95_ms_per_batch']:.3f} ms, incremental peak {b1['incremental_peak_cuda_allocated_mib']:.3f} MiB. Validation did not justify training the supplied updater. The locked eligible Test gives B1 MAE {t['recompute']['mae_mw']:.3f} MW and B3 MAE {t['output_correction']['mae_mw']:.3f} MW, a {t['output_correction']['mae_mw']-t['recompute']['mae_mw']:.3f} MW gap. These measurements do not establish a useful accuracy or compute bottleneck for a new mechanism.

Execution status: actual data, model training, Validation, conditional decision, locked Test and reporting completed for available eligible windows. **The requested complete April-August coverage is not achieved:** the strict source-vintage definition yields only 303 Test windows and last target July 19, 2026. No August window qualifies. This is a source/protocol coverage limitation. Conclusions apply to this eligible cohort, one seed and this small supplied native configuration.

## Real data and temporal audit

Forecast archives:9,658,114raw rows,9,643,219retained future finite rows,15,353production issue timestamps.2025CSV contains5,369,698rows;2026full-year resource4,288,416rows. Identical cross-file duplicate issue-target rows:0; conflicting duplicate values:0 (the loader would fail on any). Source SHA256, actual field coverage, resource404 evidence and raw missingness are retained in `DATA_AUDIT.md` and JSON.48raw2026rows lack wind/solar;1268ambiguous timestamp rows were excluded without guessing. No synthetic samples or invented vintages enter the experiment.

Actual historicND:30,238half-hours,2025Jan1 through2026Sep22.2025 official outturn schema has no A/F column;2026 uses actual rows only. Genuine windows:TRAIN1406, Validation354, Test303. Scored unfinished slots:VAL12,494, TEST10,690. All windows have `[336,3]`history, `[48,2]`A/B covariates and `[48,1]`target; finite values, strict period-end>Bmask, identical A/Btarget times, unchanged already-ended prefix, fixed-Ahistory and disjoint future-target support all verified. Real DST demand days verify46/48/50-period behavior; see raw day counts.

The requested obsolete Jan-JunID returns404; the current official resource is named full-year2026 and actually covers issues Jan1-Oct13. It supplies the complete Validation quarter. From June onward many production issues omit the first partly elapsed settlement interval: e.g. A2026July15 00:55:17, required first end01:00 but source first end01:30. Missing immediate covariates cannot be filled from a later vintage. August has124selected source A candidates and0complete immediate24h A horizons. Monthly source evidence and examples are in `outputs/coverage_gap_audit.json`; no Test target values were needed to establish this exclusion.

This is a **retrospective forecast-vintage replay**. [Official Historic Demand documentation](https://www.neso.energy/data-portal/historic-demand-data) states publication21days in arrears and possible retrospective demand edits. Historic ND is therefore not proved observable instantaneously at A. Forecast production time is also not proved public-release time. Neither live operational validity nor causal benefit is claimed.

## Actual training and checkpoint evidence

Pinned official source:`269426ea6217d7152a244da0e209d922acd46d4f`. The495473parameter numerical model and official latent loss were actually imported and trained. Stage1 executed11epochs/968steps, selected6; Stage2 executed20epochs/1760steps, selected20. Stage2 preserves E/C/P exactly; Stage1 keeps D frozen. All loss values are finite; real smoke gradients, selected latent variation, target-label foil, strict restore and full saved Validation prediction reproduction passed. `TRAINING_REPORT.md` includes measured~42.7s stage time, GPU allocations, exact objectives and selected/raw checkpoint evidence. These are reproduced NESO results, not paper NP/PJM numbers. Decoder selection still improved at the20epoch budget boundary; the budget was not changed after observing Validation/Test.

## Matched Validation accuracy

{metrics_table(val['metrics'])}

B1 changes the scored prediction by>0.001MW on100% of windows; mean absolute prediction change145.46MW. It improves55.65% and worsens44.35% of windows. The aggregate MAE gain is1.1812%; B3 gain0.9956% and B3-B1 relative gap0.1878%. The predeclared>1%unexplained-headroom gate fails; no updater is trained. All four gate outcomes and decision timing are saved before Test.

## Locked eligible Test accuracy

{metrics_table(test['metrics'])}

B1 changes100% of windows; mean absolute prediction revision91.32MW. Improvement52.48%, worsening47.52%. Its mean MAE benefit over B0 is0.3826%, not a uniform improvement. B3 gives nearly identical MAE and slightly lower MSE. Test was evaluated once after the SHA lock; no coefficients, model, bins or thresholds were selected from these results. Complete predictions, targets, timestamps and masks are retained server-side with SHA hashes.

## Dependence-aware checks

Circular moving calendar-day block bootstrap,2000draws,seed2021;1/7/14day sensitivity. Daily weighted error sums preserve the original repeated-window scoring; no IID-window t-test is used. Adjacent24h windows overlap, so intervals are descriptive conditional on this fitted seed and eligible cohort.7-day blocks are primary; all lengths are reported.

| Split | Candidate-reference | Block days | Delta MAE MW | Descriptive95% interval MW | Disjoint blocks favor candidate |
|---|---|---:|---:|---|---:|
{chr(10).join(blocks)}

The Validation7-day B1 interval barely excludes zero; do not interpret its upper endpoint near zero as strong or multi-seed evidence. The full Test comparison is reporting only and never changes the frozen decision.

## Incremental revision-time compute

{chr(10).join(compute_lines)}

Twenty warmups and100timed calls per configuration, batch1/16; CUDA synchronized. A-time history, native latent/statistics and old prediction are cached identically. Host-output B1 includes new-Bnormalization/H2D, complete native model/Decoder, D2H and MW scaling. GPU-ready excludes those transfers. B0/B2copy cached host results; B3 uses only CPUrevision arithmetic and the six TRAIN-fitted coefficients. CPU timings include the common timer/synchronization overhead. No future ND values enter inference. A-time inference, source/network latency, public availability and long-lived cache orchestration are outside this cost scope. GPU allocation columns use live Torch tensors, not total board occupancy.

B3 single-window p95{b3['p95_ms_per_batch']:.6f}ms. B1 p95{b1['p95_ms_per_batch']:.6f}ms is far below the100ms cheap-recompute threshold predeclared before measurement, for a6-hour update schedule. A new updater would need an accuracy advantage or a materially different deployment constraint to justify itself; no such advantage was measured here.

## Both improvement and degradation cases

{chr(10).join(case_lines)}

These are deterministic largest-effect illustrations, not manually filtered examples or training choices. `outputs/val_analysis.json` / `test_analysis.json` retain five examples of each type per method. Per-window CSVs, all lead/Train-quartile strata and case plots provide the full distribution.

## Decision and remaining scope

NO-GO for persistent latent updating in this V1 setup. Prefer the existing native recomputation or the TRAIN-calibrated linear correction according to the measured accuracy/compute tradeoff. No claim that an untrained updater is mathematically incapable of improving; this task deliberately requires evidence of headroom before fitting it.

Limits that affect interpretation: incomplete and nonrandom post-June eligibility; single seed and one small supplied configuration; possibly revised historic ND unavailable at production time; production versus release timestamps; fixed UTC rather than local-DST seasonal lag; Stage2 budget boundary; descriptive dependent-block intervals rather than independent-replication evidence. A future study needing complete summer/live coverage must first obtain defensible current-period vintages and publication logs, then register its changed protocol separately. Existing V1 results remain frozen.

Required code, reports, metrics, logs, source metadata, patch and figures are archived. Large official CSVs, prepared NPZs, prediction NPZs and native PTcheckpoints stay under `D:\\neso_worldts_online_v1`; sizes and exactSHA256 are in `outputs/artifact_manifest.json`. `REPRODUCE.md` contains the executed protocol and recovery steps.''')
    write('REPRODUCE.md',r'''# Reproduce this server experiment

Windows command context:
```bat
call E:\Anaconda\condabin\conda.bat activate E:\Anaconda\envs\persist_stable_251
cd /d D:\neso_worldts_online_v1
set TASK_PYTHON=D:\baseline_reproduction_seed2021_v1\.venv\Scripts\python.exe
%TASK_PYTHON% -m pip install -r requirements.txt
%TASK_PYTHON% -m pytest -q
git clone --filter=blob:none --no-checkout https://github.com/decisionintelligence/WorldTS.git upstream/WorldTS
git -C upstream/WorldTS checkout 269426ea6217d7152a244da0e209d922acd46d4f
%TASK_PYTHON% -u -m neso_online.download --resources forecast_2025 forecast_2026_jan_jun forecast_2026_jun_dec demand_2025 demand_2026 --out data/raw --allow-unavailable
%TASK_PYTHON% -u -m neso_online.prepare --forecast data/raw/forecast_2025.csv data/raw/forecast_2026_jun_dec.csv --demand data/raw/demand_2025.csv data/raw/demand_2026.csv --lookback 336 --horizon 48 --revision-hours 6 --exclude-settlement-mismatch --out outputs/dataset
%TASK_PYTHON% -u scripts/audit_dataset.py
%TASK_PYTHON% -u scripts/native_smoke.py
%TASK_PYTHON% -u -m neso_online.native_worldts train --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021 --epochs1 20 --epochs2 20 --patience 5
%TASK_PYTHON% -u scripts/audit_coverage_gap.py
%TASK_PYTHON% -u scripts/diagnose_selected_model.py
%TASK_PYTHON% -u scripts/analyze_replay.py val
%TASK_PYTHON% -u scripts/benchmark_compute.py
```

Read `outputs/model/updater_gate.json`. In this actual execution `allow_updater=false`, so updater training is correctly skipped. For an independently reproduced run whose predeclared gate is true, the only allowed extra model is the supplied updater with epochs20/patience5; analyze its Validation predictions and repeat the same compute benchmark before freezing. Do not fit extra architectures or retune on Test.

Final frozen stage:
```bat
%TASK_PYTHON% -u scripts/freeze_validation.py
%TASK_PYTHON% -u -m neso_online.native_worldts replay --data outputs/dataset --out outputs/model --worldts-source upstream/WorldTS/DAG-bench --config configs/worldts_neso.json --device cuda --batch 16 --seed 2021
%TASK_PYTHON% -u scripts/analyze_replay.py test
%TASK_PYTHON% -u scripts/finalize_artifacts.py
```

Do not rerun the final replay in the completed experiment: the driver refuses overwriting `test_replay.json`. A genuine replication uses an independent fresh directory and preserves this locked result. Training resumes from raw completed epoch state if interrupted; selected best checkpoints are separate. Existing completed native training must not be restarted to choose a different result.

Verify official Git HEAD, original ZIPSHA, raw downloaded SHA and the artifact manifest before comparison. Downloader validates cached files; current live NESO data may later change. Exact replication requires the retained server snapshot hashes, not an arbitrary later download. Source/publication caveats and missing2026summer first-period vintages are part of the benchmark definition.

Server launcher paths are environment-specific. Linux may invoke its own compatible Python directly, but retain the source/config/data hashes, seed, batches, losses and stage selection. Requirements must not downgrade RTX5090 CUDA-compatible Torch. Export only compact code/reports/CSV/logs/figures; large raw data, weights and prediction tensors remain server-side with manifest pointers.''')
    # Remove redundant backslashes introduced by the raw Windows-command text.
    reproduce=ROOT/'REPRODUCE.md';reproduce.write_text(reproduce.read_text(encoding='utf-8').replace('\\\\','\\'),encoding='utf-8')
    write('README.md','''# NESO Online WorldTS V1 — executed artifact

Real official NESO data and pinned native WorldTS were run on RTX5090. The research decision is **NO-GO for a persistent latent-state updater**; the validation-only gate correctly skipped updater training.

Read [FINAL_REPORT.md](FINAL_REPORT.md), [DATA_AUDIT.md](DATA_AUDIT.md), [TRAINING_REPORT.md](TRAINING_REPORT.md), [REPRODUCE.md](REPRODUCE.md) and the four result CSVs. Execution is complete for eligible data, with TRAIN1406, VAL354 and TEST303windows. The requested complete April-August Test is not available under the original immediate-horizon definition; eligible Test ends July19 and August contributes no window. This limitation is explicit throughout the report.

The immutable supplied engineering ZIP is retained server-side. Its original README, STATUS and execution instructions are copied to `outputs/original_docs`; synthetic/dummy tests in that package are separate from actual native training. Native checkpoints, prediction NPZs and raw CSVs remain server-side, referenced by `outputs/artifact_manifest.json`. Compact source, evidence, CSVs and figures are published.''')
    write('STATUS.md','''# Current execution status

COMPLETE_AVAILABLE_COVERAGE; NO_GO_PERSISTENT_LATENT_UPDATER. Real Stage1/Stage2, Validation, conditional updater decision, locked final Test, dependence-aware analyses and compute measurement completed.13engineering tests plus independent real native smoke/restore diagnostics passed. Updater was skipped by the predeclared Validation gate. Full April-August coverage is not met: available eligible Test ends2026July19. See FINAL_REPORT.md and EXECUTION_STATUS.json; original supplied status is archived under outputs/original_docs.''')
    write('TEST_REPORT.md','''# Current executed test evidence

Server: NVIDIA RTX 5090, Python 3.10.20, Torch 2.8.0+cu128. The supplied engineering suite passed 7 tests on this server; the repaired suite passed 13 tests. Raw logs are in outputs/logs/tests_original.log and tests_final.log.

Those tests include dummy/synthetic fixtures and are not the scientific results. Separate real evidence includes an official native-model forward/backward smoke on 16 actual TRAIN windows, finite nonzero E/C/P gradients, Decoder-only Stage 2 updates, finite [16,48,1] predictions, target-label foil, strict state hash and matching checkpoint restore. The complete selected-model Validation predictions reproduce exactly with matching stage flags (observed maximum difference 0.0 MW).

Actual native Stage 1 and Stage 2 were trained on 1406 genuine windows. Validation 354 and the locked eligible Test 303 windows were evaluated; all MW/MW² metrics independently match preserved predictions and masks. See TRAINING_REPORT.md, VALIDATION_RESULTS.csv, TEST_RESULTS.csv and FINAL_REPORT.md. The final content/lock verification is outputs/artifacts_verification.json.

The original supplied CPU/dummy report, which preceded server execution and stated that real results were unverified, is preserved as historical evidence in outputs/original_docs/TEST_REPORT.md and original_package.zip. Its synthetic sample counts must not be used as NESO experimental results. Remaining limitations are source publication timing and incomplete strict April-August coverage; these remain explicitly unresolved.''')
    figures(history,val,test,compute)
    # Preserve original provided documentation before adding current status.
    originals=ROOT/'outputs/original_docs';originals.mkdir(exist_ok=True)
    patch=[]
    with zipfile.ZipFile(ROOT/'original_package.zip') as z:
        for info in z.infolist():
            parts=pathlib.PurePosixPath(info.filename).parts
            if len(parts)<2 or info.is_dir():continue
            relative=pathlib.Path(*parts[1:]);current=ROOT/relative
            if relative.suffix.lower() in ['.py','.json','.md','.txt'] and current.is_file():
                old=z.read(info).decode('utf-8-sig');new=current.read_text(encoding='utf-8-sig')
                if relative.name in ['README.md','STATUS.md','CODEX_EXECUTION.md','TEST_REPORT.md']:(originals/relative.name).write_text(old,encoding='utf-8')
                patch.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original/'+relative.as_posix(),tofile='executed/'+relative.as_posix()))
    (ROOT/'outputs/package_patch.diff').write_text(''.join(patch),encoding='utf-8')
    write('EXECUTION_STATUS.json',json.dumps({'status':'COMPLETE_AVAILABLE_COVERAGE','research_decision':'NO_GO_PERSISTENT_LATENT_UPDATER',
        'requested_full_april_august_test_coverage':False,'updater_trained':False,'updater_skipped_by_validation_gate':True,
        'real_train_windows':1406,'real_validation_windows':354,'real_test_windows':303,'test_last_target':'2026-07-19T01:00:00Z',
        'artifact_policy':'large signals/tensors/checkpoints retained server-side with hashes','client_date':'2026-10-14'},indent=2))
    # Comprehensive immutable content manifest (exclude itself, Git/internal
    # imports, and streaming log currently being written by this script).
    files=[]
    for path in sorted(ROOT.rglob('*')):
        rel=path.relative_to(ROOT)
        if not path.is_file() or any(part in ['upstream','__pycache__','.pytest_cache'] for part in rel.parts):continue
        if rel.as_posix() in ['outputs/artifact_manifest.json','outputs/artifacts_verification.json','outputs/logs/finalize_artifacts.log'] or path.suffix in ['.part','.tmp']:continue
        files.append({'relative_path':rel.as_posix(),'size_bytes':path.stat().st_size,'sha256':file_sha(path),
                      'retained_server_only':path.suffix in ['.pt','.npz','.npy'] or rel.parts[:2]==('data','raw') and path.suffix=='.csv'})
    tracked=subprocess.check_output(['git','-C',str(ROOT/'upstream/WorldTS'),'status','--porcelain'],text=True).strip();assert not tracked,'Official tracked source modified'
    manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'server_root':str(ROOT),
              'official_source_commit':subprocess.check_output(['git','-C',str(ROOT/'upstream/WorldTS'),'rev-parse','HEAD'],text=True).strip(),
              'official_tracked_source_clean':True,'files':files,'n_files':len(files),
              'note':'Hashes describe executed server bytes. Git may normalize text line endings; immutable binary/data snapshots retain exact hashes.'}
    (ROOT/'outputs/artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    required=['DATA_AUDIT.md','ENVIRONMENT.md','ISSUE_LOG.md','CONFIG_PROVENANCE.md','TRAINING_REPORT.md','VALIDATION_RESULTS.csv','TEST_RESULTS.csv','REVISION_ANALYSIS.csv','COMPUTE_RESULTS.csv','REPRODUCE.md','FINAL_REPORT.md']
    assert all((ROOT/name).is_file() and (ROOT/name).stat().st_size for name in required)
    print('FINAL_ARTIFACTS_VERIFIED',json.dumps({'decision':'NO_GO','required_deliverables':required,'manifest_files':len(files),'source_clean':True}),flush=True)
if __name__=='__main__':main()
