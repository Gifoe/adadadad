"""Derive the locked research decision and Chinese final reports from raw CSVs."""
import csv, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'

def read(name):
    with (OUT/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def f(x,d=5):return f'{float(x):.{d}f}'
def pct(x):return f'{100*float(x):.2f}%'
def rows_for(rows,**keys):return [r for r in rows if all(str(r[k])==str(v) for k,v in keys.items())]
def one(rows,**keys):
    result=rows_for(rows,**keys);assert len(result)==1,(keys,len(result));return result[0]

def prepare():
    status=json.loads((OUT/'formal_matrix_status.json').read_text())
    assert status['all_six_locked'] and status['stage']=='complete'
    locked=json.loads((OUT/'diagnostic_lock.json').read_text())
    assert locked['all_six_fits_completed_before_first_diagnostic']
    test=json.loads((OUT/'test_results.json').read_text())
    assert test['failed']==0 and test['passed']>=48
    checkpoint=json.loads((OUT/'checkpoint_hashes.json').read_text())
    assert checkpoint['protected_inputs_unchanged'] and checkpoint['raw_stage1_snapshots']==12
    assert checkpoint['decoder_epoch_checkpoints']==126
    return test,checkpoint

def decide(s,r,c,b):
    evidence={};threshold=.02
    selection_key='val_select_selection_mae_batch_mean'
    for ds in ['NP','PJM']:
        latent=[one(s,dataset=ds,domain='validation_chronological',stage1_epoch=e) for e in [1,3,6]]
        candidates=[one(r,dataset=ds,stage1_epoch=e,decoder_checkpoint='best') for e in [1,3,6]]
        by_ep={int(x['stage1_epoch']):x for x in candidates}
        select_winner=min(candidates,key=lambda x:float(x[selection_key]))
        diagnostic_winner=min(candidates,key=lambda x:float(x['val_diagnostic_forecast_mae']))
        latent_winner=[1,3,6][min(range(3),key=lambda i:float(latent[i]['loss']))]
        winner_ep=int(select_winner['stage1_epoch'])
        runnerup=min((x for x in candidates if int(x['stage1_epoch'])!=winner_ep),
                     key=lambda x:float(x[selection_key]))
        select_gain_over_e1=((float(by_ep[1][selection_key])-
                              float(select_winner[selection_key]))/
                             float(by_ep[1][selection_key]))
        select_gain_over_runnerup=((float(runnerup[selection_key])-
                                    float(select_winner[selection_key]))/
                                   float(runnerup[selection_key]))
        diag_gain_vs_e1=((float(by_ep[1]['val_diagnostic_forecast_mae'])-
                          float(by_ep[winner_ep]['val_diagnostic_forecast_mae']))/
                         float(by_ep[1]['val_diagnostic_forecast_mae']))
        e20={e:one(c,dataset=ds,stage1_epoch=e,decoder_epoch=20) for e in [1,3,6]}
        last_checkpoints=[one(r,dataset=ds,stage1_epoch=e,decoder_checkpoint='last') for e in [1,3,6]]
        common_step_gain=((float(e20[1][selection_key])-
                           float(e20[winner_ep][selection_key]))/float(e20[1][selection_key]))
        block_rows=rows_for(b,dataset=ds,stage1_epoch=winner_ep) if winner_ep!=1 else []
        stable=False
        if block_rows:
            group360=one(block_rows,block_length=360)
            chunk=json.loads(group360['nonoverlap_block_delta_json'])
            stable=(sum(x<0 for x in chunk)>=3 and
                    all(float(row['fraction_bootstrap_better'])>=.8 for row in block_rows))
        clear_later=(latent_winner==1 and winner_ep!=1 and
                     select_gain_over_e1>threshold and select_gain_over_runnerup>threshold and
                     common_step_gain>threshold and diag_gain_vs_e1>0 and stable)
        strong_later_select=(winner_ep!=1 and select_gain_over_e1>threshold and common_step_gain>threshold)
        e1_forecast_best=(winner_ep==1 and int(diagnostic_winner['stage1_epoch'])==1)
        later_worse=(latent_winner==1 and e1_forecast_best and all(
            float(by_ep[e][selection_key])>
            float(by_ep[1][selection_key])*(1+threshold) and
            float(by_ep[e]['val_diagnostic_forecast_mae'])>
            float(by_ep[1]['val_diagnostic_forecast_mae']) for e in [3,6]))
        evidence[ds]={'latent_best_epoch':latent_winner,'forecast_best_select_epoch':winner_ep,
                      'forecast_best_diagnostic_epoch_descriptive':int(diagnostic_winner['stage1_epoch']),
                      'select_gain_over_epoch1':select_gain_over_e1,
                      'select_gain_over_runnerup':select_gain_over_runnerup,
                      'diagnostic_gain_winner_vs_epoch1':diag_gain_vs_e1,
                      'same_step20_select_gain_vs_epoch1':common_step_gain,
                      'descriptive_block_stability':stable,
                      'strong_later_select_signal':strong_later_select,
                      'clear_later_state_selection_signal':clear_later,
                      'later_worse_than_epoch1':later_worse,
                      'decoder_init_hashes_identical':len({x['decoder_init_tensor_hash'] for x in candidates})==1,
                      'step_counts_equal':all(int(x['n_optimizer_steps'])==22240 for x in last_checkpoints)}
    if all(v['clear_later_state_selection_signal'] for v in evidence.values()):
        status='GO_STATE_SELECTION_INDEPENDENT_VALIDATION'
        reason='两个任务在等训练机会下，较晚状态均比 E1 和次优候选改善超过预定的 2%，且时间隔离诊断与连续区块趋势同向；下一步只进入独立数据验证。'
    elif all(v['later_worse_than_epoch1'] for v in evidence.values()):
        status='GO_GENERALIZATION_OPTIMIZATION_AUDIT'
        reason='两个任务均由 E1 同时取得最低潜变量验证损失和最低解码预测误差；较晚状态在等预算下两项都退化。下一轮转向泛化与优化审计。'
    elif sum(v['strong_later_select_signal'] for v in evidence.values())==1:
        status='NO_GO_NEW_STATE_SELECTION_METHOD'
        strong_task=next(ds for ds,value in evidence.items() if value['strong_later_select_signal'])
        if evidence[strong_task]['diagnostic_gain_winner_vs_epoch1']<0:
            reason=(f'{strong_task} 的较晚状态只在选择集上明显更好，在时间隔离诊断集上方向反转；'
                    '另一任务没有同向的选择集收益。未满足预设跨任务和区块稳定性门槛，不据此开发新状态选择方法。')
        else:
            reason='仅一个任务在等步数下出现超过预设 2% 的较晚状态收益，另一任务没有同向证据；跨任务排序不一致，不据此开发通用状态选择方法。'
    elif all(v['select_gain_over_epoch1']<threshold for v in evidence.values()):
        status='NO_GO_NEW_STATE_SELECTION_METHOD'
        reason='两个任务均未观察到超过预先设定 2% 效果尺度的较晚状态下游收益；目前没有开发新状态选择方法的依据。'
    else:
        status='INCONCLUSIVE_TASK_DEPENDENT'
        reason='较晚状态的下游信号没有同时满足两个任务、等步数和时间区块稳定性的预设门槛；当前只能给出任务依赖的描述性结果。'
    return {'status':status,'reason':reason,'per_dataset':evidence,
            'effect_scale':threshold,'diagnostic_is_not_historically_unseen':True,
            'new_method_trained':False,'new_test_scores_used':False,
            'research_scope':'single seed, two datasets, 20-epoch matched decoder budget; no causal mechanism uniquely identified'}

def report(s,g,a,r,c,b,decision,test,checkpoint,environment):
    lines=['# Why Does WorldTS Stage 1 Peak at Epoch 1?','',
           '实验：`worldts_stage1_state_selection_audit_seed2021_v1`。官方 WorldTS 源码固定在 '
           '`269426ea6217d7152a244da0e209d922acd46d4f`。NP/PJM 均为 T=720、H=360、seed 2021。'
           '本轮没有训练新结构、没有使用 Test 预测、没有根据结果追加训练或调参。','',
           '## 问题一：为什么 Epoch 1 的 Latent Validation Loss 最低？','',
           '第 1 轮在完整固定验证窗口、五个预先固定的分组和去除 batch regularizer 的目标下均最低。'
           '同轮快照 `eval()` 的 Train Probe 与 Validation 差距随轮数增大：']
    for ds in ['NP','PJM']:
        x=one(s,dataset=ds,domain='validation_chronological',stage1_epoch=1)
        y=one(s,dataset=ds,domain='validation_chronological',stage1_epoch=6)
        parts={'l1':10*(float(y['mse_loss'])-float(x['mse_loss'])),
               'cos':15*(float(y['cosine_loss'])-float(x['cosine_loss'])),
               'reg':.1*(float(y['sigreg_loss'])-float(x['sigreg_loss']))}
        lines.append(f'- {ds}：固定 Validation {f(x["loss"])} → {f(y["loss"])}；'
                     f'匹配的 Train–Validation 差距 {f(x["matched_train_probe_gap"])} → '
                     f'{f(y["matched_train_probe_gap"])}。增量中加权轨迹余弦 '
                     f'{parts["cos"]:.5f}，加权 L1 {parts["l1"]:.5f}，regularizer '
                     f'{parts["reg"]:.5f}。')
    lines += ['', '在线训练损失逐 batch 测得，不能与轮末快照的 Validation 损失直接相减。'
              '固定 Train Probe 的轮末重评排除了这一比较口径错误。分组可改变绝对值，尤其最后 19 窗口'
              '在官方无权 batch 均值中仍占一个完整 batch 的权重，但不改变最优轮次。'
              '此实验显示目标函数的训练与验证分离，不能单凭损失上升认定预测信息丢失。','',
              '## 问题二：不同 Epoch 的 Latent State 是否发生显著变化？','',
              '| Dataset | State | Epoch 1/6 std | Epoch 1/6 effective rank | Epoch 6 vs 1 centered CKA | Train-fit Procrustes Val MSE / Epoch-1 variance |',
              '|---|---|---:|---:|---:|---:|']
    for ds in ['NP','PJM']:
        for state in ['h_hist','h_target','h_pred']:
            x=one(g,dataset=ds,scope='val',stage1_epoch=1,state=state)
            y=one(g,dataset=ds,scope='val',stage1_epoch=6,state=state)
            z=one(a,dataset=ds,stage1_epoch=6,state=state)
            lines.append(f'| {ds} | {state} | {f(x["std"],3)} / {f(y["std"],3)} | '
                         f'{f(x["effective_rank_singular_entropy"],2)} / '
                         f'{f(y["effective_rank_singular_entropy"],2)} | '
                         f'{f(z["centered_linear_cka_val"],3)} | '
                         f'{f(z["registered_val_mse_over_epoch1_val_variance"],3)} |')
    lines += ['', '三个状态的几何均改变；有效秩普遍上升。CKA 量化结构相似性，不能等同于预测信息。'
              'Procrustes 只用 16 个预定训练窗口拟合旋转，在验证窗口上评估，且没有进入部署预测。'
              '因此不把几何变化直接称为 collapse、信息丢失或已确认的接口漂移。','',
              '## 问题三：哪个 Epoch 最适合最终 Forecasting？','',
              '下表 Decoder checkpoint 只由 `V_select` 选择。Forecast MAE/MSE 均来自所有六次训练及选择锁定后'
              '才计算的 `V_diagnostic`，不能把它再用于挑选部署模型。每组 Decoder 都训练完整 20 轮 '
              '(22,240 steps)，下表报告固定预算内的 best checkpoint。','',
              '| Dataset | Stage1 Epoch | Latent Validation Loss | Decoder Best Epoch | Forecast MAE | Forecast MSE | V_select weighted MAE |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['NP','PJM']:
        for ep in [1,3,6]:
            l=one(s,dataset=ds,domain='validation_chronological',stage1_epoch=ep)
            x=one(r,dataset=ds,stage1_epoch=ep,decoder_checkpoint='best')
            lines.append(f'| {ds} | {ep} | {f(l["loss"])} | {x["decoder_epoch"]} | '
                         f'{f(x["val_diagnostic_forecast_mae"])} | '
                         f'{f(x["val_diagnostic_forecast_mse"])} | '
                         f'{f(x["val_select_forecast_mae"])} |')
    lines += ['', 'Checkpoint 选择及跨状态的选择集排序均按预注册的无权 batch 均值 MAE；表中 MAE/MSE '
              '按全部预测元素等权计算，两个统计口径均保留。'
              '同一任务三组的 Decoder 初始 tensor hash 与每轮 Train origins 顺序 hash 一致，'
              'Encoder/Condition Encoder/Predictor 全程冻结。'
              '各组最后一轮及相同步数的曲线也保留在 `decoder_training_curves.csv`；'
              '最优点不是以不同训练机会取得。','',
              '## 问题四：Latent Loss 最优是否等于最终 Forecasting 最优？','']
    for ds in ['NP','PJM']:
        x=decision['per_dataset'][ds]
        if x['forecast_best_select_epoch']==1 and x['forecast_best_diagnostic_epoch_descriptive']!=1:
            ep=x['forecast_best_diagnostic_epoch_descriptive']
            baseline=one(r,dataset=ds,stage1_epoch=1,decoder_checkpoint='best')
            candidate=one(r,dataset=ds,stage1_epoch=ep,decoder_checkpoint='best')
            sel_gain=(float(baseline['val_select_selection_mae_batch_mean'])-
                      float(candidate['val_select_selection_mae_batch_mean']))/float(baseline['val_select_selection_mae_batch_mean'])
            diag_gain=(float(baseline['val_diagnostic_forecast_mae'])-
                       float(candidate['val_diagnostic_forecast_mae']))/float(baseline['val_diagnostic_forecast_mae'])
            lines.append(f'- {ds}：latent 与 `V_select` 的预测最优均为 E1；'
                         f'诊断集描述性最优为 E{ep}。E{ep} 相对 E1 在选择集的改善 '
                         f'{pct(sel_gain)}，在诊断集的改善 {pct(diag_gain)}。')
        else:
            lines.append(f'- {ds}：latent 最优 E{x["latent_best_epoch"]}；'
                         f'`V_select` 预测最优 E{x["forecast_best_select_epoch"]}，'
                         f'隔离诊断集上的描述性最优 E{x["forecast_best_diagnostic_epoch_descriptive"]}。'
                         f'选择集 E{x["forecast_best_select_epoch"]} 相对 E1 的 MAE 改善 '
                         f'{pct(x["select_gain_over_epoch1"])}；同在 Decoder 第 20 轮的改善 '
                         f'{pct(x["same_step20_select_gain_vs_epoch1"])}；'
                         f'其诊断集相对 E1 的改善 {pct(x["diagnostic_gain_winner_vs_epoch1"])}。')
    for ds in ['NP','PJM']:
        x=decision['per_dataset'][ds]
        if x['forecast_best_select_epoch'] != x['forecast_best_diagnostic_epoch_descriptive']:
            ep=3 if 3 in (x['forecast_best_select_epoch'],x['forecast_best_diagnostic_epoch_descriptive']) else 6
            block=one(b,dataset=ds,stage1_epoch=ep,block_length=360)
            chunks=json.loads(block['nonoverlap_block_delta_json'])
            lines.append(f'- {ds} 的选择集与隔离诊断集最优 Stage-1 轮次发生反转：'
                         f'E{ep} 相对 E1 的诊断 MAE 差值 '
                         f'{f(block["observed_delta_mae_candidate_minus_epoch1"])}，'
                         f'{sum(z<0 for z in chunks)}/{len(chunks)} 个不重叠 360-origin 区块支持 E{ep}；'
                         f'360-origin block bootstrap 描述区间 '
                         f'[{f(block["bootstrap_ci025"])}, {f(block["bootstrap_ci975"])}]。')
    lines += ['', '诊断集只用于事后描述，且历史研究已经接触过完整旧验证区间，'
              '故不是独立新测试集。其 1,363 个滚动窗口高度重叠；'
              '连续区块 bootstrap 仅是敏感性描述，不作独立样本显著性检验。','',
              '| Dataset | Candidate vs E1 | Block length | ΔMAE candidate−E1 | 95% descriptive interval | Better resample fraction |',
              '|---|---|---:|---:|---:|---:|']
    for ds in ['NP','PJM']:
        for ep in [3,6]:
            for block in [360,720,1080]:
                x=one(b,dataset=ds,stage1_epoch=ep,block_length=block)
                lines.append(f'| {ds} | E{ep} | {block} | '
                             f'{f(x["observed_delta_mae_candidate_minus_epoch1"])} | '
                             f'[{f(x["bootstrap_ci025"])}, {f(x["bootstrap_ci975"])}] | '
                             f'{pct(x["fraction_bootstrap_better"])} |')
    mismatches=[ds for ds in ['NP','PJM'] if decision['per_dataset'][ds]['forecast_best_select_epoch']!=1]
    if mismatches:
        lines += ['', '在固定选择集上，'+ '、'.join(mismatches) +
                  ' 的下游最优状态与 E1 潜变量最优状态不一致；这是观察到的模型选择失配，'
                  '但仍需看时间区块和另一任务是否支持可泛化机制。']
    lines += ['', '## 问题五：真正值得研究的失败机制是什么？','',
              '严格可归因的事实是：Stage-1 轮末验证目标在两任务上均于 E1 最低，而其主要上升分量'
              '是轨迹余弦；与此同时 Encoder/target 与 predicted latent 坐标和有效秩持续变化。'
              '下游 Decoder 对比检验这种选择准则是否遗漏预测有用的状态。'
              '更直接的实证失败是时间切片上的轮次排序不稳定：NP 在选择集偏向 E3，'
              '在原始支持不重叠的诊断切片却偏向 E1；PJM 的排序方向也不同，且诊断 E3 优势很小。'
              '目前的消融设计不能区分“移动目标”“优化路径”“模型选择目标错配”各自的因果贡献，'
              '也不能推出某种新损失或模块有效。先前 Predictor-only 审计对复杂 readout-aware loss 已给 NO-GO，'
              'future-response Loop 对比为 INCONCLUSIVE；本实验不推翻那些结论，只定位 Stage-1 选择问题。','',
              '## 问题六：研究决策','',
              f'**{decision["status"]}**。{decision["reason"]}','',
              '这是 seed 2021、NP/PJM 两个任务内的机制可行性判断。'
              '若出现值得继续的状态选择信号，下一步必须在此前未用于方法选择的新数据集和多个种子上'
              '预注册确认；若没有一致信号，不增添 Jacobian、adapter、loop 或新损失。','',
              '## 可复核执行记录','',
              f'- 自动检查：{test["passed"]} PASS、{test["failed"]} FAIL；'
              f'12 个真实 raw Stage-1 checkpoint、126 个逐轮 Decoder checkpoint 均已记录 SHA-256。',
              '- 六组 Adam/type-3/batch32/20 epoch；三组 NP LR=1e−4，三组 PJM LR=2e−4。'
              '官方 Decoder mix objective 为预测 MAE + 0.3×target-state 重构 MAE。',
              '- `V_select` 原点 36,690–38,051；`V_diagnostic` 原点 40,210–41,572；'
              '两组原始时间支持中间有 1,079 小时空档。',
              '- 服务器运行环境：'+environment['gpu_name']+'，PyTorch '+environment['torch']+
              '，CUDA '+str(environment['torch_cuda'])+'；原始基线输入的 16 个 SHA-256 未改变，官方源码 312 个文件 SHA-256 重验通过。'
              '所有大权重与诊断预测阵列留在服务器；Git 保存源码、配置、日志摘要、报告、原始 CSV 和完整路径哈希。','',
              '相关原始数据：`stage1_epoch_metrics.csv`、`batch_partition_sensitivity.csv`、'
              '`latent_geometry.csv`、`latent_geometry_alignment.csv`、`decoder_training_curves.csv`、'
              '`decoder_training_results.csv`、`paired_block_statistics.csv`。'
              '七类图在 `outputs/figures`，原始数值以 CSV 为准。']
    return '\n'.join(lines)+'\n'

def cross_report(r,c,decision):
    lines=['# Cross-epoch Decoder comparison','',
           'Six independent Decoder fits use the original untrained task-specific Decoder weights. '
           'All three Stage-1 states in each task share model architecture, Train origins, randomized '
           'origin order per epoch, batch size, Adam optimizer, type-3 schedule, 20 complete epochs '
           'and 22,240 optimizer steps. The same fixed `V_select` scorer chooses checkpoints. '
           '`V_diagnostic` is computed only after all six choices are locked.','',
           '| Dataset | Stage1 E | Checkpoint | Decoder E | V_select MAE/MSE | V_diagnostic MAE/MSE | Select reconstruction MAE/MSE | Diagnostic reconstruction MAE/MSE | Train MAE at E20 | Seconds | Peak CUDA allocated MiB |',
           '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ['NP','PJM']:
        for ep in [1,3,6]:
            for label in ['best','last']:
                x=one(r,dataset=ds,stage1_epoch=ep,decoder_checkpoint=label)
                lines.append(f'| {ds} | {ep} | {label} | {x["decoder_epoch"]} | '
                             f'{f(x["val_select_forecast_mae"])}/{f(x["val_select_forecast_mse"])} | '
                             f'{f(x["val_diagnostic_forecast_mae"])}/{f(x["val_diagnostic_forecast_mse"])} | '
                             f'{f(x["val_select_reconstruction_mae"])}/{f(x["val_select_reconstruction_mse"])} | '
                             f'{f(x["val_diagnostic_reconstruction_mae"])}/{f(x["val_diagnostic_reconstruction_mse"])} | '
                             f'{f(x["training_mae_last_epoch"])} | {f(x["training_seconds"],1)} | '
                             f'{f(x["peak_cuda_mib"],1)} |')
    lines += ['', 'The optimizer step count for each last checkpoint is exactly 22,240. '
              'The best checkpoint may occur earlier because selection was allowed at step 0 '
              'and after every complete epoch; no training was stopped at that point. '
              'The E20 rows compare equal update opportunities. '
              'The `curves.csv` for each cell and combined `decoder_training_curves.csv` retain all 21 checkpoints, '
              'including step 0, training objective, forecast MAE, reconstruction MAE, learning rate, '
              'selection MAE/MSE, origin-order SHA and elapsed time. '
              'The checkpoint manifest contains file SHA-256, full optimizer and RNG recovery states, '
              'and strict model restore. Peak values refer to PyTorch process-owned allocated CUDA memory, '
              'not total board occupancy.','',
              'The Decoder predictive path used `model.forecast(history, x_future=known_covariates)` '
              'without a future-target tensor. Reconstruction used the target latent only as a diagnostic '
              'or as the official auxiliary training target. Label foil and repeated-prediction tests passed. '
              'Only Decoder parameters received optimizer updates.','',
              'Research decision: **'+decision['status']+'**. '+decision['reason']]
    return '\n'.join(lines)+'\n'

def next_report(decision):
    status=decision['status']
    lines=['# Next research decision','',f'**{status}**','',decision['reason'],'']
    if status=='GO_STATE_SELECTION_INDEPENDENT_VALIDATION':
        lines+=['两个任务的较晚 Stage-1 状态在预设等预算 Decoder 比较中超过 E1。'
                '诊断区间已被历史研究接触，因此下一轮必须预注册独立新数据集与多个种子，'
                '保持同样的原始时间支持隔离和等预算 Decoder，比较潜变量与预测两种轮次选择。'
                '这尚未支持新损失或架构。']
    elif status=='GO_GENERALIZATION_OPTIMIZATION_AUDIT':
        lines+=['另立预注册实验检查原 Stage-1 学习率、早停、Encoder/Predictor 冻结时序、'
                '目标表示漂移和数据分布变化。本轮没有隔离唯一机制，也没有支持新 readout-aware loss。']
    elif status=='NO_GO_NEW_STATE_SELECTION_METHOD':
        lines+=['当前证据不支持跨任务一致的新状态选择方法。'
                '未来若重启该方向，应先在独立数据与多个种子上确认现象。']
    else:
        lines+=['两个任务的排序或有限的独立时间区块不足以支持稳定机制。'
                '把任务差异作为独立数据、多种子、预注册实验的待检验假设；'
                '不针对当前验证区间加模块或调整阈值。']
    return '\n'.join(lines)+'\n'

def run():
    test,checkpoint=prepare()
    environment=json.loads((OUT/'environment.json').read_text())
    s=read('stage1_epoch_metrics.csv');g=read('latent_geometry.csv')
    a=read('latent_geometry_alignment.csv');r=read('decoder_training_results.csv')
    c=read('decoder_training_curves.csv');b=read('paired_block_statistics.csv')
    assert len(r)==12 and len(c)==126 and len(b)==12
    decision=decide(s,r,c,b)
    (OUT/'research_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (OUT/'CROSS_EPOCH_DECODER_REPORT.md').write_text(cross_report(r,c,decision),encoding='utf-8')
    (OUT/'FINAL_RESEARCH_REPORT.md').write_text(report(s,g,a,r,c,b,decision,test,checkpoint,environment),encoding='utf-8')
    (OUT/'NEXT_RESEARCH_DECISION.md').write_text(next_report(decision),encoding='utf-8')
    print('FINAL_REPORTS_COMPLETE',decision['status'],flush=True)

if __name__=='__main__':run()
