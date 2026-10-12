"""Evidence-based interpretation and provenance finalization; no training or hyperparameter edits."""
import pathlib,json,csv,statistics,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1];O=ROOT/'outputs'
def read(name):return json.loads((O/name).read_text(encoding='utf-8'))
def rows(name):
 with (O/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def append_once(name,marker,text):
 p=O/name;old=p.read_text(encoding='utf-8').split(marker)[0].rstrip();p.write_text(old+'\n\n'+marker+'\n\n'+text+'\n',encoding='utf-8')
def write_rows(name,items):
 with (O/name).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
def main():
 assert json.loads((ROOT/'STATUS.json').read_text())['overall_complete']
 a=read('audit_analysis.json');decision=read('research_decision.json');r=rows('training_results.csv');commands=read('execution_commands.json');assert len(r)==6 and all(x['exit_code']==0 for x in commands['steps'])
 checkpoint_index=[]
 for v in r:
  m=read('runs/'+v['dataset']+'_'+v['method']+'_s2021/metrics.json')
  for kind in ['best','last']:checkpoint_index.append({'dataset':v['dataset'],'method':v['method'],'kind':kind,'optimizer_step':m['best_step'] if kind=='best' else m['steps'],'absolute_server_path':m[kind+'_checkpoint'],'sha256':m[kind+'_checkpoint_sha256'],'format':'torch dictionary Model + step + initial_checkpoint_sha256','delivery':'server; Git contains index only'})
 write_rows('checkpoints/CHECKPOINT_INDEX.csv',checkpoint_index)
 (O/'checkpoints/README.md').write_text('# Full checkpoints\n\nAll12 best/last complete-model checkpoints exist on the server under D:\\worldts_readout_alignment_feasibility_seed2021_v1\\outputs\\checkpoints. CHECKPOINT_INDEX.csv records their exact paths, optimizer steps and SHA256. Selected checkpoint reloads match predictions with max difference0. Optimizer/RNG resume files and validation arrays are retained in outputs/runs; the final artifact manifest records their hashes. Large tensor files are not embedded in Git.\n',encoding='utf-8')
 lookup={(v['dataset'],v['method']):v for v in r};full={};groups={}
 for v in rows('error_alignment.csv'):groups.setdefault((v['dataset'],v['batch']),[]).append(v)
 components=[]
 for (ds,bi),g in groups.items():
  l1=statistics.mean(float(v['latent_l1']) for v in g);co=statistics.mean(float(v['latent_trajectory_cosine']) for v in g);rp=float(g[0]['batch_regularizer_pred']);rt=float(g[0]['batch_regularizer_target']);actual=float(g[0]['batch_full_latent_objective']);manual=10*l1+15*co+.05*rp+.05*rt
  assert abs(manual-actual)<1e-4
  components.append({'dataset':ds,'batch':int(bi),'n_origins':len(g),'full_batch_latent_objective':actual,'batch_latent_L1':l1,'batch_trajectory_cosine_loss':co,'prediction_legacy_VC':rp,'target_legacy_VC_constant':rt,'manual_component_reconstruction_abs_difference':abs(actual-manual),'interpretation':'batch objective; variance/covariance couples batch and time'})
 write_rows('latent_objective_components.csv',components)
 for ds in ['NP','PJM']:full[ds]=statistics.mean(v['full_batch_latent_objective'] for v in components if v['dataset']==ds)
 q1='**1. 是否存在 latent/readout 不一致？** 有弱状态方向对齐，但未发现系统性反向冲突。NP/PJM 全预测窗口的完整目标状态梯度余弦均值为0.08077/0.07179，负余弦比例均为0；固定8个 batch 的 Predictor 参数梯度余弦为0.57837/0.25416，负比例也均为0。NP 最后120步切片出现2/4,883个负余弦（0.041%），其余完整目标预测段均为0。高误差四分位同样没有负余弦；NP 高误差组余弦反而更高（0.1033 vs低误差0.0698），PJM 高/低误差组为0.0682/0.0697。所有六个最佳 checkpoint 在相同8个 batch 上的参数余弦仍全部为正。不能把接近正交的状态梯度写成真实参数更新冲突。'
 q2='**2. 是否是可利用的 Predictor 瓶颈？** 没有跨任务的证据。相对A0，A1/A2在NP上MAE分别恶化3.079%/1.727%，在PJM上改善1.456%/1.278%。NP只有5/14个360小时区块改善，PJM为9/14；PJM两种改进的360/720/1680小时区块bootstrap区间均包含0。A1在NP的360小时描述性95%区间为[-6.103%,-0.249%]，PJM为[-1.032%,3.997%]。区间针对已选模验证集，不能作为无偏泛化检验。继续训练本身使NP的A0相对Step0改善4.181%，超过A1的1.230%和A2的2.527%；因此不能将任何Step0改善归功于decoded监督。'
 q3='**3. 简单Decoded Supervision是否已经足够？** PJM的全部已观察小幅收益由普通decoded MAE（A1）取得；A2未超过A1，增加latent项并未提供额外收益。NP则A0最好。当前既没有必要性证据支持新Jacobian/readout-aware loss，也没有稳定跨任务收益支持直接替换原目标。'
 q4='**4. Idea A是否值得继续？** 对新型复杂readout-aware loss给出NO-GO。普通任务监督单独为INCONCLUSIVE：PJM约1.5%局部改善，但NP约3.1%相对退化，单seed且区块不确定性未排除0。初始latent L1为NP0.2526/PJM0.1983，预测MAE为0.20377/0.16604，参数对齐也明显不同，说明不能宣称机制普适。'
 q5='**5. 下一轮解决什么？** 本轮未找到简单监督无法解决的明确新机制瓶颈；不启动Jacobian loss或新表示机制。若继续验证普通任务监督，唯一具体问题是“PJM的小幅收益能否在未参与选择的数据集和额外预设seed上重复，同时解释NP为何A0更好”。冻结Encoder/Decoder的本轮实验无法因果区分Encoder表示不足、Decoder几何和时间分布变化，不能据此编造其中任何一个失败机制。'
 training_gradients=rows('training_gradient_trajectory.csv');train_summary=[]
 for ds in ['NP','PJM']:
  for method in ['A0','A1','A2']:
   g=[float(v['predictor_gradient_cosine']) for v in training_gradients if v['dataset']==ds and v['method']==method];train_summary.append({'dataset':ds,'method':method,'sampled_checks':len(g),'mean_parameter_cosine':statistics.mean(g),'minimum_parameter_cosine':min(g),'negative_checks':sum(v<0 for v in g),'mode':'Predictor train; not the fixed eval validation audit'})
 write_rows('training_gradient_summary.csv',train_summary)
 q1+=' 训练模式下按步数稀疏抽样的'+str(len(training_gradients))+'次参数诊断中有'+str(sum(float(v['predictor_gradient_cosine'])<0 for v in training_gradients))+'次负余弦（NP A0为1/17，PJM A0为1/21、A2为2/21，其余0）。这说明部分更新存在短暂冲突；它既不是全训练batch的普查比例，也没有改变固定验证审计与跨任务干预的否定证据。原始轨迹及汇总均保留。'
 text='\n\n'.join([q1,q2,q3,q4,q5]);append_once('FINAL_RESEARCH_REPORT.md','## Evidence-based answers (Chinese)',text);append_once('NEXT_RESEARCH_DECISION.md','## Concrete decision evidence',q1+'\n\n'+q2+'\n\n'+q5)
 append_once('ALIGNMENT_AUDIT.md','## Observed conflict and dataset dependence',q1+'\n\nEqual-batch mean official full latent objective: NP='+str(full['NP'])+', PJM='+str(full['PJM'])+'. Components and verified weighted reconstruction are saved in latent_objective_components.csv. This scalar mean refers to batch objectives, not independent sample full losses.\n\nNP/PJM 360-hour latent-L1 vs forecast-MAE block Pearson=0.6379/0.6155, Spearman=0.6088/0.6967. Full state cosine vs error Pearson=0.6465/-0.1273. These differ across tasks and depend on block length; at1680 hours only3 blocks remain. Decoded gap and forecast error are almost identical because target reconstruction is small, not because a prediction lower bound has been established. Target reconstruction MAE=0.00509/0.01087, compared with forecast MAE=0.20377/0.16604.\n')
 append_once('PREFLIGHT_REPORT.md','## Numerical execution and validation-order correction','Preflight attempts1–4 are preserved in outputs/logs. Attempt1 used batch32 for independent test prediction while the original artifact used batch64; its maximum difference2.384185791015625e-6 failed the original1e-6 tolerance. Returning to the original batch64 yields bitwise identical original test forecasts for all four tasks. Attempts2–4 then exposed the shuffled/unweighted validation issue. The official validation loader used shuffle=True and an unweighted mean across batches, including a short tail batch; historical ordering was not archived. All new comparisons use the same chronological batch32 partition with the unchanged official criterion. The current validator matches the official validator exactly on the same batches. Historical randomized scores fall inside the mathematically possible tail-partition range, which is a consistency check, not a proof of the lost historical order. No prediction tolerance was relaxed and no candidate test forecasts were generated.')
 formal=sum(x['seconds'] for x in commands['steps'] if x['name'].startswith('formal_'));peak=max(float(x['peak_allocated_GPU_MiB']) for x in r);timing={'formal_command_wall_seconds':formal,'training_and_periodic_validation_seconds':sum(float(x['seconds_training_and_periodic_validation']) for x in r),'pipeline_wall_seconds':commands['seconds'],'successful_preflight_seconds':read('preflight_metrics.json')['seconds'],'initial_gradient_audit_seconds':read('alignment_summary.json')['seconds'],'peak_process_owned_allocated_gpu_mib':peak,'actual_optimizer_steps':sum(int(x['steps']) for x in r),'formal_fits':6,'smoke_fits':6,'seconds_do_not_include_manual_setup_and_failed_preflight_debug':True}
 (O/'resource_summary.json').write_text(json.dumps(timing,indent=2),encoding='utf-8');append_once('FINAL_RESEARCH_REPORT.md','## Measured resources',json.dumps(timing,indent=2));print(json.dumps({'interpretation':'NO-GO new complex loss; ordinary decoded supervision INCONCLUSIVE','resources':timing,'reports_finalized':True}))
if __name__=='__main__':main()
