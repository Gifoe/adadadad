from common import *
def main():
    p=ROOT/'configs/preregistration.json'
    if p.exists():print('PREREGISTRATION_REUSED',sha(p));return
    plan={'version':'v1','created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'source_commit':SOURCE_COMMIT,'initialization':'checkpoint_init: same verified seed2021 stage2 checkpoint for every method and continuation seed; not independent pretraining seeds','seeds':[2021,2022,2023],'budget':{'stage1_epochs':20,'stage2_epochs':20,'patience':5,'batch_size':32,'loss':'unchanged official L1+trajectory cosine+legacy_vc then decoder MAE+0.3 reconstruction MAE','scheduler':'official type3','learning_rates':'published per task'},'selection':'official stage1 validation latent objective / stage2 decoded validation MAE; architectures and K frozen before new test evaluation','positive_gate':'After fixed seed2021 long-task matrix: mean relative validation MAE gain of best shared loop over strongest nonrecurrent control >=0.005, at least one task >=0.01, neither task worse by >0.02. Gate uses validation only. If false, still run minimal Stage4, synthesize Stage5, and mark optional multi-seed high-cost expansion NOT_RUN.','main_comparator':'Strongest predeclared nonrecurrent control by validation MAE, selected separately per task; freeze before expanded-seed test access','test_time_depths':[1,2,4],'bootstrap':{'repetitions':2000,'block_lengths':[360,720,1680],'unit':'contiguous test origin time blocks, never iid overlapping windows','seed':9281},'engineering_thresholds':{'mean_relative_mse_gain':.03,'same_checkpoint_K4_over_K1':.01,'mae_not_worse':True},'cost_match_tolerances':{'total_parameter_relative_diff':.10,'whole_prediction_latency_relative_diff':.25},'candidate_timing_note':'K1 one-shot has same new parameter count as K4 shared recurrence; K4 unshared has same operation count; width128 one-shot controls whole-model parameter/cost approximately, with actual cost measurements and mismatches disclosed','synthetic':{'a':[.2,.95],'noise_std':[.05],'points':4096,'lookback':24,'horizon':48,'seed':419,'latent_dim':16,'condition_dim':8,'causal_patch':3,'stage_epochs':[20,20],'methods':['B0','B1','C5K4']}}
    rows=[]
    def add(stage,ds,h,v,k=1,width=64,s=2021,optional=False):
        rows.append({'run_id':f'{ds}_H{h}_{v}_K{k}_s{s}','stage':stage,'dataset':ds,'horizon':h,'variant':v,'k':k,'width':width,'seed':s,'initialization':'checkpoint_init','optional':optional,'status':'PLANNED'})
    for ds in ['NP','PJM']:
        for v,w in [('B0',64),('B1',64),('B2',96)]:add(2,ds,360,v,width=w)
        for v,k,w in [('C1',4,64),('C3',4,64),('C4',1,128),('C5',2,64),('C5',4,64)]:add(3,ds,360,v,k,w)
        for v in ['NOANCHOR','FULL']:add(4,ds,360,v,4)
        for v,k in [('B0',1),('B1',1),('C5',4)]:add(4,ds,24,v,k)
        for s in [2022,2023]:
            for v,k,w in [('B0',1,64),('B1',1,64),('B2',1,96),('C3',4,64),('C4',1,128),('C5',2,64),('C5',4,64)]:add(5,ds,360,v,k,w,s,True)
    plan['experiments']=rows;write_json(p,plan);write_csv(ROOT/'EXPERIMENT_MATRIX.csv',rows)
    (ROOT/'RESEARCH_HYPOTHESIS_TRACKER.md').write_text('# H1–H3 preregistered tracker\n\nH1: local future-condition support. Source inspection supports the pathway claim, numerical verification pending.\n\nH2: one-shot temporal mixing improves out-of-sample errors under matched two-stage continued training. PENDING.\n\nH3: anchored shared recurrence exceeds strongest reasonable nonrecurrent control under measured parameter/cost budgets. PENDING. No positive conclusion is assumed.\n',encoding='utf-8')
    print('CONFIGURATION_AND_TEST_SET_FROZEN',sha(p),len(rows),flush=True)
if __name__=='__main__':main()
