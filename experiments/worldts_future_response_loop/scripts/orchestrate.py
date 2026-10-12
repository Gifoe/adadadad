from common import *
import subprocess,traceback
def run_command(script,*args,prefix):
    n=len(list((ROOT/'logs').glob(prefix+'_attempt*.log')))+1;log=ROOT/'logs'/(prefix+'_attempt'+str(n)+'.log');command=[sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,args)];start=time.time()
    with log.open('wb') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    write_json(log.with_suffix('.json'),{'command':command,'start_time':start,'end_time':time.time(),'exit_code':r.returncode,'sha256':sha(log)});print('COMMAND_EXIT',prefix,r.returncode,str(log),flush=True)
    if script=='train.py':
        out=ROOT/'runs'/prefix;out.mkdir(exist_ok=True);(out/('train_log_attempt'+str(n)+'.txt')).write_bytes(log.read_bytes())
    if r.returncode:raise RuntimeError(str(log))
def main():
    plan=json.loads((ROOT/'configs/preregistration.json').read_text());matrix=plan['experiments']
    run_command('link_data.py',prefix='data_link')
    for stage in [2,3,4]:
        for row in [r for r in matrix if r['stage']==stage]:
            p=ROOT/'runs'/row['run_id']/'metrics.json'
            if not p.is_file():
                write_json(ROOT/'STATUS.json',{'stage':stage,'status':'RUNNING','active_run':row['run_id'],'overall_complete':False});run_command('train.py','--run-id',row['run_id'],prefix=row['run_id'])
            m=json.loads(p.read_text());assert sha(m['checkpoint_path'])==m['checkpoint_sha256'];row.update({'status':'COMPLETE','mse':m['reproduced_mse'],'mae':m['reproduced_mae'],'validation_mae':m['validation_mae'],'train_seconds':m['train_seconds'],'result_path':str(p)})
            write_csv(ROOT/'EXPERIMENT_MATRIX.csv',matrix)
        run_command('post_analysis.py',prefix='stage'+str(stage)+'_checkpoint_cost_and_mechanism_audit')
        run_command('summarize.py','--stage',stage,prefix='stage'+str(stage)+'_report')
    # Validation-only allocation rule. Test scores never determine expansion.
    gains=[];selection=[]
    for ds in ['NP','PJM']:
        m=[json.loads(p.read_text()) for p in (ROOT/'runs').glob(ds+'_H360_*/metrics.json')];m=[x for x in m if x['seed']==2021]
        non=min([x for x in m if x['variant'] in ['B0','B1','B2','C3','C4']],key=lambda x:x['validation_mae']);loop=min([x for x in m if x['variant']=='C5'],key=lambda x:x['validation_mae']);gain=1-loop['validation_mae']/non['validation_mae'];gains.append(gain);selection.append({'dataset':ds,'nonrecurrent':non['run_id'],'shared_loop':loop['run_id'],'validation_mae_gain':gain})
    expand=np.mean(gains)>=.005 and max(gains)>=.01 and min(gains)>=-.02
    write_json(ROOT/'results/validation_expansion_decision.json',{'test_scores_used':False,'expand_three_seed':bool(expand),'selection':selection,'gain_mean':float(np.mean(gains)),'preregistration_sha256':sha(ROOT/'configs/preregistration.json')})
    for row in [r for r in matrix if r['optional']]:
        if expand:
            p=ROOT/'runs'/row['run_id']/'metrics.json'
            if not p.is_file():run_command('train.py','--run-id',row['run_id'],prefix=row['run_id'])
            m=json.loads(p.read_text());row.update({'status':'COMPLETE','mse':m['reproduced_mse'],'mae':m['reproduced_mae'],'validation_mae':m['validation_mae'],'train_seconds':m['train_seconds'],'result_path':str(p)})
        else:row.update({'status':'NOT_RUN','reason':'Preregistered validation-only positive-signal gate was not met; no test-driven stop or configuration selection.'})
        write_csv(ROOT/'EXPERIMENT_MATRIX.csv',matrix)
    verified=json.loads((ROOT/'results/independent_verification.json').read_text())
    if len(verified['results'])!=len(list((ROOT/'runs').glob('*/metrics.json'))):run_command('post_analysis.py',prefix='mechanism_robustness_and_depth')
    run_command('summarize.py','--stage','5',prefix='stage5_statistical_synthesis')
    print('ALL_RESEARCH_STAGES_EXECUTED',flush=True)
if __name__=='__main__':main()
