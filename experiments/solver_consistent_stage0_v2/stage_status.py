"""Append-only stage status records plus the current state snapshot."""
import argparse, hashlib, json, pathlib, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parent;ART=ROOT/'artifacts';ART.mkdir(exist_ok=True)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
def git_head():
    try:return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:return '6071738c90d468f1f2f89552ede44669ac07c3da'
def record(stage,event,passed=None,exit_code=None,evidence=None,state=None):
    now=time.time();manifest=ROOT/'data/generated/manifest.json';config=ROOT/'configs/frozen.json'
    if not config.exists():config=ROOT/'configs/base.json'
    item={'stage':stage,'event':event,'unix_time':now,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(now)),
          'passed':passed,'exit_code':exit_code,'input_config_sha256':sha(config),'data_manifest_sha256':sha(manifest),
          'git_commit':git_head(),'evidence':evidence or []}
    with (ART/'status_history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,separators=(',',':'))+'\n')
    current={'pipeline_state':state or stage,'updated_utc':item['utc'],'current_stage':stage,'last_event':event,
             'formal_seed0_complete':False,'formal_updates':{'vanilla_loop':0,'step_conditioned_loop':0,'vector_field':0},'last_record':item}
    (ART/'status.json').write_text(json.dumps(current,indent=2),encoding='utf-8')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage');p.add_argument('event',choices=['START','END']);p.add_argument('--passed',choices=['true','false']);p.add_argument('--exit-code',type=int);p.add_argument('--evidence',action='append');p.add_argument('--state');a=p.parse_args()
    record(a.stage,a.event,None if a.passed is None else a.passed=='true',a.exit_code,a.evidence,a.state)
