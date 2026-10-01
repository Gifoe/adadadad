import json,os,pathlib,platform,subprocess,sys,time,traceback
ROOT=pathlib.Path(__file__).parent
os.chdir(ROOT)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
(ROOT/'artifacts').mkdir(exist_ok=True)

def status(stage,**kwargs):
    data={'stage':stage,'pid':os.getpid(),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs}
    (ROOT/'artifacts/status.json').write_text(json.dumps(data,indent=2))
    print('STATUS',data,flush=True)

def command(args,output=None):
    status('command',command=args)
    if output:
        with output.open('w',encoding='utf-8') as f: subprocess.run([sys.executable]+args,stdout=f,stderr=subprocess.STDOUT,check=True)
    else: subprocess.run([sys.executable]+args,check=True)

def shortcut_audit():
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    result={}
    datasets={s:[json.loads(l) for l in (ROOT/'data/generated'/f'{s}.jsonl').open(encoding='utf-8')] for s in ['train','validation','iid','ood_6','ood_8','ood_12']}
    for mode in ['query_only','bag_of_words']:
        def texts(rows): return [r['text'].split('True or false: ')[-1] if mode=='query_only' else r['text'] for r in rows]
        tf=TfidfVectorizer(max_features=8192,ngram_range=(1,1)); train=tf.fit_transform(texts(datasets['train']))
        clf=LogisticRegression(max_iter=300,random_state=0).fit(train,[r['label'] for r in datasets['train']])
        result[mode]={s:float(np.mean(clf.predict(tf.transform(texts(rows)))==np.array([r['label'] for r in rows]))) for s,rows in datasets.items() if s!='train'}
    # Removing distractors can make the task an XOR of premise/query negation.
    # Test this explicitly: a unigram linear classifier cannot detect parity reliably.
    result['negation_parity_no_training']={s:float(np.mean(np.array([int(r['text'].lower().split().count('not')%2==0) for r in rows])==np.array([r['label'] for r in rows]))) for s,rows in datasets.items()}
    result['negation_parity_caution']='If near perfect, inference hops are unnecessary for this dataset; report shortcut-dominated conclusions.'
    (ROOT/'artifacts/shortcut_audit.json').write_text(json.dumps(result,indent=2))

def main():
    import torch,yaml
    from tokenize_data import prepare
    from runtime import NAMES,build
    from models.common import counts
    c=json.loads((ROOT/'configs/base.json').read_text())
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'total_vram_mb':torch.cuda.get_device_properties(0).total_memory/2**20,
        'deterministic_algorithms':'enabled, warn_only=False; unsupported nondeterministic operations raise','CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG']}
    (ROOT/'artifacts/environment.json').write_text(json.dumps(environment,indent=2))
    command(['-m','pip','freeze'],ROOT/'artifacts/environment_packages.txt')
    if not (ROOT/'data/generated/manifest.json').exists(): command(['data/generate_parallel.py'])
    status('A2 tokenizer')
    if not (ROOT/'configs/frozen.json').exists(): c=prepare(c)
    else: c=json.loads((ROOT/'configs/frozen.json').read_text())
    # Audit the actually chosen length before model training. The length256 probe
    # is insufficient if deep OOD tokenization requires a longer positional range.
    if c['max_length']>256 and not (ROOT/'artifacts/frozen_length_memory.json').exists():
        from runtime import amp,sync,seed_all
        sizing=[]
        for micro in [64,32,16,8]:
            m=build(c,'step_conditioned_loop').cuda().train()
            x=torch.randint(4,c.get('actual_vocab_size',c['vocab_size']),(micro,c['max_length']),device='cuda')
            torch.cuda.reset_peak_memory_stats()
            try:
                with amp(): loss=m(x,8).float().square().mean()
                loss.backward(); sync(); peak=torch.cuda.max_memory_allocated()/2**20
                sizing.append({'micro_batch':micro,'max_length':c['max_length'],'peak_vram_mb':peak})
                fits=peak<environment['total_vram_mb']*.75
            except torch.cuda.OutOfMemoryError:
                sizing.append({'micro_batch':micro,'oom':True}); fits=False
            del m,x
            if 'loss' in locals(): del loss
            torch.cuda.empty_cache()
            if fits:
                c['micro_batch']=micro; break
        else: raise RuntimeError('No safe microbatch for chosen max length')
        (ROOT/'artifacts/frozen_length_memory.json').write_text(json.dumps(sizing,indent=2))
        (ROOT/'configs/frozen.json').write_text(json.dumps(c,indent=2))
        with (ROOT/'EXPERIMENT_CHANGELOG.md').open('a',encoding='utf-8') as f:
            f.write(f'\n- Final pretraining length audit selected max_length={c["max_length"]}; sizing chose identical micro_batch={c["micro_batch"]} for all models at <75% of GPU VRAM. Full sizing evidence artifacts/frozen_length_memory.json.\n')
    for name in NAMES:
        (ROOT/'configs'/f'{name}.yaml').write_text(yaml.safe_dump({'model':name,**c}))
        m=build(c,name); print('PARAMETERS',name,counts(m),flush=True); del m
    command(['-m','pytest','tests','-q'],ROOT/'artifacts/unit_tests.txt')
    command(['data/audit_semantics.py'])
    status('A1 shortcut audit'); shortcut_audit()
    shortcut=json.loads((ROOT/'artifacts/shortcut_audit.json').read_text())
    if shortcut['negation_parity_no_training']['iid']>=.95:
        raise RuntimeError('Invalid dataset: near-perfect negation-parity shortcut; training gated')
    command(['data/audit_structural_shortcut.py'])
    structural=json.loads((ROOT/'artifacts/structural_shortcut_audit.json').read_text())
    if structural['splits']['iid']['all']['accuracy']>=.95:
        command(['abort_report.py'])
        status('blocked',decision='UNCLEAR',reason='No-training structural shortcut solves >=95% IID; formal training prohibited on invalid task')
        return
    # Successful gates are reusable; checkpoints never cross from sanity into formal training.
    for name in NAMES:
        gate=ROOT/'artifacts/sanity'/name/'gate.json'
        if not gate.exists() or not json.loads(gate.read_text())['passed']:
            status('Check A depth1',model=name); command(['train.py','--model',name,'--sanity'])
    for name in NAMES:
        status('A8 formal training',model=name); command(['train.py','--model',name])
    for name in NAMES:
        status('A9-A11 evaluation',model=name); command(['evaluate.py','--model',name])
    status('A12 numerical diagnostic'); command(['diagnostics.py'])
    for name in NAMES:
        status('A13 latency',model=name); command(['benchmark_latency.py','--model',name])
    status('A14-A15 aggregation and report'); command(['aggregate_results.py']); command(['make_figures.py'])
    status('complete',decision=json.loads((ROOT/'artifacts/decision.json').read_text())['decision'])

if __name__=='__main__':
    status('initializing')
    try: main()
    except BaseException as e:
        status('failed',error=str(e),traceback=traceback.format_exc()); traceback.print_exc(); sys.exit(1)
