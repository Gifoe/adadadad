"""Deterministic parallel generation for Stage-0 V2 symmetric data."""
from __future__ import annotations
import argparse,collections,concurrent.futures,contextlib,hashlib,io,json,os,pathlib,random,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT='0a6412b6fddf46324a1cb96e066dd7b3d89b87d6'
from symmetric_adapter import adapt,GENERATOR_VERSION
from graph_hash import invariant_hash

def sha_bytes(x):return hashlib.sha256(x).hexdigest()
def canonical_id(formulas,conclusion,entities,to_tptp):
    import re
    s='\n'.join(sorted(to_tptp(x) for x in formulas))+'\n'+to_tptp(conclusion)
    for e in entities:s=re.sub(r'\b'+re.escape(e)+r'\b','ENTITY',s)
    return sha_bytes(s.encode())

def worker(task):
    depth,seed,count=task
    import sys
    legacy=ROOT.parent/'solver_consistent_stage0';sys.path[:0]=[str(legacy/'data'),str(legacy/'vendor/prontoqa')]
    from official_adapter import official_generator
    g,concepts=official_generator();import fol
    random.seed(seed);np.random.seed(seed);rows=[];got=collections.Counter();attempts=failures=adapter_failures=0;adapter_failure_reasons=collections.Counter()
    while len(rows)<count:
        attempts+=1
        if attempts>2_000_000:raise RuntimeError(f'generation exhausted depth={depth}')
        with contextlib.redirect_stdout(io.StringIO()):x=g.generate_question(depth+1,concepts,formula_ordering='random',distractors='relevant')
        if x[0] is None:failures+=1;continue
        q,query,formulas,_,answer,proof=x;y=int(answer=='True')
        if got[y]>=count//2:continue
        try:q,query,formulas,answer,proof,meta=adapt(g,concepts,q,query,formulas,answer,proof)
        except (ValueError,AssertionError) as exc:
            adapter_failures+=1;adapter_failure_reasons[str(exc) or type(exc).__name__]+=1;continue
        qtext=query.split('True or false: ',1)[1].strip().rstrip('.');qfol=g.parse_sentence(qtext,g.morphology,False)
        first=proof[0].conclusion;a=first.operand if isinstance(first,fol.FOLNot) else first;selected=a.args[0].constant
        text=q+' '+query;row={'id':sha_bytes(text.encode()),'underlying_id':canonical_id(formulas,qfol,g.available_entity_names,fol.fol_to_tptp),
            'graph_hash':invariant_hash(formulas,qfol,selected,fol),'text':text,'label':y,'depth':depth,'official_steps':depth+1,'generator_version':GENERATOR_VERSION}
        rows.append(row);got[y]+=1
    return rows,attempts,failures,adapter_failures,dict(adapter_failure_reasons)

def generate(output,counts,workers=None,seed_namespace=1000):
    output.mkdir(parents=True,exist_ok=True);workers=workers or min(8,os.cpu_count() or 1);start=time.time();seen_text=set();seen_logic=set();seen_graph=set()
    manifest={'seed':0,'seed_namespace':seed_namespace,'generator_version':GENERATOR_VERSION,'official_commit':UPSTREAM_COMMIT,
      'depth_definition':'actual Modus Ponens hops; official_steps=depth+1 includes starting axiom','repair':'predicate-renamed connected proof component grounded on a different entity; isolated opposite rule removed',
      'workers':workers,'deduplication':'global exact text, canonical FOL/entity-normalized, and predicate-renaming-invariant signed graph hash','splits':{}}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
      for split_index,(split,n) in enumerate(counts.items()):
        depths=[1,2,3,4] if split in ('train','validation','iid') else [int(split.rsplit('_',1)[1])];quota=n//(2*len(depths));got=collections.Counter();attempts=failures=adapter_failures=duplicates=0;adapter_failure_reasons=collections.Counter();round_id=0
        path=output/f'{split}.jsonl'
        with path.open('w',encoding='utf-8') as f:
          while sum(got.values())<n:
            pending=[d for d in depths if any(got[d,y]<quota for y in (0,1))]
            tasks=[]
            for i in range(workers):
              d=pending[i%len(pending)];s=int(np.random.SeedSequence([seed_namespace,split_index,d,round_id,i]).generate_state(1)[0]);tasks.append((d,s,40 if n<10000 else 100))
            for rows,a,failed,af,reasons in pool.map(worker,tasks):
              attempts+=a;failures+=failed;adapter_failures+=af
              adapter_failure_reasons.update(reasons)
              for row in rows:
                d,y=row['depth'],row['label']
                if got[d,y]>=quota:continue
                if row['id'] in seen_text or row['underlying_id'] in seen_logic or row['graph_hash'] in seen_graph:duplicates+=1;continue
                seen_text.add(row['id']);seen_logic.add(row['underlying_id']);seen_graph.add(row['graph_hash']);got[d,y]+=1;f.write(json.dumps(row,separators=(',',':'))+'\n')
            round_id+=1;f.flush();print(split,sum(got.values()),'/',n,'round',round_id,'duplicates',duplicates,'elapsed',round(time.time()-start),flush=True)
            if round_id>max(100,n//20):raise RuntimeError(f'isomorphism-unique generation exhausted: {split} {got}')
        manifest['splits'][split]={'count':n,'label_distribution':{str(y):sum(got[d,y] for d in depths) for y in (0,1)},'depth_distribution':{str(d):sum(got[d,y] for y in (0,1)) for d in depths},
          'sha256':sha_bytes(path.read_bytes()),'attempts':attempts,'official_failures':failures,'adapter_failures':adapter_failures,'adapter_failure_reasons':dict(adapter_failure_reasons),'duplicates_rejected':duplicates}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['count']=sum(x['count'] for x in manifest['splits'].values());manifest['elapsed_sec']=time.time()-start;manifest['manifest_payload_sha256']=sha_bytes(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,default=ROOT/'data/generated');p.add_argument('--pilot',action='store_true');p.add_argument('--workers',type=int,default=0);p.add_argument('--seed-namespace',type=int,default=1000);a=p.parse_args()
    counts={'train':100000,'validation':10000,'iid':10000,'ood_6':5000,'ood_8':5000,'ood_12':5000}
    if a.pilot:counts={'train':800,'validation':80,'iid':80,'ood_6':40,'ood_8':40,'ood_12':40}
    generate(a.output,counts,a.workers or None,a.seed_namespace)
