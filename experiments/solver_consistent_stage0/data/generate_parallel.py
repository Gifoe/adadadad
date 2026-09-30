"""Deterministic ordered CPU workers around the official lexical adapter."""
import collections,concurrent.futures,contextlib,hashlib,io,json,os,pathlib,random,time
import numpy as np
from generate_prontoqa import ROOT,VERSION,sha,underlying_key

def block(task):
    depth,seed,count=task
    from official_adapter import official_generator
    g,concepts=official_generator(); import fol
    random.seed(seed); np.random.seed(seed)
    got=collections.Counter(); rows=[]; attempts=0; failures=0
    while len(rows)<count:
        attempts+=1
        if attempts>1000000: raise RuntimeError(f'Official generator exhausted block depth{depth}')
        with contextlib.redirect_stdout(io.StringIO()):
            q,query,formulas,_,answer,proof=g.generate_question(depth+1,concepts,formula_ordering='random',distractors='relevant')
        if q is None: failures+=1; continue
        y=int(answer=='True')
        if got[y]>=count//2: continue
        text=q+' '+query
        key=underlying_key(formulas,proof[-1].conclusion,g.available_entity_names,fol.fol_to_tptp)
        rows.append({'id':sha(text),'underlying_id':key,'text':text,'label':y,'depth':depth,'official_steps':depth+1}); got[y]+=1
    return rows,attempts,failures

def generate():
    counts={'train':100000,'validation':10000,'iid':10000,'ood_6':5000,'ood_8':5000,'ood_12':5000}
    out=ROOT/'data/generated'; out.mkdir(parents=True,exist_ok=True)
    workers=min(8,os.cpu_count() or 1); seen=set(); texts=set(); start=time.time()
    manifest={'seed':0,'official_commit':VERSION,'depth_definition':'inference hops; official num_deduction_steps=depth+1 includes axiom',
        'arguments':{'formula_ordering':'random','ontology':'fictional','deduction_rule':'ModusPonens','distractors':'relevant',
            'concept_names':'concept0..concept2047 via official Morphology.add_noun','property_family_extension':'32 disjoint 4-property families attribute0..attribute127 at official generate_theory input'},
        'workers':workers,'generation_seed_protocol':'ordered blocks, numpy.SeedSequence([0,split_index,depth,round,worker_index]); worker random and numpy RNG reset per block',
        'leakage_scope':'global canonical FOL facts plus proven conclusion, sentence order/entity/query polarity normalized; predicate-renamed structures can recur','splits':{}}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for split_index,(split,n) in enumerate(counts.items()):
            depths=[1,2,3,4] if split in ['train','validation','iid'] else [int(split.split('_')[-1])]
            quota=n//(2*len(depths)); got=collections.Counter(); attempts=0; failures=0; duplicates=0; round_id=0
            path=out/(split+'.jsonl')
            with path.open('w',encoding='utf-8') as f:
                while sum(got.values())<n:
                    pending=[d for d in depths if any(got[d,y]<quota for y in [0,1])]
                    tasks=[]
                    for i in range(workers):
                        d=pending[i%len(pending)]; seed=int(np.random.SeedSequence([0,split_index,d,round_id,i]).generate_state(1)[0]); tasks.append((d,seed,100))
                    for rows,a,failed in pool.map(block,tasks):
                        attempts+=a; failures+=failed
                        for row in rows:
                            d,y=row['depth'],row['label']
                            if got[d,y]>=quota: continue
                            if row['underlying_id'] in seen or row['id'] in texts: duplicates+=1; continue
                            seen.add(row['underlying_id']); texts.add(row['id']); got[d,y]+=1; f.write(json.dumps(row)+'\n')
                    round_id+=1; f.flush()
                    print(split,sum(got.values()),'/',n,'round',round_id,'elapsed',round(time.time()-start),flush=True)
            manifest['splits'][split]={'count':n,'label_distribution':{str(y):sum(got[d,y] for d in depths) for y in [0,1]},
                'depth_distribution':{str(d):sum(got[d,y] for y in [0,1]) for d in depths},
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'attempts':attempts,'failed_generations':failures,'duplicates_rejected':duplicates}
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    manifest['elapsed_sec']=time.time()-start
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__': generate()
