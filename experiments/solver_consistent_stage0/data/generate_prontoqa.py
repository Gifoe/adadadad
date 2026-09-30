"""Calls the unmodified official generator; no independently invented reasoning rules."""
import argparse, collections, contextlib, hashlib, io, json, os, pathlib, random, re, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION = '0a6412b6fddf46324a1cb96e066dd7b3d89b87d6'

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def underlying_key(formulas, conclusion, entity_names, to_tptp):
    # Canonicalizes ordering, entity names and query polarity. Opposite-label
    # questions about the same facts cannot cross splits. Predicate lexicon stays.
    text = '\n'.join(sorted(to_tptp(x) for x in formulas)) + '\n' + to_tptp(conclusion)
    for name in entity_names:
        text = re.sub(r'\b' + name + r'\b', 'ENTITY', text)
    return sha(text)

def generate(output, counts, seed=0):
    from official_adapter import official_generator
    g,concepts=official_generator()
    import fol
    random.seed(seed); np.random.seed(seed)
    output.mkdir(parents=True, exist_ok=True)
    seen_text, seen_logic = set(), set()
    manifest = {'seed':seed, 'official_commit':VERSION,
                'depth_definition':'ModusPonens inference hops; official num_deduction_steps=depth+1 includes starting axiom',
                'arguments':{'formula_ordering':'random','ontology':'fictional','distractors':'relevant','deduction_rule':'ModusPonens','concept_names':'concept0..concept2047, registered through official Morphology.add_noun','property_family_extension':'32 disjoint 4-property families attribute0..attribute127 supplied to official generate_theory API'},
                'leakage_scope':'global canonical FOL facts plus proven conclusion, sentence order/entity/query polarity normalized; predicate-renamed isomorphic structures can recur',
                'splits':{}}
    start = time.time()
    for split, n in counts.items():
        depths = [1,2,3,4] if split in ('train','validation','iid') else [int(split.split('_')[-1])]
        quotas = {(d,y):n//(2*len(depths)) for d in depths for y in (0,1)}
        assert sum(quotas.values()) == n
        got = collections.Counter(); attempts = 0; failures = 0; duplicates = 0
        path = output / (split+'.jsonl')
        with path.open('w',encoding='utf-8') as out:
            while sum(got.values()) < n:
                pending = [d for d in depths if any(got[d,y]<quotas[d,y] for y in (0,1))]
                d = random.choice(pending); attempts += 1
                if attempts > max(10000,n*200): raise RuntimeError(f'Generation exhausted {split}: {got}')
                with contextlib.redirect_stdout(io.StringIO()):
                    q,query,formulas,_,answer,proof = g.generate_question(d+1,concepts,formula_ordering='random',distractors='relevant')
                if q is None: failures += 1; continue
                assert answer in ('True','False')
                y = int(answer == 'True')
                if got[d,y] >= quotas[d,y]: continue
                text = q+' '+query
                key = underlying_key(formulas,proof[-1].conclusion,g.available_entity_names,fol.fol_to_tptp)
                text_key = sha(text)
                if key in seen_logic or text_key in seen_text: duplicates += 1; continue
                seen_logic.add(key); seen_text.add(text_key); got[d,y] += 1
                out.write(json.dumps({'id':text_key,'underlying_id':key,'text':text,'label':y,'depth':d,'official_steps':d+1})+'\n')
                if sum(got.values()) % 5000 == 0: print(split,sum(got.values()),'elapsed',round(time.time()-start),flush=True)
        manifest['splits'][split] = {'count':n,'label_distribution':{str(y):sum(got[d,y] for d in depths) for y in (0,1)},
            'depth_distribution':{str(d):sum(got[d,y] for y in (0,1)) for d in depths},
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'attempts':attempts,'failed_generations':failures,'duplicates_rejected':duplicates}
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['elapsed_sec']=time.time()-start
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=pathlib.Path,default=ROOT/'data/generated'); p.add_argument('--pilot',action='store_true'); a=p.parse_args()
    counts={'train':100000,'validation':10000,'iid':10000,'ood_6':5000,'ood_8':5000,'ood_12':5000}
    if a.pilot: counts={'train':800,'validation':80,'iid':80,'ood_6':40,'ood_8':40,'ood_12':40}
    generate(a.output,counts)
