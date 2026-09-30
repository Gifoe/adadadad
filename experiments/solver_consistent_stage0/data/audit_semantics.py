"""Independent label/minimum-hop audit, never used to construct training data."""
import json,pathlib,sys
import numpy as np
from official_adapter import official_generator
ROOT=pathlib.Path(__file__).resolve().parents[1]

def audit():
    g,_=official_generator(); import fol
    def parts(f): return sum((parts(x) for x in f.operands),[]) if isinstance(f,fol.FOLAnd) else [f]
    def atom(f,entity):
        neg=isinstance(f,fol.FOLNot); f=f.operand if neg else f
        assert isinstance(f,fol.FOLFuncApplication) and len(f.args)==1
        arg=f.args[0]
        return f.function,(arg.constant if isinstance(arg,fol.FOLConstant) else entity),neg
    result={'scope':'64 deterministic examples per available split/depth; audit only, no training supervision','splits':{},'failures':[]}
    for path in sorted((ROOT/'data/generated').glob('*.jsonl')):
        rows=[json.loads(l) for l in path.open(encoding='utf-8')]; selected=[]
        for d in sorted({r['depth'] for r in rows}):
            candidates=[r for r in rows if r['depth']==d]; ix=np.random.default_rng(6600+d).choice(len(candidates),64,replace=False); selected.extend(candidates[i] for i in ix)
        checked=0
        for row in selected:
            facts,query=row['text'].split('True or false: ')
            q=g.parse_sentence(query.strip().rstrip('.'),g.morphology,False)
            inner=q.operand if isinstance(q,fol.FOLNot) else q; entity=inner.args[0].constant
            target=atom(q,entity); opposite=(target[0],target[1],not target[2]); known={}; rules=[]
            for sentence in facts.strip().rstrip('.').split('. '):
                f=g.parse_sentence(sentence,g.morphology,False)
                if isinstance(f,fol.FOLForAll):
                    implication=f.operand; assert isinstance(implication,fol.FOLIfThen)
                    antecedents=[atom(x,entity) for x in parts(implication.antecedent)]
                    rules.extend((antecedents,atom(x,entity)) for x in parts(implication.consequent))
                else:
                    for x in parts(f): known[atom(x,entity)]=0
            changed=True
            while changed:
                changed=False
                for antecedents,consequent in rules:
                    if all(a in known for a in antecedents):
                        depth=1+max(known[a] for a in antecedents)
                        if depth<known.get(consequent,100000): known[consequent]=depth; changed=True
            proof_target=target if row['label'] else opposite
            valid=(target in known)==bool(row['label']) and (opposite in known)!=bool(row['label']) and known.get(proof_target)==row['depth']
            if not valid: result['failures'].append({'id':row['id'],'split':path.stem,'requested_depth':row['depth'],'actual_minimum_depth':known.get(proof_target),'target_entailed':target in known,'opposite_entailed':opposite in known,'label':row['label']})
            checked+=1
        result['splits'][path.stem]={'checked':checked}
    result['passed']=not result['failures']
    (ROOT/'artifacts').mkdir(exist_ok=True)
    (ROOT/'artifacts/semantic_data_audit.json').write_text(json.dumps(result,indent=2))
    print('Semantic/minimum-hop audit',result['passed'],result['splits'],flush=True)
    if not result['passed']: raise RuntimeError('Semantic data audit failed')

if __name__=='__main__': audit()
