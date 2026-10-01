"""Label-free structural shortcut audit; does not execute any proof steps."""
import pathlib,json,re,collections
ROOT=pathlib.Path(__file__).resolve().parents[1]
def predict(text):
    facts,query=text.split('True or false: ')
    qm=re.search(r' is (not )?(?:a |an )?(.+?)\.?$',query.strip())
    assert qm,query
    qneg=bool(qm[1]); target=qm[2].rstrip('.')
    frequency=collections.Counter(re.findall(r'\bconcept(\d+)s?\b',facts,re.IGNORECASE))
    candidates=[]
    for s in facts.strip().rstrip('.').split('. '):
        if re.search(r'\b'+re.escape(target)+r'\.?$',s):
            source=re.search(r'\bconcept(\d+)s?\b',s,re.IGNORECASE)
            if source and s.startswith(('Every ','Each ','All ','Concept')):
                consequent_negative=bool(re.search(r'\b(?:is|are) not '+re.escape(target)+r'$',s))
                candidates.append((frequency[source[1]],consequent_negative))
    if not candidates:return 0,False
    candidates.sort(reverse=True)
    unique=len(candidates)==1 or candidates[0][0]>candidates[1][0]
    return int(candidates[0][1]==qneg),unique
if __name__=='__main__':
    out={'method':'Select query-goal rule with most frequent antecedent concept, then compare its negation to query; no premise reachability or chain execution; no training','splits':{}}
    for path in sorted((ROOT/'data/generated').glob('*.jsonl')):
        groups=collections.defaultdict(lambda:[0,0,0])
        for line in path.open(encoding='utf-8'):
            r=json.loads(line); p,unique=predict(r['text'])
            for key in ['all',str(r['depth'])]:
                groups[key][0]+=int(p==r['label']);groups[key][1]+=1;groups[key][2]+=int(unique)
        out['splits'][path.stem]={k:{'accuracy':v[0]/v[1],'n':v[1],'unique_choice_fraction':v[2]/v[1]} for k,v in groups.items()}
    (ROOT/'artifacts').mkdir(exist_ok=True);(ROOT/'artifacts/structural_shortcut_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

