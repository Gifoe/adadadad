"""Full independent semantic, checksum, balance, and dedup audit for Stage-0 V2."""
from __future__ import annotations
import argparse, collections, hashlib, json, pathlib, sys, time

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(HERE)]
from official_adapter import official_generator
from graph_hash import invariant_hash
from generate_v2 import canonical_id, sha_bytes, GENERATOR_VERSION

def split_text(text):
    facts,query=text.split('True or false: ',1)
    return [s for s in facts.strip().rstrip('.').split('. ') if s],query.strip().rstrip('.')

def literal(formula,fol):
    neg=isinstance(formula,fol.FOLNot); atom=formula.operand if neg else formula
    if not isinstance(atom,fol.FOLFuncApplication) or len(atom.args)!=1:return None
    arg=atom.args[0]
    return atom.function,getattr(arg,'constant',None),neg

def rule(formula,fol):
    if not isinstance(formula,fol.FOLForAll) or not isinstance(formula.operand,fol.FOLIfThen):return None
    a=literal(formula.operand.antecedent,fol); b=literal(formula.operand.consequent,fol)
    if a is None or b is None:return None
    return (a[0],a[2]),(b[0],b[2])

def parse_row(row,g,fol):
    sentences,qtext=split_text(row['text']); formulas=[]
    for sentence in sentences:
        f=g.parse_sentence(sentence,g.morphology,False)
        if f is None:raise ValueError(f'unparseable sentence: {sentence}')
        rendered=g.inflect(g.yield_tokens(g.formula_to_clause(f,g.morphology,False)),end_punctuation='.')
        if g.parse_sentence(rendered[:-1],g.morphology,False)!=f:raise ValueError('parser round-trip mismatch')
        formulas.append(f)
    q=g.parse_sentence(qtext,g.morphology,False)
    if q is None:raise ValueError('unparseable query')
    rendered=g.inflect(g.yield_tokens(g.formula_to_clause(q,g.morphology,False)),end_punctuation='.')
    if g.parse_sentence(rendered[:-1],g.morphology,False)!=q:raise ValueError('query parser round-trip mismatch')
    return formulas,q

def entailment(formulas,entities,fol):
    known={}; rules=[]
    for f in formulas:
        r=rule(f,fol)
        if r: rules.append(r); continue
        l=literal(f,fol)
        if l and l[1] is not None: known[(l[0],l[1],l[2])]=0
    changed=True
    while changed:
        changed=False
        for (ap,an),(bp,bn) in rules:
            for entity in entities:
                a=(ap,entity,an); b=(bp,entity,bn)
                if a in known:
                    d=known[a]+1
                    if d<known.get(b,10**9):known[b]=d;changed=True
    return known

def audit(data_dir,out_path):
    started=time.time();g,_=official_generator();import fol
    manifest=json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    result={'scope':'all rows','data_dir':str(data_dir.resolve()),'generator_version':GENERATOR_VERSION,
            'started_unix':started,'splits':{},'failure_count':0,'failures':[]}
    seen={'id':set(),'underlying_id':set(),'graph_hash':set()}; total=0
    expected_fields={'id','underlying_id','graph_hash','text','label','depth','official_steps','generator_version'}
    def fail(row,split,kind,detail=None):
        result['failure_count']+=1
        if len(result['failures'])<200:result['failures'].append({'id':row.get('id'),'split':split,'kind':kind,'detail':detail})
    for path in sorted(data_dir.glob('*.jsonl')):
        split=path.stem; counts=collections.Counter(); checked=0
        if split not in manifest.get('splits',{}):fail({},split,'missing_manifest_entry')
        if split in manifest.get('splits',{}) and sha_bytes(path.read_bytes())!=manifest['splits'][split]['sha256']:fail({},split,'sha256_mismatch')
        for line_no,line in enumerate(path.open(encoding='utf-8'),1):
            checked+=1;total+=1
            try: row=json.loads(line)
            except Exception as exc:fail({},split,'invalid_json',str(exc));continue
            if set(row)!=expected_fields:fail(row,split,'field_set',str(sorted(set(row)^expected_fields)))
            counts[(row.get('depth'),row.get('label'))]+=1
            if row.get('generator_version')!=GENERATOR_VERSION:fail(row,split,'generator_version')
            if row.get('official_steps')!=row.get('depth',-99)+1:fail(row,split,'official_steps')
            if hashlib.sha256(row.get('text','').encode()).hexdigest()!=row.get('id'):fail(row,split,'id_mismatch')
            for key in seen:
                value=row.get(key)
                if value in seen[key]:fail(row,split,'global_duplicate',key)
                seen[key].add(value)
            try:
                formulas,q=parse_row(row,g,fol); qlit=literal(q,fol)
                if qlit is None or qlit[1] is None:raise ValueError('query is not a ground unary literal')
                entities={l[1] for f in formulas if (l:=literal(f,fol)) and l[1] is not None}
                entities.add(qlit[1]); known=entailment(formulas,entities,fol)
                target=qlit;opp=(qlit[0],qlit[1],not qlit[2]);proof=target if row['label'] else opp
                if (target in known)!=bool(row['label']):fail(row,split,'label_semantics',{'target':target in known})
                if (opp in known)==bool(row['label']):fail(row,split,'opposite_semantics',{'opposite':opp in known})
                if target in known and opp in known:fail(row,split,'contradiction')
                if known.get(proof)!=row['depth']:fail(row,split,'minimum_depth',{'actual':known.get(proof),'declared':row['depth']})
                uid=canonical_id(formulas,q,g.available_entity_names,fol.fol_to_tptp)
                if uid!=row['underlying_id']:fail(row,split,'underlying_id_mismatch')
                gh=invariant_hash(formulas,q,qlit[1],fol)
                if gh!=row['graph_hash']:fail(row,split,'graph_hash_mismatch')
            except Exception as exc:fail(row,split,'exception',repr(exc))
        result['splits'][split]={'checked':checked,'depth_label_counts':{f'{d}:{y}':n for (d,y),n in sorted(counts.items())}}
        m=manifest.get('splits',{}).get(split,{})
        if checked!=m.get('count'):fail({},split,'count_mismatch',{'actual':checked,'manifest':m.get('count')})
        vals=list(counts.values())
        if vals and len(set(vals))!=1:fail({},split,'depth_label_imbalance',dict(result['splits'][split]['depth_label_counts']))
    payload=dict(manifest); claimed=payload.pop('manifest_payload_sha256',None)
    actual=sha_bytes(json.dumps(payload,sort_keys=True,separators=(',',':')).encode())
    if claimed!=actual:fail({},'manifest','manifest_payload_sha256',{'claimed':claimed,'actual':actual})
    if total!=manifest.get('count'):fail({},'manifest','total_count',{'actual':total,'manifest':manifest.get('count')})
    result.update({'checked':total,'passed':result['failure_count']==0,'ended_unix':time.time(),'elapsed_sec':time.time()-started,
                   'manifest_payload_sha256':claimed,'unique_counts':{k:len(v) for k,v in seen.items()}})
    out_path.parent.mkdir(parents=True,exist_ok=True);out_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('checked','failure_count','passed','elapsed_sec')},indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=pathlib.Path,default=ROOT/'data/generated');p.add_argument('--output',type=pathlib.Path,default=ROOT/'artifacts/semantic_data_audit.json');a=p.parse_args()
    r=audit(a.data_dir,a.output);raise SystemExit(0 if r['passed'] else 2)
