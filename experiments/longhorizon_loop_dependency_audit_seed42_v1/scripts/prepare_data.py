"""Exact executable references and deterministic matched state-chain diagnostics."""
import argparse,collections,copy,hashlib,importlib.util,json,math,pathlib,random,re,sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
REF=re.compile(r'^\$(var_\d+)\.([A-Za-z_]\w*)\$?$')

def typed_equal(a,b):
    if isinstance(a,bool) or isinstance(b,bool):return type(a) is type(b) and a==b
    if isinstance(a,(int,float)) and isinstance(b,(int,float)):
        return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9)
    if isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(typed_equal(a[k],b[k]) for k in a)
    return a==b

def finite_value(x):
    if isinstance(x,(int,float)) and not isinstance(x,bool):return math.isfinite(float(x)) and abs(x)<=1e20
    if isinstance(x,(tuple,list)):return all(finite_value(v) for v in x)
    if isinstance(x,dict):return all(finite_value(v) for v in x.values())
    return isinstance(x,(str,bool,type(None)))

def references(value,path=()):
    if isinstance(value,str):
        m=REF.fullmatch(value)
        if m:yield path,m.group(1),m.group(2),value
        elif value.startswith('$var'):raise ValueError('malformed variable reference')
    elif isinstance(value,list):
        for i,v in enumerate(value):yield from references(v,path+(i,))
    elif isinstance(value,dict):
        for k,v in value.items():yield from references(v,path+(k,))

def resolve(value,variables,override=None,path=()):
    if override is not None and path==override[0]:return override[1]
    if isinstance(value,str) and REF.fullmatch(value):
        m=REF.fullmatch(value);return variables[m.group(1)][m.group(2)]
    if isinstance(value,list):return [resolve(v,variables,override,path+(i,)) for i,v in enumerate(value)]
    if isinstance(value,dict):return {k:resolve(v,variables,override,path+(k,)) for k,v in value.items()}
    return value

def execute(calls,tools,module,override=None):
    """Success is separate from the return value, including valid False/zero."""
    variables={};results=[];specs={t['name']:t for t in tools}
    try:
        for j,call in enumerate(calls):
            label=call['label'].lstrip('$');assert label not in variables,'duplicate variable label'
            spec=specs[call['name']];assert len(spec['output_parameters'])==1,'unsupported multi-output function'
            args=resolve(call['arguments'],variables,override[1:] if override and override[0]==j else None)
            values=list(args.values())
            # Explicit resource exclusions before executing the official math
            # functions. No replacement implementations or rounded labels.
            if call['name'] in ['factorial','permutation','choose'] and any(isinstance(v,(int,float)) and abs(v)>500 for v in values):raise ValueError('factorial/combinatorics resource cap')
            if call['name']=='power' and (len(values)!=2 or abs(values[1])>64):raise ValueError('power resource cap')
            result=getattr(module,call['name'])(*values)
            if not finite_value(result):raise ValueError('nonfinite or resource-scale output')
            output_name=next(iter(spec['output_parameters']));variables[label]={output_name:result};results.append(result)
        return {'ok':True,'value':results[-1],'intermediate':results,'variables':variables}
    except Exception as e:return {'ok':False,'value':None,'error':type(e).__name__+': '+str(e)[:160],'failed_step':len(results)}

def nestful_rows(source):
    spec=importlib.util.spec_from_file_location('official_nestful_math',source/'data_v2/executable_functions/basic_functions.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    names={n for n in dir(module) if callable(getattr(module,n)) and getattr(getattr(module,n),'__module__',None)==module.__name__}
    rejects=collections.Counter();valid=[]
    for line in (source/'data_v2/nestful_data.jsonl').read_text(encoding='utf8').splitlines():
        row=json.loads(line);calls=row['output']
        if not (2<=len(calls)<=20):rejects['call_count_outside_2_to_20']+=1;continue
        if not all(c['name'] in names for c in calls):rejects['non_basic_math_function']+=1;continue
        ex=execute(calls,row['tools'],module)
        if not ex['ok']:rejects['execution_failure:'+ex['error'].split(':')[0]]+=1;continue
        if not typed_equal(ex['value'],row['gold_answer']):rejects['execution_gold_mismatch_typed']+=1;continue
        second=execute(calls,row['tools'],module)
        if not second['ok'] or not typed_equal(ex['intermediate'],second['intermediate']):rejects['nondeterministic_execution']+=1;continue
        labels={};edges=[];all_edges=[];depth=[];signature=[]
        try:
            for j,call in enumerate(calls):
                parents=[];sig_args=[]
                for path,label,field,raw in references(call['arguments']):
                    if label not in labels or field not in ex['variables'][label]:raise ValueError('invalid or future reference')
                    i=labels[label];parents.append(i);original=ex['variables'][label][field]
                    altered=[not original] if isinstance(original,bool) else [original+1,original-1] if isinstance(original,(int,float)) else []
                    meaningful=False
                    for replacement in altered:
                        changed=execute(calls,row['tools'],module,override=(j,path,replacement))
                        if changed['ok'] and not typed_equal(changed['value'],ex['value']):meaningful=True;break
                    edge={'producer':i,'consumer':j,'argument_path':list(path),'reference':raw,'semantically_active':meaningful}
                    all_edges.append(edge)
                    if meaningful:edges.append(edge)
                    else:rejects['excluded_pseudo_or_final_irrelevant_edge']+=1
                    sig_args.append([list(path),i,field])
                depth.append(1+max((depth[i] for i in parents),default=0));labels[call['label'].lstrip('$')]=j
                signature.append([call['name'],sorted(call['arguments']),sig_args])
        except Exception:rejects['invalid_dependency_graph']+=1;continue
        if not edges:rejects['no_semantically_active_dependency']+=1;continue
        reference='[\n';units=[]
        for j,call in enumerate(calls):
            text=json.dumps(call,ensure_ascii=False,separators=(',',':'));start=len(reference);reference+=text+(',\n' if j<len(calls)-1 else '\n')
            marker='"label":';at=text.index(marker)+len(marker);encoded=json.dumps(call['label'],ensure_ascii=False)
            labelspan=[start+at+1,start+at+len(encoded)-1]
            identity=re.search(r'var_(\d+)',call['label']);assert identity
            source_anchor_span=[labelspan[0]+identity.start(1),labelspan[0]+identity.end(1)]
            uses=[]
            for e in edges:
                if e['consumer']!=j:continue
                key=e['argument_path'][0];find_from=text.index(json.dumps(key)+':')+len(json.dumps(key))+1
                encoded_ref=json.dumps(e['reference']);offset=text.index(encoded_ref,find_from)
                use_span=[start+offset+1,start+offset+len(encoded_ref)-1];identifier=re.search(r'var_(\d+)',e['reference']);assert identifier
                uses.append({**e,'char_span':use_span,'anchor_char_span':[use_span[0]+identifier.start(1),use_span[0]+identifier.end(1)]})
            units.append({'index':j,'source_char_span':source_anchor_span,'source_full_label_char_span':labelspan,'unit_char_span':[start,start+len(text)],'uses':uses})
        reference+=']'
        sig=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
        instruction='Produce an executable JSON array of tool calls for this question. Each call must contain name, label (such as $var_1), and arguments. Reuse previous outputs as $var_N.result$. Return only the array.\nQuestion: '+row['input']+'\nAvailable tools: '+json.dumps(row['tools'],ensure_ascii=False,separators=(',',':'))
        valid.append({'sample_id':'nestful_'+row['sample_id'],'parent_id':'nestful_'+row['sample_id'],'dataset':'nestful_v2_math',
                      'original_prompt':row['input'],'tools':row['tools'],'prompt':instruction,'reference':reference,'reference_calls':calls,
                      'gold_answer':row['gold_answer'],'executed_intermediates':ex['intermediate'],'execution_ok':True,
                      'N':len(calls),'H':max(depth),'dag_edges':all_edges,'valid_dependency_edges':edges,'units':units,'structure_group':sig,
                      'semantic_scope':'variable-binding reference spans; numeric returns are not text in the original NESTFUL reference'})
    return valid,dict(rejects)

def select_nestful(rows):
    rng=random.Random(42);rng.shuffle(rows);unique={}
    for row in rows:unique.setdefault(row['structure_group'],row)
    if len(unique)<96:raise RuntimeError(f'Only {len(unique)} unique executable structures; never duplicate to fill96')
    strata=collections.defaultdict(list)
    for row in unique.values():strata[(min(row['H']//3,4),min(row['N']//4,4))].append(row)
    queue=[]
    while any(strata.values()):
        for key in sorted(strata):
            if strata[key]:queue.append(strata[key].pop())
    # Interleave independently of score, preserving similar structural strata.
    chosen=[]
    for index,row in enumerate(queue[:96]):
        row['split']='dev' if index%3==0 else 'audit';chosen.append(row)
    assert sum(r['split']=='dev' for r in chosen)==32 and sum(r['split']=='audit' for r in chosen)==64
    return chosen,{'executable_candidates':len(rows),'distinct_structural_groups':len(unique),'selected':96,'cross_split_structure_overlap':0}

def transition(state,operation):
    if operation=='ADD':return (state+1)%8
    if operation=='XOR':return state^1
    if operation=='INV':return 7-state
    if operation=='ROT':return ((state<<1)|(state>>2))&7
    raise ValueError(operation)

def chains():
    result=[];seen_initials=set();seen_ops=set()
    for split,count,seed in [('dev',16,42),('audit',64,1042)]:
        rng=random.Random(seed)
        for group in range(count):
            while True:
                initial=[rng.randrange(8) for _ in range(8)];operations=['ADD','XOR','INV','ROT']*4;rng.shuffle(operations)
                if tuple(initial) not in seen_initials and tuple(operations) not in seen_ops:break
            seen_initials.add(tuple(initial));seen_ops.add(tuple(operations))
            for renamed in [False,True]:
                alphabet='ABCDEFGH' if not renamed else 'IJKLMNOP'
                for H in [2,4,8,16]:
                    states=[];edges=[];lines=[];C=16//H
                    for j,op in enumerate(operations):
                        parent=j-1 if j%H else None;chain=j//H
                        value=initial[chain] if parent is None else states[parent]
                        states.append(transition(value,op));src=f'INIT{chain:02}' if parent is None else f'STEP{parent:02}'
                        lines.append(f'STEP{j:02}: {op} {src}')
                        if parent is not None:edges.append({'producer':parent,'consumer':j,'semantically_active':True})
                    tables='\n'.join(op+': '+','.join(alphabet[i]+'>'+alphabet[transition(i,op)] for i in range(8)) for op in ['ADD','XOR','INV','ROT'])
                    starts=' '.join(f'INIT{i:02}={alphabet[value]}' for i,value in enumerate(initial))
                    prompt='Execute exactly these16 deterministic symbol transitions, in listed order. Sources STEPxx refer to earlier outputs. Return only a JSON array of16 single-letter states, one per STEP00..STEP15. Use the exact lookup tables.\n'+tables+'\n'+starts+'\n'+'\n'.join(lines)
                    letters=[alphabet[s] for s in states];reference=json.dumps(letters,separators=(',',':'));units=[]
                    for j in range(16):
                        pos=1+j*4+1 # character index of each one-letter JSON value
                        assert reference[pos]==letters[j]
                        units.append({'index':j,'source_char_span':[pos,pos+1],'unit_char_span':[pos-1,pos+2],
                                      'uses':[dict(e,char_span=[pos,pos+1]) for e in edges if e['consumer']==j]})
                    parent_id=f'controlled_{split}_{group:03}'
                    result.append({'sample_id':parent_id+f'_H{H}_'+('rename' if renamed else 'base'),'parent_id':parent_id,
                                   'dataset':'controlled_symbol_chains_new_diagnostic','split':split,'H':H,'N':16,'independent_chains':C,
                                   'renamed':renamed,'alphabet':alphabet,'initial_states':initial,'operations':operations,
                                   'prompt':prompt,'reference':reference,'gold_answer':letters,'dag_edges':edges,'units':units,
                                   'operation_histogram':dict(collections.Counter(operations)),'semantic_scope':'actual intermediate symbol states',
                                   'primary':H in [4,16] and not renamed,'structure_group':parent_id})
    for split in ['dev','audit']:
        grouped=collections.defaultdict(list)
        for r in result:
            if r['split']==split:grouped[(r['parent_id'],r['renamed'])].append(r)
        for group in grouped.values():
            assert len({len(r['prompt']) for r in group})==1 and len({len(r['reference']) for r in group})==1
            assert all(r['operation_histogram']=={'ADD':4,'XOR':4,'INV':4,'ROT':4} for r in group)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--nestful',default=str(ROOT/'upstream/NESTFUL'));p.add_argument('--tokenize',action='store_true');a=p.parse_args()
    valid,rejects=nestful_rows(pathlib.Path(a.nestful));controlled=chains()
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    if a.tokenize:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/'models/TaH2-1.7B-max2',local_files_only=True)
        kept=[];overlong=[]
        for row in valid+controlled:
            prefix=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            full=prefix+row['reference'];encoding=tok(full,return_offsets_mapping=True,add_special_tokens=False)
            if len(encoding['input_ids'])>4096:
                overlong.append({'sample_id':row['sample_id'],'tokens':len(encoding['input_ids'])});continue
            row['prefix_text']=prefix;row['input_ids']=encoding['input_ids'];offsets=encoding['offset_mapping'];base=len(prefix)
            for unit in row['units']:
                def span(chars):
                    begin,end=[base+x for x in chars];indices=[i for i,(left,right) in enumerate(offsets) if left<end and right>begin]
                    assert indices and indices[0]>0;return [indices[0],indices[-1]+1]
                unit['source_token_span']=span(unit['source_char_span']);unit['source_anchor']=unit['source_token_span'][0]-1
                unit['unit_token_span']=span(unit['unit_char_span'])
                unit['predicted_anchor_token']=tok.decode([row['input_ids'][unit['source_anchor']+1]])
                for use in unit['uses']:
                    use['consumer_token_span']=span(use['char_span']);use['consumer_anchor']=span(use.get('anchor_char_span',use['char_span']))[0]-1
                    use['predicted_anchor_token']=tok.decode([row['input_ids'][use['consumer_anchor']+1]])
            row['prompt_tokens']=len(tok(prefix,add_special_tokens=False)['input_ids']);row['total_tokens']=len(row['input_ids']);kept.append(row)
        # Token-limit eligibility is checked before locking the cohort and
        # before observing any model score. Never truncate a selected input.
        selected,counts=select_nestful([r for r in kept if r['dataset']=='nestful_v2_math'])
        rows=selected+[r for r in kept if r['dataset']=='controlled_symbol_chains_new_diagnostic'];counts['token_limit_exclusions']=overlong
        if any(r['dataset']=='nestful_v2_math' for r in rows):
            counts['retained_nestful_by_split']=dict(collections.Counter(r['split'] for r in rows if r['dataset']=='nestful_v2_math'))
        path=out/'sample_manifest.jsonl'
    else:
        selected,counts=select_nestful(valid);rows=selected+controlled;path=out/'untokenized_samples.jsonl'
    if path.exists():
        previous=path.read_bytes();content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows).encode('utf8')
        assert previous==content,'A fixed sample manifest must never be overwritten with a different cohort'
    else:path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    info={'nestful_source_commit':'fc2c4123e73500a56185a5fb354f05d1c8b4890c','source_schema':['sample_id','input','output','tools','gold_answer'],
          'nestful':counts,'rejections':rejects,'controlled_groups':{'dev':16,'audit':64},'controlled_generated_inputs_with_rename':640,
          'controlled_primary_audit_inputs':128,'controlled_definition':'new deterministic diagnostic, not a public existing benchmark',
          'typed_bool_numeric_comparison':'False and0 are different; execution status separate from return value',
          'function_execution':'exact official pure-math basic_functions; call arguments positional in original ordered dictionary order',
          'total_manifest_rows':len(rows),'tokenized':a.tokenize,'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/('data_audit_tokenized.json' if a.tokenize else 'data_audit.json')).write_text(json.dumps(info,indent=2),encoding='utf8')
    print('DATA_PREPARATION_VERIFIED',json.dumps(info),flush=True)
if __name__=='__main__':main()
