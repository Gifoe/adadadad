"""Reconstruct paths from official released atomic facts, not regenerated data."""
import collections,hashlib,json,pathlib,re,random
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent;DATA=ROOT/'data';ART=ROOT/'official_artifacts'
PAT=re.compile(r'<[^>]+>')
def dump(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def prepare(H_train=6):
    DATA.mkdir(exist_ok=True)
    vocab=json.loads((ART/'vocab.json').read_text());vocab=['<pad>']+vocab if '<pad>' not in vocab else vocab
    assert len(vocab)==217 and '<state>' not in vocab
    ids={t:i for i,t in enumerate(vocab)};dump(DATA/'vocab.json',vocab)
    raw=json.loads((ART/'train.json').read_text());table={};train_records=[]
    for i,r in enumerate(raw):
        tok=PAT.findall(r['input_text']);d=sum(t.startswith('<r_') for t in tok)
        if d==1:
            target=PAT.findall(r['target_text']);tail=next(t for t in reversed(target) if t.startswith('<e_'))
            key=tok[0]+':'+tok[1]
            assert key not in table or table[key]==tail;table[key]=tail
        if d<=min(5,H_train):train_records.append((i,r,d))
    assert len(table)==2000 and len(train_records)==62000
    dump(DATA/'transition_table.json',table);del raw
    selected_D=list(range(2,H_train+1))+[d for d in [6,8,10,12,16,20] if d>H_train]
    test_raw=json.loads((ART/'test.json').read_text());test_records=[]
    for i,r in enumerate(test_raw):
        match=re.fullmatch(r'(\d+)hop_test',r.get('type',''))
        if match and int(match[1]) in selected_D:test_records.append((i,r,int(match[1])))
    counts=collections.Counter(d for _,_,d in test_records)
    assert all(counts[d]==750 for d in selected_D),counts
    del test_raw
    record_sets={};source_ids={}
    def convert(rows,name):
        x=np.zeros((len(rows),50),dtype=np.int64);target=np.zeros((len(rows),24),dtype=np.int64);hops=[];records=[];sample_ids=[]
        for j,(source_i,r,d) in enumerate(rows):
            tok=PAT.findall(r['input_text']);assert len(tok)==d+1 and tok[0].startswith('<e_')
            e=tok[0];path=[]
            for rel in tok[1:]:e=table[e+':'+rel];path.append(e)
            expected=next(t for t in reversed(PAT.findall(r['target_text'])) if t.startswith('<e_'))
            assert e==expected,(name,source_i,e,expected)
            x[j,:len(tok)]=[ids[t] for t in tok]
            for k in range(24):target[j,k]=ids[path[min(k,d-1)]]
            sid=f'official_{name}_{source_i:07d}';sample_ids.append(sid);hops.append(d)
            records.append(dict(sample_id=sid,source_index=source_i,input_text=r['input_text'],target_text=r['target_text'],start_entity=tok[0],relations=tok[1:],state_path=path,final_entity=e,hop_count=d,source_type=r.get('type')))
        np.savez(DATA/f'{name}.npz',input_ids=x,state_targets=target,hops=np.array(hops),sample_ids=np.array(sample_ids))
        dump(DATA/f'{name}_records.json',records);record_sets[name]=records;source_ids[name]=[r['input_text'] for r in records]
    convert(train_records,'train');convert(test_records,'test')
    assert not set(source_ids['train'])&set(source_ids['test'])
    all_selected=record_sets['train']+record_sets['test'];sample=random.Random(42).sample(all_selected,1000)
    for r in sample:
        e=r['start_entity']
        for rel,target in zip(r['relations'],r['state_path']):e=table[e+':'+rel];assert e==target
    dump(DATA/'integrity_check.json',dict(status='PASS',all_paths_replayed=len(all_selected),random_replays=1000,train_test_overlap=0,training_hop_limit=min(5,H_train),training_records=len(train_records),test_counts=dict(counts),source_type_labels_not_used_for_training_hop=True,permutation_relations=all(len({table[f'<e_{e}>:<r_{r}>'] for e in range(200)})==200 for r in range(10)),source_sha256={n:hashlib.sha256((ART/n).read_bytes()).hexdigest() for n in ['train.json','test.json','vocab.json']}))
    rng=np.random.RandomState(42);flat=[]
    while len(flat)<5000*128:flat.extend(rng.permutation(len(train_records)).tolist())
    np.save(DATA/'batch_plan.npy',np.array(flat[:5000*128],dtype=np.int32).reshape(5000,128))
    dump(DATA/'batch_plan_manifest.json',dict(seed=42,updates=5000,batch_size=128,sha256=hashlib.sha256((DATA/'batch_plan.npy').read_bytes()).hexdigest()))
    print('OFFICIAL_DATA_REPLAY_PASS',len(train_records),len(test_records),dict(counts),flush=True)
if __name__=='__main__':prepare()
