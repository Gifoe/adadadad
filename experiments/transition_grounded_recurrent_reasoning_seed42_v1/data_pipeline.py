"""Use actual official generators; extend records with table-replayed state paths."""
import ast,json,pathlib,random,hashlib
import numpy as np
from tqdm import tqdm
ROOT=pathlib.Path(__file__).resolve().parent
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def official_functions():
    source=(ROOT/'official/getNhopfact.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('build_atomic','build_nhop_facts_fast','split_facts')]
    ns={'np':np,'random':random,'tqdm':tqdm}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'official/getNhopfact.py','exec'),ns)
    return ns
def generate():
    random.seed(42);np.random.seed(42)
    ns=official_functions()
    entities,relations,atomic=ns['build_atomic'](200,10)
    vocab=['<pad>']+entities+relations+['<mask>','<sep>','<a>','</a>','<q>','</q>','<state>']
    ids={x:i for i,x in enumerate(vocab)}
    table={h+':'+r:t for h,r,t in atomic}
    def record(fact,hop,i):
        start=fact[0];rels=fact[1:-1];e=start;path=[]
        for r in rels:e=table[e+':'+r];path.append(e)
        assert e==fact[-1]
        inp=''.join(fact[:-1])+'<state>'
        return dict(sample_id=f'D{hop:02d}_{i:05d}',input_text=inp,target_text=inp+e+'</a>',hop_count=hop,start_entity=start,relations=rels,state_path=path,final_entity=e,state_pos=hop+1)
    trains={1:[record(f,1,i) for i,f in enumerate(atomic)]};tests={}
    generated={d:ns['build_nhop_facts_fast'](atomic,d,15750) for d in range(2,21)}
    # Official getNhopfact consumes five example draws/hop before splitting.
    for d in range(2,21):random.sample(generated[d],5)
    for d in range(2,21):
        facts=[record(f,d,i) for i,f in enumerate(generated[d])]
        a,b=ns['split_facts'](facts,15000,750)
        trains[d]=a;tests[d]=b
        assert not ({x['input_text'] for x in a}&{x['input_text'] for x in b})
    data=ROOT/'data';gen=data/'generated';gen.mkdir(parents=True,exist_ok=True)
    allrecords=trains[1]+[x for d in range(2,21) for x in trains[d]+tests[d]]
    for x in random.Random(42).sample(allrecords,1000):
        e=x['start_entity']
        for r,t in zip(x['relations'],x['state_path']):e=table[e+':'+r];assert e==t
        assert e==x['final_entity']
    train=trains[1]+[x for d in range(2,6) for x in trains[d]]
    test=[x for d in (2,3,4,5,6,8,10,12,16,20) for x in tests[d]]
    for name,records in [('train',train),('test',test)]:
        dump(gen/(name+'.json'),records)
        n=len(records);x=np.zeros((n,22),dtype=np.int64);paths=np.zeros((n,24),dtype=np.int64);hops=[]
        for i,r in enumerate(records):
            tok=[r['start_entity']]+r['relations']+['<state>']
            x[i,:len(tok)]=[ids[t] for t in tok]
            for k in range(24):paths[i,k]=ids[r['state_path'][min(k,r['hop_count']-1)]]
            hops.append(r['hop_count'])
        np.savez(gen/(name+'.npz'),input_ids=x,state_targets=paths,hops=hops,state_pos=np.array(hops)+1,sample_ids=[r['sample_id'] for r in records])
    # Keep all splits for reproducibility; only <=5 appears in train.json/npz.
    for d in range(2,21):dump(gen/f'{d}hop_train.json',trains[d]);dump(gen/f'{d}hop_test.json',tests[d])
    dump(data/'vocab.json',vocab);dump(data/'transition_table.json',table)
    dump(data/'generation_config.json',dict(seed=42,entities=200,relations=10,max_hop=20,generator='official functions extracted verbatim with AST to avoid module top-level I/O',chains_per_hop=15750,train_per_hop=15000,test_per_hop=750,train_max_hop=5))
    dump(data/'dataset_stats.json',dict(atomic=2000,total_generated=len(allrecords),actual_training=len(train),test=len(test),train_hops={str(d):len(trains[d]) for d in range(1,6)},test_hops={str(d):len(tests[d]) for d in tests},sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in gen.glob('*.npz')}))
    dump(data/'integrity_check.json',dict(status='PASS',random_replays=1000,all_generated_paths_replayed=len(allrecords),mismatches=0,split_overlap=0,training_ood_examples=0,permutation_relations=all(len({table[h+':'+r] for h in entities})==200 for r in relations)))
    print('DATA_PASS',len(train),len(test),flush=True)
if __name__=='__main__':generate()
