import pathlib,json,hashlib,sys,re,collections
import torch
ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
result={}
for p in (ROOT/'official_artifacts').glob('*.pt'):
    ck=torch.load(p,map_location='cpu',weights_only=True)
    sd=ck.get('model_state_dict',ck)
    rows={k:tuple(v.shape) for k,v in sd.items() if torch.is_tensor(v)}
    metadata={k:v for k,v in ck.items() if k not in ['model_state_dict','optimizer_state_dict']}
    for k,v in list(metadata.items()):
        if torch.is_tensor(v):metadata[k]=v.tolist()
    optim=ck.get('optimizer_state_dict',{})
    result[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),keys=list(ck),metadata=metadata,state_shapes=rows,optimizer_groups=optim.get('param_groups'),optimizer_step_values=sorted(set(float(v['step']) for v in optim.get('state',{}).values() if 'step' in v)),all_parameters_finite=all(bool(torch.isfinite(v).all()) for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()))
    del ck,sd
    print(p.name,json.dumps({k:v for k,v in result[p.name].items() if k!='state_shapes'},default=str),flush=True)
for name in ['vocab.json','test_small.json','train.json','test.json']:
    p=ROOT/'official_artifacts'/name
    if not p.exists():continue
    data=json.loads(p.read_text(encoding='utf-8'));info=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),length=len(data),examples=data[:2])
    if name!='vocab.json':info['type_counts']=dict(collections.Counter(x.get('type','unknown') for x in data));info['hops_counts']=dict(collections.Counter(len(re.findall(r'<r_[^>]+>',x['input_text'])) for x in data))
    result[name]=info;print(name,json.dumps(info,ensure_ascii=False)[:5000],flush=True)
    del data
(OUT/'official_artifact_inspection.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
