import argparse,json,time,torch
from runtime import ROOT,build,amp,sync,load_data,batch,device
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--micro',type=int,required=True);a=p.parse_args()
c=json.loads((ROOT/'configs/frozen.json').read_text());c['micro_batch']=a.micro
dev=device();data=load_data('train');np=__import__('numpy');idx=np.argsort(data['original_lengths'])[-a.micro:];m=build(c,a.model).to(dev).train();x,y=batch(data,idx,dev)
torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();row={'model':a.model,'micro_batch':a.micro,'sequence_length':int(x.shape[1]),'budget':8,'checkpointing':c['activation_checkpointing'],'attention_backend':c['attention_backend']}
try:
    m.zero_grad(set_to_none=True)
    with amp():loss=torch.nn.functional.cross_entropy(m(x,8),y)
    loss.backward();sync();row.update({'status':'PASS','loss':float(loss.detach()),'seconds':time.perf_counter()-begin,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mb':torch.cuda.max_memory_reserved()/2**20,'total_vram_mb':torch.cuda.get_device_properties(0).total_memory/2**20})
except torch.cuda.OutOfMemoryError as exc:
    row.update({'status':'OOM','error':str(exc),'total_vram_mb':torch.cuda.get_device_properties(0).total_memory/2**20})
(ROOT/'artifacts').mkdir(exist_ok=True)
with (ROOT/'artifacts/hardware_probe.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
