"""Post-training execution sanity; does not select or alter a model."""
import torch,numpy as np,json
from run_experiment import OUT,ROOT,batch,load_data,dump
from evaluate import load_model
torch.set_num_threads(1)
train,test=load_data();m=load_model('baseline');m.num_iterations=5
results={}
with torch.no_grad():
    for split,data in [('train',train),('test',test)]:
        results[split]={}
        rng=np.random.RandomState(42)
        for d in [1,2,3,4,5]:
            pool=np.flatnonzero(data['hops']==d)
            if not len(pool):continue
            ind=rng.choice(pool,min(512,len(pool)),replace=False);good=0;pad_good=0;agree=0;diff=0.
            for s in range(0,len(ind),128):
                ix=ind[s:s+128];x,mask,p,t=batch(data,ix);x7,mask7,p7,t7=batch(data,ix,length=7)
                with torch.autocast('cuda',dtype=torch.bfloat16):a=m(x,mask,p,True);b=m(x7,mask7,p7,True)
                pred=a.final_logits.argmax(-1);padded=b.final_logits.argmax(-1)
                good+=int((pred==t[:,4]).sum());pad_good+=int((padded==t[:,4]).sum());agree+=int((pred==padded).sum());diff=max(diff,float((a.final_logits-b.final_logits).abs().max()))
            results[split][str(d)]=dict(n=len(ind),correct=good,accuracy=good/len(ind),padded_correct=pad_good,prediction_agreement=agree/len(ind),max_padding_logit_difference=diff)
    x,mask,p,t=batch(test,np.flatnonzero(test['hops']==5)[:8]);m.num_iterations=5
    with torch.autocast('cuda',dtype=torch.bfloat16):a=m(x,mask,p,True)
    m.num_iterations=8
    with torch.autocast('cuda',dtype=torch.bfloat16):b=m(x,mask,p,True);noop=m(x,mask,p,True,patch=(2,b.state_hidden_per_recurrence[1]))
    prefix=max(float((a.state_logits_per_recurrence[k]-b.state_logits_per_recurrence[k]).abs().max()) for k in range(5))
    noop_error=float((noop.final_logits-b.final_logits).abs().max());assert prefix==0 and noop_error==0
    results['recurrence_prefix_max_difference']=prefix;results['exact_raw_state_noop_max_difference']=noop_error
    # Verify independent text records agree with tokenized arrays.
    vocab=json.loads((ROOT/'data/vocab.json').read_text());ids={v:i for i,v in enumerate(vocab)}
    for split,data in [('train',train),('test',test)]:
        raw=json.loads((ROOT/f'data/generated/{split}.json').read_text())
        for i in np.random.RandomState(42).choice(len(raw),1000,replace=False):
            r=raw[i];tok=[r['start_entity']]+r['relations']+['<state>'];assert [ids[x] for x in tok]==data['input_ids'][i,:len(tok)].tolist();assert ids[r['final_entity']]==data['state_targets'][i,-1]
    results['text_cache_checks']=2000;results['status']='PASS'
dump(OUT/'post_training_execution_sanity.json',results);print(json.dumps(results,indent=2),flush=True)
