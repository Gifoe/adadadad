import gc, hashlib, json, pathlib
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
ROOT=pathlib.Path(__file__).parent

def json_batches(path,size=1000):
    batch=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch)==size:yield batch;batch=[]
    if batch:yield batch

def prepare(c):
    target=ROOT/'data/tokenized';target.mkdir(parents=True,exist_ok=True);source=ROOT/'data/generated'
    tok=Tokenizer(models.BPE(unk_token='[UNK]'));tok.normalizer=normalizers.Lowercase()
    tok.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=False),pre_tokenizers.Whitespace()])
    tok.train_from_iterator((json.loads(l)['text'] for l in (source/'train.jsonl').open(encoding='utf-8')),
        trainers.BpeTrainer(vocab_size=c['vocab_size'],special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]']))
    tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
    assert [tok.token_to_id(x) for x in ['[PAD]','[UNK]','[CLS]','[SEP]']]==[0,1,2,3]
    tok.save(str(target/'tokenizer.json'))
    paths=sorted(source.glob('*.jsonl'));stats={};all_lengths={}
    for path in paths:
        lengths=[]
        for rows in json_batches(path):
            enc=tok.encode_batch([x['text'] for x in rows]);lengths.extend(len(x.ids) for x in enc);del enc,rows
        a=np.asarray(lengths,dtype=np.int32);all_lengths[path.stem]=a
        stats[path.stem]={'count':len(a),'max_length':int(a.max()),'p95_length':float(np.percentile(a,95)),'p99_length':float(np.percentile(a,99)),'truncation_rate_at_256':float((a>256).mean())}
    required=max(x['p99_length'] for x in stats.values());maxlen=max(c['max_length'],int(np.ceil(required/32)*32))
    for path in paths:
        n=stats[path.stem]['count'];ids=np.zeros((n,maxlen),dtype=np.int32);labels=np.empty(n,dtype=np.int8);depths=np.empty(n,dtype=np.int8);lengths=np.empty(n,dtype=np.int16);i=0
        for rows in json_batches(path):
            enc=tok.encode_batch([x['text'] for x in rows])
            for row,x in zip(rows,enc):
                vals=x.ids[:maxlen]
                if len(x.ids)>maxlen:vals[-1]=3
                ids[i,:len(vals)]=vals;labels[i]=row['label'];depths[i]=row['depth'];lengths[i]=len(vals);i+=1
            del enc,rows
        original=all_lengths[path.stem];truncated=original>maxlen
        np.savez_compressed(target/(path.stem+'.npz'),tokens=ids,labels=labels,depths=depths,lengths=lengths,original_lengths=original,truncated=truncated)
        stats[path.stem]['truncation_rate']=float(truncated.mean());del ids,labels,depths,lengths;gc.collect()
    c['max_length']=maxlen;c['actual_vocab_size']=tok.get_vocab_size();stats['chosen_max_length']=maxlen;stats['tokenizer_train_split_only']=True
    stats['normalization']='lowercase; digit spans isolated before whitespace BPE; same concept ID shared across case/plural forms'
    stats['tokenizer_sha256']=hashlib.sha256((target/'tokenizer.json').read_bytes()).hexdigest()
    (target/'audit.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');(ROOT/'configs/frozen.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    print(json.dumps({'actual_vocab_size':c['actual_vocab_size'],'max_length':maxlen,'tokenizer_sha256':stats['tokenizer_sha256']},indent=2),flush=True);return c
