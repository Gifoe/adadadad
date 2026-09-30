import json, pathlib, hashlib
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, normalizers
ROOT=pathlib.Path(__file__).parent

def prepare(c):
    target=ROOT/'data/tokenized'; target.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/generated'
    tok=Tokenizer(models.BPE(unk_token='[UNK]')); tok.normalizer=normalizers.Lowercase()
    tok.pre_tokenizer=pre_tokenizers.Sequence([pre_tokenizers.Digits(individual_digits=False),pre_tokenizers.Whitespace()])
    tok.train_from_iterator((json.loads(l)['text'] for l in (source/'train.jsonl').open(encoding='utf-8')),
        trainers.BpeTrainer(vocab_size=c['vocab_size'],special_tokens=['[PAD]','[UNK]','[CLS]','[SEP]']))
    tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',2),('[SEP]',3)])
    assert [tok.token_to_id(x) for x in ['[PAD]','[UNK]','[CLS]','[SEP]']]==[0,1,2,3]
    tok.save(str(target/'tokenizer.json'))
    stats={}; rows={}
    for path in sorted(source.glob('*.jsonl')):
        r=[json.loads(x) for x in path.open(encoding='utf-8')]; enc=tok.encode_batch([x['text'] for x in r]); rows[path.stem]=(r,enc)
        lengths=np.array([len(x.ids) for x in enc]); stats[path.stem]={'count':len(r),'max_length':int(lengths.max()),'p99_length':float(np.percentile(lengths,99)),'truncation_rate_at_256':float((lengths>256).mean())}
    # Decide using length audit only, before weights/training/results. Keep every split <=1% truncation.
    required=max(x['p99_length'] for x in stats.values())
    maxlen=max(c['max_length'],int(np.ceil(required/32)*32))
    for name,(r,enc) in rows.items():
        ids=np.zeros((len(r),maxlen),dtype=np.int32); lengths=[]
        for i,x in enumerate(enc):
            vals=x.ids[:maxlen]
            if len(x.ids)>maxlen: vals[-1]=3
            ids[i,:len(vals)]=vals; lengths.append(len(vals))
        np.savez_compressed(target/(name+'.npz'),tokens=ids,labels=np.array([x['label'] for x in r]),depths=np.array([x['depth'] for x in r]),lengths=lengths,
            original_lengths=np.array([len(x.ids) for x in enc]),truncated=np.array([len(x.ids)>maxlen for x in enc]))
        stats[name]['truncation_rate']=sum(len(x.ids)>maxlen for x in enc)/len(enc)
    c['max_length']=maxlen; c['actual_vocab_size']=tok.get_vocab_size()
    stats['chosen_max_length']=maxlen; stats['tokenizer_train_split_only']=True
    stats['normalization']='lowercase; digit spans isolated before whitespace BPE; same concept ID shared across case/plural forms'
    stats['tokenizer_sha256']=hashlib.sha256((target/'tokenizer.json').read_bytes()).hexdigest()
    (target/'audit.json').write_text(json.dumps(stats,indent=2))
    (ROOT/'configs/frozen.json').write_text(json.dumps(c,indent=2))
    return c
