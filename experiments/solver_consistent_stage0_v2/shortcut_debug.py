import sys,pathlib,json,collections
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'data'))
from official_adapter import official_generator
from audit_structural_shortcut import predict
G,_=official_generator();import fol
for line in (ROOT/'data/generated/validation.jsonl').open(encoding='utf-8'):
    r=json.loads(line);p,u=predict(r['text'])
    if p!=r['label']:
        print(json.dumps(r));print(p,u);break

