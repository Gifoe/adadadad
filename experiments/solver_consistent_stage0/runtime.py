import contextlib, hashlib, json, pathlib, random
import numpy as np
import torch
from models.common import Model, counts
ROOT=pathlib.Path(__file__).parent
NAMES=['vanilla_loop','step_conditioned_loop','vector_field']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available(): torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

def config(): return json.loads((ROOT/'configs/frozen.json').read_text())
def amp(): return torch.autocast('cuda',dtype=torch.bfloat16) if torch.cuda.is_available() else contextlib.nullcontext()
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
def device(): return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data(split):
    a=np.load(ROOT/'data/tokenized'/f'{split}.npz')
    return {key:a[key] for key in a.files}

def batch(data,indices,dev):
    # Dynamic length, identical indices imply identical sequence lengths across models.
    length=int(data['lengths'][indices].max())
    # Fixed 64-token buckets bound allocator shape diversity; masks retain actual lengths.
    length=min(data['tokens'].shape[1], ((length+63)//64)*64)
    x=torch.as_tensor(data['tokens'][indices,:length].astype(np.int64),device=dev)
    y=torch.as_tensor(data['labels'][indices],dtype=torch.long,device=dev)
    return x,y

def build(c,name):
    seed_all(c['seed']); base=Model(c,'vanilla_loop')
    if name=='vanilla_loop': return base
    seed_all(c['seed']); m=Model(c,name)
    shared=base.state_dict()
    if name=='vector_field':
        shared={('core.core.'+k[len('core.'):] if k.startswith('core.') else k):v for k,v in shared.items()}
    missing,extra=m.load_state_dict(shared,strict=False)
    assert not extra and all('condition' in x for x in missing)
    return m

def checkpoint_model(name,which='best'):
    c=config(); m=build(c,name).to(device())
    p=ROOT/'artifacts'/name/f'{which}.pt'
    saved=torch.load(p,map_location=device(),weights_only=False); m.load_state_dict(saved['model']); m.eval()
    return m,c,saved

def training_plan(n,c):
    rng=np.random.default_rng(c['seed']); order=[]; needed=c['updates']*c['effective_batch']
    while sum(len(x) for x in order)<needed: order.append(rng.permutation(n))
    indices=np.concatenate(order)[:needed].reshape(c['updates'],c['effective_batch'])
    budgets=np.random.default_rng(c['seed']+17001).choice(c['train_budgets'],c['updates'])
    digest=hashlib.sha256(indices.tobytes()+budgets.tobytes()).hexdigest()
    return indices,budgets,digest
