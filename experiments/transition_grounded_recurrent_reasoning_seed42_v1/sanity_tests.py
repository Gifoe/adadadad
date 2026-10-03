"""Rerun required sanity tests without training or changing saved weights."""
import torch
from run_experiment import seed,make_model,CK,unit,load_data
torch.set_num_threads(1);seed()
train,test=load_data();model=make_model(218).cuda()
model.load_state_dict(torch.load(CK/'initial_weights.pt',weights_only=True))
unit(model,train,test)
print('SANITY_PASS independent baseline CE vs lambda0',flush=True)
