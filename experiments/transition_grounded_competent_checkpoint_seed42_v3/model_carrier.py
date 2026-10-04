"""Read/patch the existing prediction position; no parameters added."""
import pathlib,sys
from types import SimpleNamespace
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'official'))
from gpt_utils_extrapolation import RecurrentGPT2Block
from transformers import GPT2Config

class CarrierRecurrentGPT2(RecurrentGPT2Block):
    def forward(self,input_ids,attention_mask=None,return_states=False,patch=None):
        if not return_states and patch is None:
            return super().forward(input_ids,attention_mask)
        h=self.dropout(self.token_embedding(input_ids));initial=h
        if self.positional_embedding_type!='none':raise ValueError('This audited checkpoint is NoPE')
        mask=None if attention_mask is None else (1-attention_mask[:,None,None,:].to(h.dtype))*-10000.
        states=[];logits=[]
        for k in range(1,self.num_iterations+1):
            if self.input_injection and k>1:h=h+initial
            for block in self.blocks:h=block(h,attention_mask=mask)[0]
            if patch is not None and patch[0]==k:
                h=h.clone();h[:,-1]=patch[1].to(h.dtype)
            states.append(h[:,-1])
            # Exact existing decoder including final LN and full-position GEMM.
            logits.append(self.lm_head(self.ln_f(h))[:,-1])
        return SimpleNamespace(states=states,state_logits=logits,final_logits=logits[-1])

def model(vocab_size=217):
    cfg=GPT2Config(vocab_size=vocab_size,n_positions=50,n_ctx=50,n_embd=768,n_layer=4,n_head=12,embd_pdrop=0.,attn_pdrop=0.,resid_pdrop=0.,_attn_implementation='eager')
    return CarrierRecurrentGPT2(cfg,2,positional_embedding_type='none',input_injection=False,c_scale=0.)

def objectives(out,target,coefficient):
    final=torch.nn.functional.cross_entropy(out.final_logits.float(),target[:,-1])
    transition=torch.stack([torch.nn.functional.cross_entropy(z.float(),target[:,k]) for k,z in enumerate(out.state_logits[:-1])]).mean()
    return final+coefficient*transition,final,transition
