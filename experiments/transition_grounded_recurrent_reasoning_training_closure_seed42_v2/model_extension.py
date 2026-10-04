"""Extend official recurrent block without changing its layers or initialization."""
import pathlib,sys
from types import SimpleNamespace
import torch
import torch.nn.functional as F
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent/'official'))
from gpt_utils_extrapolation import RecurrentGPT2Block,CompositionDataset
from transformers import GPT2Config

class TransitionRecurrentGPT2(RecurrentGPT2Block):
    def forward(self,input_ids,attention_mask=None,state_pos=None,return_intermediate=False,patch=None):
        if state_pos is None:state_pos=attention_mask.sum(1)-1 if attention_mask is not None else torch.full((input_ids.shape[0],),input_ids.shape[1]-1,device=input_ids.device,dtype=torch.long)
        h=self.dropout(self.token_embedding(input_ids));initial=h
        mask=None if attention_mask is None else (1-attention_mask[:,None,None,:].to(h.dtype))*-10000.
        full=[];states=[];logits=[];rows=torch.arange(h.shape[0],device=h.device)
        for k in range(1,self.num_iterations+1):
            if self.input_injection and k>1:h=h+initial
            for block in self.blocks:h=block(h,attention_mask=mask)[0]
            if patch is not None and patch[0]==k:
                h=h.clone();h[rows,state_pos]=patch[1].to(h.dtype)
            if return_intermediate:
                full.append(h);q=h[rows,state_pos];states.append(q);logits.append(self.lm_head(self.ln_f(q)))
        normalized=self.ln_f(h)
        if not return_intermediate:return SimpleNamespace(logits=self.lm_head(normalized))
        return SimpleNamespace(hidden_states_per_recurrence=full,state_hidden_per_recurrence=states,state_logits_per_recurrence=logits,final_logits=logits[-1],logits=self.lm_head(normalized))

def make_model(vocab_size):
    config=GPT2Config(vocab_size=vocab_size,n_positions=50,n_ctx=50,n_embd=768,n_layer=4,n_head=12,embd_pdrop=0.,attn_pdrop=0.,resid_pdrop=0.,_attn_implementation='eager')
    return TransitionRecurrentGPT2(config,5,positional_embedding_type='none',input_injection=False,c_scale=0.)

def losses(out,targets,coefficient):
    final=F.cross_entropy(out.final_logits.float(),targets[:,4])
    intermediate=torch.stack([F.cross_entropy(z.float(),targets[:,k]) for k,z in enumerate(out.state_logits_per_recurrence[:-1])]).mean()
    return final+coefficient*intermediate,final,intermediate
