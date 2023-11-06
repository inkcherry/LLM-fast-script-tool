import torch
import habana_frameworks.torch.core as htcore
import time


class time_context():
    def __init__(self,op_name) -> None:
        self.op_name=op_name
    def __enter__(self):
        torch.hpu.synchronize()
        self.st=time.time()
        return
    def __exit__(self,exc_type, exc_value, traceback):
        torch.hpu.synchronize()
        en = time.time()
        du = en-self.st
        print(f"op:{self.op_name},time:{du}")
        
        
# chatglm2
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor: # rope_cache: [1024, 1, 32, 2]
    
    with time_context("bf_stack"):

        # x: [sq, b, np, hn]
        sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3) # query: [1024,1,32,128], key: [1024,1,2,128]
        rot_dim = rope_cache.shape[-2] * 2 # 32 * 2 = 64
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:] # query: x-> [1024, 1, 32, 64] x_pass-> [1024, 1, 32, 64] key: x->[1024,1,2,64], x_pass->[1024,1,2,64]
        # truncate to support variable sizes
        rope_cache = rope_cache[:sq] # [1024, 1, 32, 2]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2) # query: [1024,1,32,32,2]  key: [1024,1,2,32,2]
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2) # [1024, 1, 1, 32, 2]
    with time_context("stack"):

        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        ) # xshaped[..., 0]: [1024, 1, 32, 32], xshaped[..., 1]: [1024, 1, 32, 32], rope_cache[..., 0]: [1024, 1, 1, 32], rope_cache[..., 1]: [1024, 1, 1, 32]
    with time_context("af_stack"):
        # x_out2: [1024, 1, 32, 32, 2]
        x_out2 = x_out2.flatten(3) # [1024, 1, 32, 64]
    return torch.cat((x_out2, x_pass), dim=-1) # [1024,1,32,128]

# gpt-j

def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb_gptj(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)
from einops import rearrange

#palm
def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_palm(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


#--------------------------

def bench_chatglm():
    rope_cache_=torch.rand(1024,1,32,2).to('hpu')
    x_=torch.rand(1024, 1, 32, 64).to('hpu')
    with time_context("apply_rotary_pos_emb"):
        res = apply_rotary_pos_emb(x_,rope_cache_)
        
        
        
bench_chatglm()
