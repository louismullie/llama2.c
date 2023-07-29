import torch
import time
from typing import Tuple
import os

# Define the original functions (before changes)
def precompute_freqs_cis_orig(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast_orig(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb_orig(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast_orig(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# New functions
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], -1)

def reshape_for_broadcast(freqs_cos: torch.Tensor, freqs_sin: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    shape = [1] * ndim
    shape[1] = x.shape[1]
    shape[-1] = x.shape[-1] // 2
    return freqs_cos.view(*shape), freqs_sin.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:

    freqs_cos, freqs_sin = freqs[..., 0], freqs[..., 1]
    freqs_cos, freqs_sin = reshape_for_broadcast(freqs_cos, freqs_sin, xq)

    xq = xq.view(*xq.shape[:-1], -1, 2)
    xk = xk.view(*xk.shape[:-1], -1, 2)

    xq_r, xq_i = xq[..., 0], xq[..., 1]
    xk_r, xk_i = xk[..., 0], xk[..., 1]

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin 
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos

    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin 
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = torch.stack((xq_out_r, xq_out_i), dim=-1).flatten(-2)
    xk_out = torch.stack((xk_out_r, xk_out_i), dim=-1).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Sample data
batch_size = 128
dim = 288
seq_len = 256
kv_heads = 4
theta = 10000.0

# Original operations on CPU
xq = torch.randn(batch_size, seq_len, kv_heads, dim)
xk = torch.randn(batch_size, seq_len, kv_heads, dim)
freqs_cis_orig = precompute_freqs_cis_orig(dim, seq_len, theta)

start_time = time.time()
for i in range(0, 10): 
    xq_out_orig, xk_out_orig = apply_rotary_emb_orig(xq, xk, freqs_cis_orig)
end_time = time.time()
print("Original time (cpu):", end_time - start_time)

# New operations on MPS
device = 'mps'
xq, xk = xq.to(device), xk.to(device)
freqs_cis = precompute_freqs_cis(dim, seq_len, theta).to(device)

start_time = time.time()
for i in range(0, 10): 
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis)
end_time = time.time()
print(f"New time ({device}):", end_time - start_time)

# Copy to CPU for comparison
freqs_cis = freqs_cis.to('cpu')
xq_out, xk_out = xq_out.to('cpu'), xk_out.to('cpu')

print("All close in freqs_cos:", torch.allclose(freqs_cis_orig.real, freqs_cis[..., 0]))
print("All close in freqs_sin:", torch.allclose(freqs_cis_orig.imag, freqs_cis[..., 1]))

print("All close in xq_out_complex:", torch.allclose(xq_out_orig, xq_out, atol=1e-6))
print("All close in xk_out_complex:", torch.allclose(xk_out_orig, xk_out, atol=1e-6))

print("Max difference in xq_out:", torch.max(torch.abs(xq_out - xq_out_orig)))
print("Max difference in xk_out:", torch.max(torch.abs(xk_out - xk_out_orig)))

# ./run ./out/model.bin 1.0 256 "What heavy metal is the most abundant in the Earth's crust?"