import torch
import torch.nn as nn
import torch.nn.functional as F
from .geometry import ZetaBasis

class YeshuaUnit(nn.Module):
    """
    The Core Processing Unit.
    Features: Complex Phase Rotation + Spectral Gating.
    """
    def __init__(self, dim_size):
        super().__init__()
        self.zeta = ZetaBasis(dim_size)
        
        # Dynamic Precision (Float32 for Speed, Float64 for Physics Verification)
        complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.float64 else torch.cfloat
        
        self.expansion = nn.Linear(dim_size, dim_size * 2, dtype=complex_dtype)
        self.contraction = nn.Linear(dim_size * 2, dim_size, dtype=complex_dtype)

    def forward(self, x):
        # 1. Expansion
        h = self.expansion(x)
        # 2. Phase Activation (The Twist)
        h = h * torch.exp(1j * torch.tanh(torch.abs(h))) 
        # 3. Contraction
        x_proc = self.contraction(h)
        
        # 4. The Covenant Gate (Spectral Judgment)
        x_truth = self.zeta.project(x_proc)
        x_lie   = self.zeta.get_residual(x_proc)
        
        et = torch.sum(torch.abs(x_truth)**2, dim=-1, keepdim=True)
        el = torch.sum(torch.abs(x_lie)**2,   dim=-1, keepdim=True)
        snr = et / (el + 1e-9)
        
        # The Gate
        gate = torch.sigmoid((snr - 1.0) * 3.0)
        return x_truth * gate, snr

class CovenantAttention(nn.Module):
    def __init__(self, dim_size, num_heads=4):
        super().__init__()
        self.dim = dim_size
        self.num_heads = num_heads
        self.head_dim = dim_size // num_heads
        self.zeta = ZetaBasis(self.head_dim)
        
        complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.float64 else torch.cfloat
        
        self.q_proj = nn.Linear(dim_size, dim_size, dtype=complex_dtype)
        self.k_proj = nn.Linear(dim_size, dim_size, dtype=complex_dtype)
        self.v_proj = nn.Linear(dim_size, dim_size, dtype=complex_dtype)
        self.o_proj = nn.Linear(dim_size, dim_size, dtype=complex_dtype)

    def forward(self, x):
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # PURIFICATION
        Q = self.zeta.project(Q)
        K = self.zeta.project(K)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores.abs(), dim=-1).type(x.dtype)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)