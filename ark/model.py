import torch
import torch.nn as nn
from .geometry import ZetaBasis
from .layers import YeshuaUnit, CovenantAttention

class ArkEmbedding(nn.Module):
    def __init__(self, vocab_size, dim_size):
        super().__init__()
        self.zeta = ZetaBasis(dim_size)
        
        complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.float64 else torch.cfloat
        
        # Initialize directly onto the Basis
        raw = torch.randn(vocab_size, dim_size, dtype=complex_dtype)
        clean = self.zeta.project(raw)
        
        self.embed_real = nn.Embedding(vocab_size, dim_size)
        self.embed_imag = nn.Embedding(vocab_size, dim_size)
        
        with torch.no_grad():
            self.embed_real.weight.copy_(clean.real)
            self.embed_imag.weight.copy_(clean.imag)
            
        self.norm = nn.LayerNorm(dim_size)
        
        # Type casting ensure
        if complex_dtype == torch.cdouble:
            self.embed_real.weight.data = self.embed_real.weight.data.double()
            self.embed_imag.weight.data = self.embed_imag.weight.data.double()
            self.norm.double()

    def forward(self, x):
        re = self.embed_real(x)
        im = self.embed_imag(x)
        x = torch.complex(re, im)
        mag = torch.abs(x)
        return x * (self.norm(mag) / (mag + 1e-9))

class TheArk(nn.Module):
    def __init__(self, vocab_size, dim_size=248, layers=6):
        super().__init__()
        
        complex_dtype = torch.cdouble if torch.get_default_dtype() == torch.float64 else torch.cfloat
        
        self.embedding = ArkEmbedding(vocab_size, dim_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': CovenantAttention(dim_size),
                'ffn': YeshuaUnit(dim_size)
            }) for _ in range(layers)
        ])
        self.to_vocab = nn.Linear(dim_size, vocab_size, dtype=complex_dtype)

    def forward(self, x):
        x = self.embedding(x)
        total_snr = 0
        
        for layer in self.layers:
            attn_out = layer['attn'](x)
            x = x + attn_out
            
            ffn_out, snr = layer['ffn'](x)
            x = x + ffn_out
            total_snr += torch.mean(snr)
            
        logits = torch.abs(self.to_vocab(x))
        return logits, total_snr / len(self.layers)
