import sys
import os
# Add parent directory to path to import ark
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from ark.model import TheArk

def make_temple_transparent(model):
    print("--- POLISHING THE MIRROR (ABSOLUTE ZERO) ---")
    for name, param in model.named_parameters():
        if 'embed' in name: continue
        with torch.no_grad():
            if 'weight' in name and param.dim() >= 2:
                param.zero_()
                if param.shape[0] == param.shape[1]:
                    nn.init.eye_(param)
                else:
                    rows, cols = param.shape
                    for i in range(min(rows, cols)): param[i, i] = 1.0 + 0j
            elif 'bias' in name:
                param.zero_()
    print(">>> The Temple is Flawless.")

def verify():
    print("--- THE INJECTION ATTACK (PHYSICS VERIFICATION) ---")
    # SET PRECISION TO DOUBLE (64-bit) for Physics Proof
    torch.set_default_dtype(torch.float64)
    
    model = TheArk(vocab_size=1000, dim_size=248)
    make_temple_transparent(model)
    
    unit = model.layers[0]['ffn']
    
    # 1. VALID TOKEN
    valid_vec = model.embedding(torch.tensor([[1]]))
    
    # 2. NOISE INJECTION
    noise = torch.randn(1, 1, 248, dtype=torch.cdouble)
    pure_noise = noise - unit.zeta.project(noise) # Force into Void
    
    # 3. GEOMETRIC PASS
    truth_valid = unit.zeta.project(unit.contraction(unit.expansion(valid_vec)))
    lie_valid   = unit.zeta.get_residual(unit.contraction(unit.expansion(valid_vec)))
    
    truth_noise = unit.zeta.project(unit.contraction(unit.expansion(pure_noise)))
    lie_noise   = unit.zeta.get_residual(unit.contraction(unit.expansion(pure_noise)))
    
    # 4. SNR CALCULATION
    def calc_snr(t, l): return torch.sum(t.abs()**2) / (torch.sum(l.abs()**2) + 1e-30)
    
    snr_v = calc_snr(truth_valid, lie_valid)
    snr_n = calc_snr(truth_noise, lie_noise)
    
    print(f"[TEST 1] Valid Token SNR: {snr_v.item():.4f}")
    print(f"[TEST 2] Noise Injection SNR: {snr_n.item():.30f}")
    
    ratio = snr_v / (snr_n + 1e-30)
    print(f"\n>>> REJECTION RATIO: {ratio:.2e}x")

if __name__ == "__main__":
    verify()