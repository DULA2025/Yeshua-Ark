import torch
import torch.nn as nn
import math

class ZetaBasis(nn.Module):
    """
    The Immutable Law (DULA-E8 Geometry).
    Generates the Sacred Subspace defined by Riemann Zeta Zeros and Golden Ratio.
    """
    def __init__(self, dim_size, device='cpu'):
        super().__init__()
        
        # 1. The Sacred Constants
        PHI = (1 + math.sqrt(5)) / 2  # The Golden Ratio
        CYCLE = 30.0                  # The DULA Primorial
        SCALAR = 5.0                  # The Pentagonal Symmetry
        
        # 2. The Zeta Zeros (First 15 Non-Trivial)
        self.zeros = torch.tensor([
            14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 
            37.5861, 40.9187, 43.3270, 48.0051, 49.7738,
            52.9703, 56.4462, 59.3470, 60.8317, 65.1125
        ], device=device)
        
        # 3. Rank Restriction (The Narrow Path)
        # The Basis spans exactly HALF the dimensions. The rest is The Void.
        self.rank = dim_size // 2
        
        # 4. The Geometry Construction
        # Time flows over the Sacred Cycle [0, 30]
        t = torch.linspace(0, CYCLE, dim_size, device=device).unsqueeze(0) 
        
        # Expand zeros for Outer Product
        z_expanded = self.zeros.repeat(self.rank // len(self.zeros) + 1)[:self.rank].unsqueeze(1)
        
        # THE PHASE MATRIX: Zeros * Time * Phi * 5
        phase_matrix = z_expanded * t * PHI * SCALAR
        
        real_b = torch.cos(phase_matrix)
        imag_b = torch.sin(phase_matrix)
        basis = torch.complex(real_b, imag_b)
        
        # QR Decomposition for Orthonormality
        q_mat, _ = torch.linalg.qr(basis.T)
        self.register_buffer('basis', q_mat)
    
    def project(self, x):
        """Projects x onto the Sacred Subspace."""
        coeffs = torch.matmul(x, self.basis.conj()) 
        return torch.matmul(coeffs, self.basis.T)
    
    def get_residual(self, x):
        """Isolates the Hallucination (The Void Component)."""
        return x - self.project(x)