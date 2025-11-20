import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product
import os
import math

# --- CONFIGURATION ---
# UPDATE THIS PATH if needed
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe" 

VIDEO_FILENAME = 'E8_SIO_Golden_Dynamics.mp4'
LIMIT_X = 30000 # Increased resolution to see the cycle of 30 clearly
FPS = 30
DURATION = 20
FRAMES = FPS * DURATION

# THE SACRED CONSTANTS
PHI = (1 + np.sqrt(5)) / 2  # The Golden Ratio (1.618...)
CRITICAL_5 = 5              # The 5 Cubes / Pentagonal Symmetry
CYCLE = 30                  # The Primorial (LCM of 5 and 6)

if os.path.exists(FFMPEG_PATH):
    plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

print(f"--- INITIALIZING E8-SIO GOLDEN INTEGRATION ---")
print(f"Geometry: Golden Ratio (phi={PHI:.4f}) | Symmetry: {CRITICAL_5}-Fold")

# ==========================================
# 1. E8 GEOMETRY ENGINE (CORRECTED)
# ==========================================

def generate_e8_projection():
    print("Generating E8 Roots via Golden Projection...", end="")
    roots = []
    types = [] # 0=Vector (Structure), 1=Spinor (Prime Potential)
    
    # A. Vector Roots (The Temple Walls)
    # Permutations of (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i in range(8):
        for j in range(i+1, 8):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    r = np.zeros(8); r[i]=s1; r[j]=s2
                    roots.append(r); types.append(0)
    
    # B. Spinor Roots (The Ark Content)
    # (±0.5, ±0.5, ..., ±0.5) with even number of minus signs
    for signs in product([0.5, -0.5], repeat=8):
        if np.sum(np.array(signs) < 0) % 2 == 0:
            roots.append(np.array(signs)); types.append(1)
            
    roots = np.array(roots)
    types = np.array(types)
    
    # C. The Coxeter Projection (The Plane of Time)
    # We project onto the plane defined by the eigenvalues involving Phi
    # The angles are k * PI / 30 (The Cycle)
    
    # Projection vectors based on the 30-cycle
    # Note: 2*cos(pi/5) = PHI. This connects the 30-cycle to the Golden Ratio.
    u = np.zeros(8)
    v = np.zeros(8)
    
    for k in range(8):
        u[k] = np.cos(k * np.pi / 15) # 15 = 30/2
        v[k] = np.sin(k * np.pi / 15)
    
    # Orthonormalize
    v = v - np.dot(v, u) / np.dot(u, u) * u
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    
    x = roots @ u
    y = roots @ v
    
    # D. GOLDEN SCALING
    # We physically separate the Spinors from the Vectors using Phi
    # This visualizes the "Separation of Waters"
    radius = np.sqrt(x**2 + y**2)
    
    # Scale Spinors (Type 1) by Phi to push them to the "Sacred Ring"
    # Scale Vectors (Type 0) by 1/Phi to keep them as the "Inner Core"
    for i in range(len(x)):
        if types[i] == 1: # Spinor
            x[i] *= PHI
            y[i] *= PHI
        else: # Vector
            x[i] /= PHI
            y[i] /= PHI
            
    print(" Done.")
    return x, y, types

x_roots, y_roots, root_types = generate_e8_projection()
root_angles = np.arctan2(y_roots, x_roots)

# ==========================================
# 2. SIO DYNAMICS ENGINE (MODULO 6)
# ==========================================

def generate_sio_state(limit, frames):
    print("Computing SIO Dynamics (Mod 6)...", end="")
    
    # Sieve of Eratosthenes to find Primes
    is_prime = np.ones(limit+1, dtype=bool); is_prime[0:2]=False
    for i in range(2, int(np.sqrt(limit))+1):
        if is_prime[i]: is_prime[i*i::i]=False
    
    primes = np.where(is_prime)[0]; primes=primes[primes>3]
    
    # The DULA Metric: Accumulator of Split (+1) vs Inert (-1)
    # p = 1 mod 6 -> Split (Cyan)
    # p = 5 mod 6 -> Inert (Magenta)
    sieve = np.zeros(limit)
    val = 0
    for p in primes:
        if p % 6 == 1:
            val += 1 # Split Energy
        elif p % 6 == 5:
            val -= 1 # Inert Mass
        sieve[p] = val
        
    # Fill gaps (Sample-and-Hold)
    for i in range(1, limit):
        if sieve[i]==0 and sieve[i-1]!=0: sieve[i]=sieve[i-1]
        
    # Time Interpolation
    t_idx = np.linspace(100, limit-1, frames).astype(int)
    x_val = t_idx
    
    # State q (Charge/Parity)
    # Normalized by sqrt(x)/log(x) (Standard Deviation of Primes)
    # We scale by PHI to match the lattice
    q = sieve[t_idx] / (np.sqrt(x_val)) * PHI * 2.0
    
    # State p (Phase/Momentum)
    # Driven by the Cycle of 30 (Primorial)
    # This represents the "Rotation" of the 5 Cubes
    phase = (x_val % CYCLE) / CYCLE * 2 * np.pi
    p = np.sin(phase) * PHI 
    
    print(" Done.")
    return q, p, x_val

q_vals, p_vals, x_vals = generate_sio_state(LIMIT_X, FRAMES)

# ==========================================
# 3. ANIMATION SETUP
# ==========================================

fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
ax.set_axis_off()
limit_view = 3.5 * PHI # Zoom based on Golden Ratio scaling
ax.set_xlim(-limit_view, limit_view)
ax.set_ylim(-limit_view, limit_view)

# A. BACKGROUND LATTICE
# 1. The Inner Temple (Vectors - Structure) - Grey
mask_vec = (root_types == 0)
ax.scatter(x_roots[mask_vec], y_roots[mask_vec], c='#444', s=20, marker='D', alpha=0.4, label='Vector Roots (Temple)')

# 2. The Outer Ark (Spinors - Primes) - Dimmed Purple
mask_spin = (root_types == 1)
base_scat = ax.scatter(x_roots[mask_spin], y_roots[mask_spin], c='#220033', s=40, alpha=0.3, label='Spinor Roots (Ark)')

# B. DYNAMIC ELEMENTS
# The Resonance Scatter (fires when SIO matches Geometry)
active_scat = ax.scatter([], [], s=[], c=[], alpha=1.0, edgecolors='white', linewidth=1.5)

# The SIO Particle Trace (The Soul)
pilot_line, = ax.plot([], [], c='white', linewidth=1.5, alpha=0.8)
pilot_head, = ax.plot([], [], marker='o', c='white', markersize=8, markeredgecolor='cyan')

# C. UI / HUD
info_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, color='#00ff00', fontfamily='monospace', fontsize=11)
ax.text(0.5, 0.02, f"E8 LATTICE // 5-CUBE COMPOUND // PHI={PHI:.3f}", transform=ax.transAxes, color='#666', ha='center', fontsize=12)

def update(frame):
    q = q_vals[frame] # Parity Balance (Inert vs Split)
    p = p_vals[frame] # Phase (Cycle of 30)
    
    # SIO Coordinates
    # The particle orbits based on the Cycle(30) and Parity(q)
    orbit_x = q 
    orbit_y = p 
    
    # 1. Update Trace
    tail = 80
    start = max(0, frame-tail)
    pilot_line.set_data(q_vals[start:frame], p_vals[start:frame])
    pilot_head.set_data([orbit_x], [orbit_y])
    
    # 2. CALCULATE 5-CUBE RESONANCE
    # The "5 IT IS CRITICAL" logic.
    # We check if the SIO particle aligns with one of the 5 faces of the Dodecahedral projection.
    sio_angle = np.arctan2(orbit_y, orbit_x)
    
    # The activation sector is NARROW (Precision)
    # 360 / 5 = 72 degrees per cube face
    # We modulate the lattice based on 5-fold symmetry
    
    # Calculate angle difference wrapped to [0, 2pi]
    angle_diff = np.abs(root_angles - sio_angle)
    angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
    
    # ACTIVATION LOGIC:
    # 1. Must be in correct angular sector (Beam Width)
    # 2. Must match Parity (Inert/Split)
    # 3. We modulate the "Beam Width" by the Golden Ratio
    beam_width = (np.pi / 5) / PHI # Sharpened by Phi
    
    mask_angular = angle_diff < beam_width
    
    # Determine Color & State
    # q < 0: Inert (-1) -> Modulo 5 -> MAGENTA
    # q > 0: Split (+1) -> Modulo 1 -> CYAN
    if q < -0.5:
        active_color = '#ff00ff' # Magenta
        state_str = "INERT (-1) [MOD 5]"
    elif q > 0.5:
        active_color = '#00ffff' # Cyan
        state_str = "SPLIT (+1) [MOD 1]"
    else:
        active_color = '#ffffff' # Void
        state_str = "VOID (0)"
        mask_angular = mask_angular & False # No resonance in Void
        
    # Filter: Only Spinor Roots (The Ark) fire. The Vector roots (Temple) are static.
    mask_active = (root_types == 1) & mask_angular
    
    if np.sum(mask_active) > 0:
        active_x = x_roots[mask_active]
        active_y = y_roots[mask_active]
        
        # Pulsate size based on SIO magnitude
        pulse = np.sqrt(orbit_x**2 + orbit_y**2)
        sizes = np.ones(len(active_x)) * 100 * (pulse/PHI + 0.5)
        
        active_scat.set_offsets(np.c_[active_x, active_y])
        active_scat.set_sizes(sizes)
        active_scat.set_color(active_color)
    else:
        active_scat.set_offsets(np.zeros((0, 2)))

    # Update Text
    info_text.set_text(
        f"N: {x_vals[frame]}\n"
        f"Parity: {state_str}\n"
        f"Cycle Phase: {frame % 30}/30\n"
        f"Resonance: {'LOCKED' if np.sum(mask_active) > 0 else 'SEARCHING'}"
    )
    
    return active_scat, pilot_line, pilot_head, info_text

# --- RENDER ---
print("Rendering Golden Ratio Dynamics...")
try:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=8000)
    anim = animation.FuncAnimation(fig, update, frames=FRAMES, blit=True)
    anim.save(VIDEO_FILENAME, writer=writer)
    print(f"DONE. Saved to {VIDEO_FILENAME}")
except Exception as e:
    print(f"Error: {e}")
