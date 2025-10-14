import pandas as pd
import numpy as np
from scipy.stats import entropy
from itertools import combinations

# ---------- Config ----------
VDD = 0.8                # set your supply here
THRESH = VDD / 2          # Vdd/2 threshold

# ---------- Load & clean ----------
df = pd.read_csv("ddpuf_mc.csv")

# Extract only bit columns (E0..E127)
bit_cols = [c for c in df.columns if c.startswith("E")]

# Treat the sentinel "800e-3" as invalid before numeric parsing
raw = df[bit_cols].replace("800e-3", np.nan)
raw = raw.apply(pd.to_numeric, errors="coerce")

# Drop any rows that contain NaN in any bit column (i.e., incomplete responses)
clean = raw.dropna(axis=0, how="any")

# ---------- Analog -> Binary ----------
# v > Vdd/2 => 1 ; v < Vdd/2 => 0
# (Values equal to Vdd/2 are rare here because the "800e-3" sentinel was NaN'd out)
bit_data = (clean > THRESH).astype(int)

# ---------- 1) UNIQUENESS ----------
def hamming_distance(a, b):
    # a, b are 1D arrays of bits
    return np.sum(a != b) / len(a)

hamming_distances = []
for i, j in combinations(range(len(bit_data)), 2):
    hamming_distances.append(hamming_distance(bit_data.iloc[i].values, bit_data.iloc[j].values))

uniqueness = float(np.mean(hamming_distances)) if hamming_distances else np.nan
print(f"Devices kept: {len(bit_data)}   Bits per device: {bit_data.shape[1]}")
print(f"Uniqueness: {uniqueness:.4f}")

# ---------- 2) BIT ALIASING (avg bias from 0.5) ----------
bit_means = bit_data.mean(axis=0)              # P(bit==1) per column
bit_aliasing = np.mean(np.abs(bit_means - 0.5) * 2)  # 0 balanced, 1 fully biased
print(f"Bit Aliasing (avg bias): {bit_aliasing:.4f}")

# ---------- 3) SHANNON ENTROPY ----------
def bit_entropy(p):
    return -p*np.log2(p) - (1-p)*np.log2(1-p) if 0 < p < 1 else 0.0

avg_entropy = float(np.mean([bit_entropy(p) for p in bit_means]))
print(f"Average Shannon Entropy: {avg_entropy:.4f} bits")
