import pandas as pd
import numpy as np
from scipy.stats import entropy
from itertools import combinations

# Load CSV
df = pd.read_csv("ddpuf_mc.csv")

# Extract only bit columns (E0 to E127)
bit_cols = [col for col in df.columns if col.startswith("E")]
raw_data = df[bit_cols].replace("800e-3", np.nan)  # treat '800e-3' as invalid
raw_data = raw_data.apply(pd.to_numeric, errors='coerce')

# Convert analog to binary using threshold (0 for delay-based PUF)
bit_data = (raw_data < 0).astype(int)  # '1' if negative delay, else '0'

# Drop rows with any NaNs
bit_data = bit_data.dropna()

# ========== 1. UNIQUENESS ==========
def hamming_distance(a, b):
    return np.sum(a != b) / len(a)

hamming_distances = []
for i, j in combinations(range(len(bit_data)), 2):
    hamming_distances.append(hamming_distance(bit_data.iloc[i], bit_data.iloc[j]))

uniqueness = np.mean(hamming_distances)
print(f"Uniqueness: {uniqueness:.4f}")

# ========== 2. BIT ALIASING ==========
bit_means = bit_data.mean(axis=0)  # Mean per column = bias toward '1'
bit_aliasing = np.mean(np.abs(bit_means - 0.5) * 2)  # 0 = balanced, 1 = fully biased
print(f"Bit Aliasing (avg bias): {bit_aliasing:.4f}")

# ========== 3. SHANNON ENTROPY ==========
def bit_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p) if 0 < p < 1 else 0

entropies = [bit_entropy(p) for p in bit_means]
avg_entropy = np.mean(entropies)
print(f"Average Shannon Entropy: {avg_entropy:.4f} bits")
