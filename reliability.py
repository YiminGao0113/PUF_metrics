# import pandas as pd

# def parse_value(val):
#     try:
#         return float(val.replace('e', 'E'))
#     except:
#         return None

# def compute_reliability_and_flaky_bits(csv_path, num_bits=128, num_runs=100, threshold=0.4):
#     df = pd.read_csv(csv_path)
#     df = df.dropna(subset=['Output', 'Nominal'])

#     bit_values = [[] for _ in range(num_bits)]

#     for _, row in df.iterrows():
#         name = row['Output']
#         if not isinstance(name, str) or not name.startswith("E"):
#             continue
#         try:
#             bit_idx = int(name.split("_")[0][1:])
#             val = parse_value(str(row['Nominal']))
#             if val is not None and bit_idx < num_bits:
#                 bin_val = 1 if val > threshold else 0
#                 bit_values[bit_idx].append(bin_val)
#         except Exception as e:
#             print(f"Error parsing row: {row}\n{e}")

#     # Print how many values we collected per bit
#     print("\nCollected values per bit (first 8 bits):")
#     for i in range(8):
#         print(f"Bit {i}: {len(bit_values[i])} values → {set(bit_values[i])}")

#     stable_bits = 0
#     flaky_bits = []

#     for idx, bvals in enumerate(bit_values):
#         if len(bvals) >= num_runs and len(set(bvals[:num_runs])) == 1:
#             stable_bits += 1
#         else:
#             flaky_bits.append(idx)
#             print(f"\nFlaky bit {idx}:")
#             print(f"  Number of runs: {len(bvals)}")
#             print(f"  Binary values (first {min(num_runs, len(bvals))}): {bvals[:num_runs]}")
#             print(f"  Unique values: {set(bvals[:num_runs])}")

#     reliability = stable_bits / num_bits
#     print(f"\nTotal bits: {num_bits}")
#     print(f"Stable bits: {stable_bits}")
#     print(f"Reliability: {reliability:.4f}")
#     print(f"Flaky bits ({len(flaky_bits)}): {flaky_bits}")

#     return bit_values  # return for further use in bias computation


# def compute_bias(bit_values, num_runs=100):
#     total_ones = 0
#     total_count = 0

#     for bvals in bit_values:
#         vals = bvals[:num_runs]
#         total_ones += sum(vals)
#         total_count += len(vals)

#     if total_count == 0:
#         print("Bias could not be computed (no valid data)")
#         return None

#     bias = abs((total_ones / total_count) - 0.5)
#     direction = '1s' if (total_ones / total_count) > 0.5 else '0s'
#     print(f"\nPUF Bias: {bias:.4f} (toward {direction})")
#     return bias


# # Usage
# bit_vals = compute_reliability_and_flaky_bits('multipleruns.csv', num_bits=128, num_runs=50, threshold=0.4)
# # bit_vals = compute_reliability_and_flaky_bits('nandpuf_50runscsv.csv', num_bits=128, num_runs=50, threshold=0.4)
# compute_bias(bit_vals, num_runs=50)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Helper to parse scientific notation safely ----------
def parse_value(val):
    try:
        return float(val.replace('e', 'E'))
    except:
        return None

# ---------- Read CSV and build binary matrix: shape = (num_runs, num_bits) ----------
def build_matrix(csv_path, num_bits=128, num_runs=50, threshold=0.4):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Output', 'Nominal'])

    bit_runs = [[] for _ in range(num_bits)]

    for _, row in df.iterrows():
        name = row['Output']
        if not isinstance(name, str) or not name.startswith("E"):
            continue
        try:
            bit_idx = int(name.split("_")[0][1:])
            val = parse_value(str(row['Nominal']))
            if val is not None and bit_idx < num_bits:
                bin_val = 1 if val > threshold else 0
                bit_runs[bit_idx].append(bin_val)
        except Exception as e:
            print(f"Error parsing row: {row}\n{e}")

    max_len = max(len(r) for r in bit_runs)
    matrix = np.full((max_len, num_bits), np.nan)
    for bit in range(num_bits):
        vals = bit_runs[bit][:max_len]
        matrix[:len(vals), bit] = vals

    return matrix

# ---------- Plot 0/1 heatmap ----------
def plot_binary_heatmap(matrix):
    plt.figure(figsize=(14, 6))
    sns.heatmap(matrix, cmap='Greys', cbar=True, xticklabels=8, yticklabels=10)
    plt.title('Bit Stability Across Runs (0 = black, 1 = white)')
    plt.xlabel('Bit Index')
    plt.ylabel('Run Index')
    plt.tight_layout()
    plt.show()

# ---------- Plot reliability heatmap ----------
def plot_reliability_heatmap(matrix):
    std_per_bit = np.nanstd(matrix, axis=0)
    reliability_score = 1.0 - std_per_bit  # 1 = stable

    plt.figure(figsize=(14, 2))
    cmap = sns.color_palette("RdYlGn", as_cmap=True)
    sns.heatmap(
        reliability_score[np.newaxis, :],
        cmap=cmap, cbar=True,
        xticklabels=8, yticklabels=False,
        vmin=0.0, vmax=1.0
    )
    plt.title("Reliability Heatmap (Green = Stable, Red = Flaky)")
    plt.xlabel("Bit Index")
    plt.tight_layout()
    plt.show()

    return reliability_score

# ---------- Main Execution ----------
if __name__ == "__main__":
    matrix = build_matrix("multipleruns.csv", num_bits=128, num_runs=50, threshold=0.4)
    plot_binary_heatmap(matrix)
    reliability = plot_reliability_heatmap(matrix)

    # Optional: print summary
    reliable_bits = np.sum(reliability >= 0.95)
    print(f"\nReliable bits (≥ 0.95): {reliable_bits}/128")
    print("Flaky bits:", np.where(reliability < 0.95)[0].tolist())
