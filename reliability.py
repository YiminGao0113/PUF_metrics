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
