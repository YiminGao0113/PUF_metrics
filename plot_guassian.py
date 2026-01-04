#!/usr/bin/env python3
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
THRESH_V = 0.4
PUF_SIG = "Out0"
COUNTER_BITS = ["q0", "q1", "q2"]  # LSB -> MSB  (0..7)
USE_COUNTER = "mode"              # "mode" (recommended) or "mean_round"
MAX_COUNTER = 7
# ---------------------------------------

PARAM_RE = re.compile(r"Parameters:\s*mc_iteration\s*=\s*(\d+)", re.IGNORECASE)
OUT_RE   = re.compile(r"^E_(Out0|q0|q1|q2)_iter(\d+)$", re.IGNORECASE)

def parse_csv_blocks(path: str) -> pd.DataFrame:
    rows = []
    current_mc = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = PARAM_RE.match(line)
            if m:
                current_mc = int(m.group(1))
                continue
            if current_mc is None:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            output = parts[2]
            val_s  = parts[3]
            m2 = OUT_RE.match(output)
            if not m2:
                continue
            sig = m2.group(1)
            it  = int(m2.group(2))
            try:
                val = float(val_s)
            except ValueError:
                val = np.nan
            rows.append((current_mc, sig, it, val))
    return pd.DataFrame(rows, columns=["mc", "sig", "iter", "value"])

def build_tensor(df, n_mc, n_iter, sigs):
    tensor = {}
    for sig in sigs:
        mat = np.full((n_mc, n_iter), np.nan, dtype=float)
        sub = df[df["sig"].str.lower() == sig.lower()]
        for _, r in sub.iterrows():
            mc, it = int(r["mc"]), int(r["iter"])
            if 1 <= mc <= n_mc and 0 <= it < n_iter:
                mat[mc-1, it] = float(r["value"])
        tensor[sig] = mat
    return tensor

def to_bits(mat, thresh):
    return np.where(np.isnan(mat), np.nan, (mat > thresh).astype(float))

def majority_bit(bits_2d):
    # bits_2d: (n_mc, n_iter) in {0,1,nan}
    ones  = np.nansum(bits_2d == 1, axis=1)
    zeros = np.nansum(bits_2d == 0, axis=1)
    # ties -> 1 (fine; extremely rare unless exactly balanced)
    return np.where(ones >= zeros, 1, 0).astype(int)

def counter_values(q_bits):
    q0, q1, q2 = q_bits["q0"], q_bits["q1"], q_bits["q2"]
    valid = ~(np.isnan(q0) | np.isnan(q1) | np.isnan(q2))
    cnt = np.full(q0.shape, np.nan, dtype=float)
    cnt[valid] = q0[valid] + 2*q1[valid] + 4*q2[valid]
    return cnt

def per_mc_counter_metric(cnt_2d, kind="mode"):
    n_mc = cnt_2d.shape[0]
    out = np.full(n_mc, np.nan, dtype=float)
    for i in range(n_mc):
        v = cnt_2d[i, :]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        if kind == "mode":
            vals, counts = np.unique(v.astype(int), return_counts=True)
            out[i] = float(vals[np.argmax(counts)])
        elif kind == "mean_round":
            out[i] = float(int(np.rint(np.mean(v))))
        else:
            raise ValueError("kind must be 'mode' or 'mean_round'")
    return out

def bucketize_counter(c):
    """
    Your requested bins:
      <=1 as one bucket labeled 1
      then 2..7
    We map <=1 -> 1, keep others as-is, clamp to [1..MAX_COUNTER].
    """
    if np.isnan(c):
        return np.nan
    c = int(c)
    if c <= 1:
        return 1
    if c > MAX_COUNTER:
        return MAX_COUNTER
    return c

def fit_gaussian_from_counts(x_positions, counts):
    """
    Fit a zero-mean Gaussian shape only for visualization:
      sigma estimated from weighted variance around 0.
    """
    x = np.asarray(x_positions, dtype=float)
    w = np.asarray(counts, dtype=float)
    wsum = np.sum(w)
    if wsum <= 0:
        return 1.0
    mu = 0.0
    var = np.sum(w * (x - mu)**2) / wsum
    sigma = np.sqrt(max(var, 1e-9))
    return sigma

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_signed_counter_gaussian.py <nominal_full.csv> [out.png]")
        sys.exit(1)

    in_csv = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) >= 3 else "signed_counter_gaussian.png"

    df = parse_csv_blocks(in_csv)
    if df.empty:
        raise SystemExit("Parsed 0 rows. Check your signal names: E_Out0_iter#, E_q0_iter#, ...")

    n_mc   = int(df["mc"].max())
    n_iter = int(df["iter"].max()) + 1

    tensor = build_tensor(df, n_mc, n_iter, [PUF_SIG] + COUNTER_BITS)

    # Bits
    out_bits = to_bits(tensor[PUF_SIG], THRESH_V)
    q_bits   = {q: to_bits(tensor[q], THRESH_V) for q in COUNTER_BITS}

    # Majority output bit per MC (sign)
    out_maj = majority_bit(out_bits)  # 1 or 0

    # Counter per read, then per MC metric
    cnt = counter_values(q_bits)
    cnt_metric = per_mc_counter_metric(cnt, kind=USE_COUNTER)

    per_mc = pd.DataFrame({
        "mc": np.arange(1, n_mc+1),
        "out_maj": out_maj,
        "cnt_metric": cnt_metric,
    }).dropna()

    per_mc["bucket"] = per_mc["cnt_metric"].apply(bucketize_counter).astype(int)

    # Count per bucket per output
    buckets = [1] + list(range(2, MAX_COUNTER+1))
    left_counts  = []  # Out=1 (left)
    right_counts = []  # Out=0 (right)

    for b in buckets:
        sub = per_mc[per_mc["bucket"] == b]
        left_counts.append(int(np.sum(sub["out_maj"] == 1)))
        right_counts.append(int(np.sum(sub["out_maj"] == 0)))

    # Build symmetric x-positions (negative for out=1, positive for out=0)
    x_left  = -np.array(buckets, dtype=float)
    x_right =  np.array(buckets, dtype=float)

    # Gaussian overlay (use total counts mirrored)
    total_counts = np.array(left_counts) + np.array(right_counts)
    x_abs = np.array(buckets, dtype=float)
    sigma = fit_gaussian_from_counts(x_abs, total_counts)

    x_curve = np.linspace(-MAX_COUNTER-0.5, MAX_COUNTER+0.5, 400)
    pdf = np.exp(-0.5*(x_curve/sigma)**2) / (sigma*np.sqrt(2*np.pi))

    # Scale Gaussian to match histogram height roughly:
    # scale by total population * binwidth (≈1)
    scale = np.sum(total_counts) * 1.0
    y_curve = pdf * scale

    # ---- Plot ----
    plt.figure(figsize=(7.2, 3.8), dpi=250)
    ax = plt.gca()

    barw = 0.85
    ax.bar(x_left,  left_counts,  width=barw, align="center", label="Out=1 (left)")
    ax.bar(x_right, right_counts, width=barw, align="center", label="Out=0 (right)")

    ax.plot(x_curve, y_curve, linewidth=1.8, label="Gaussian-shaped guide")

    # Axis cosmetics
    xticks = list(range(-MAX_COUNTER, 0)) + [0] + list(range(1, MAX_COUNTER+1))
    ax.set_xticks(xticks)

    # Label bucket 1 as "≤1"
    xticklabels = []
    for t in xticks:
        if t == 0:
            xticklabels.append("0")
        else:
            a = abs(t)
            if a == 1:
                xticklabels.append("≤1")
            else:
                xticklabels.append(str(a))
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel("Proxy |Δ| bucket from transient counter (mirrored by majority output bit)")
    ax.set_ylabel("Number of MC instances")
    ax.set_title("Mirrored distribution: output sign × transient-counter magnitude (proxy)")

    ax.axvline(0, linewidth=1.0)
    ax.set_xlim(-(MAX_COUNTER+1), (MAX_COUNTER+1))

    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Wrote {out_png}")

if __name__ == "__main__":
    main()import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load per-MC data
df = pd.read_csv("per_mc_transient.csv")

# Drop invalid
df = df.dropna(subset=["cnt_mean", "stability"])

# Sign from majority output
# Out=1 → negative delay, Out=0 → positive
sign = np.where(df["stability"] >= 0.5, -1, 1)

# Pseudo-delay (monotonic proxy)
pseudo_delay = sign * np.log1p(df["cnt_mean"])

# ---- Plot ----
plt.figure(figsize=(8, 4.5))

# Histogram (density)
bins = np.linspace(pseudo_delay.min(), pseudo_delay.max(), 30)
plt.hist(
    pseudo_delay,
    bins=bins,
    density=True,
    color="#c44e52",
    alpha=0.35,
    edgecolor="none",
    label="Reconstructed Δdelay distribution"
)

# Scatter (rug-style)
y_jitter = np.random.uniform(0, 0.02, size=len(pseudo_delay))
plt.scatter(
    pseudo_delay,
    y_jitter,
    s=10,
    color="#c44e52",
    alpha=0.6
)

# Gaussian fit (guide only)
mu, sigma = np.mean(pseudo_delay), np.std(pseudo_delay)
x = np.linspace(pseudo_delay.min(), pseudo_delay.max(), 400)
plt.plot(
    x,
    norm.pdf(x, mu, sigma),
    color="black",
    lw=1.8,
    label="Gaussian fit (guide)"
)

# Counter bucket annotations
for k in [0, 1, 2, 3]:
    width = np.log1p(k + 1)
    plt.axvspan(-width, width, color="gray", alpha=0.05)

# Labels
plt.xlabel("Signed delay imbalance (digital proxy)")
plt.ylabel("Probability density")
plt.title("Reconstructed Delay Distribution via Oscillation-Driven Screening")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()