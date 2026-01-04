#!/usr/bin/env python3
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
THRESH_V = 0.4
PUF_SIG = "Out0"
COUNTER_BITS = ["q0", "q1", "q2"]   # LSB -> MSB
OUT_PREFIX = "E_"                  # CSV signal names look like E_Out0_iter0
BIN_EDGES = [-np.inf, 1, 2, 3, np.inf]
BIN_LABELS = ["≤1", "1–2", "2–3", ">3"]
# ---------------------------------------

PARAM_RE = re.compile(r"Parameters:\s*mc_iteration\s*=\s*(\d+)", re.IGNORECASE)
OUT_RE   = re.compile(r"^E_(Out0|q0|q1|q2)_iter(\d+)$", re.IGNORECASE)


# -------- CSV PARSING --------
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


def build_tensor(df: pd.DataFrame, n_mc: int, n_iter: int, sigs):
    tensor = {}
    for sig in sigs:
        mat = np.full((n_mc, n_iter), np.nan, dtype=float)
        sub = df[df["sig"].str.lower() == sig.lower()]
        # iterrows is OK for your scale; keep simple/robust
        for _, r in sub.iterrows():
            mc = int(r["mc"])
            it = int(r["iter"])
            if 1 <= mc <= n_mc and 0 <= it < n_iter:
                mat[mc - 1, it] = float(r["value"])
        tensor[sig] = mat
    return tensor


# -------- BIT & RELIABILITY --------
def to_bits(mat: np.ndarray, thresh: float) -> np.ndarray:
    # returns float array {0,1,nan}
    return np.where(np.isnan(mat), np.nan, (mat > thresh).astype(float))


def puf_majority_and_stability(bits: np.ndarray):
    """
    bits: (n_mc, n_iter) in {0,1,nan}
    Returns:
      maj: (n_mc,) majority bit (ties -> 1)
      stability: (n_mc,) fraction of reads matching majority
      strict: (n_mc,) 1 if all reads identical (ignoring NaN), else 0
    """
    ones  = np.nansum(bits == 1, axis=1)
    zeros = np.nansum(bits == 0, axis=1)

    maj = np.where(ones >= zeros, 1.0, 0.0)  # ties -> 1
    matches = np.where(np.isnan(bits), np.nan, (bits == maj[:, None]).astype(float))
    stability = np.nanmean(matches, axis=1)

    strict = np.full(bits.shape[0], np.nan, dtype=float)
    for i in range(bits.shape[0]):
        b = bits[i, :]
        b = b[~np.isnan(b)]
        if b.size == 0:
            strict[i] = np.nan
        else:
            strict[i] = 1.0 if np.all(b == b[0]) else 0.0

    return maj, stability, strict


def ber_per_mc(bits: np.ndarray, maj: np.ndarray) -> np.ndarray:
    """
    Intra-chip BER per MC:
      BER_i = mean_j [ bits[i,j] != maj[i] ] over valid reads
    Returns fraction in [0,1]
    """
    ber = np.full(bits.shape[0], np.nan, dtype=float)
    for i in range(bits.shape[0]):
        b = bits[i, :]
        b = b[~np.isnan(b)]
        if b.size == 0:
            ber[i] = np.nan
        else:
            ber[i] = np.mean(b != maj[i])
    return ber


# -------- COUNTER --------
def counter_values(q_bits_dict):
    """
    q_bits_dict: q0..q2 each (n_mc,n_iter) {0,1,nan}
    returns cnt (n_mc,n_iter) float with nan if any bit nan in that read
    """
    q0, q1, q2 = (q_bits_dict["q0"], q_bits_dict["q1"], q_bits_dict["q2"])
    valid = ~(np.isnan(q0) | np.isnan(q1) | np.isnan(q2))
    cnt = np.full(q0.shape, np.nan, dtype=float)
    cnt[valid] = (q0[valid] + 2*q1[valid] + 4*q2[valid])
    return cnt


def per_mc_counter_stats(cnt: np.ndarray):
    """
    cnt: (n_mc,n_iter) float with NaN
    Returns per-MC: mode, mean, median, max, frac_ge3
    """
    n_mc = cnt.shape[0]
    mode = np.full(n_mc, np.nan)
    mean = np.full(n_mc, np.nan)
    med  = np.full(n_mc, np.nan)
    mx   = np.full(n_mc, np.nan)
    frac_ge3 = np.full(n_mc, np.nan)

    for i in range(n_mc):
        v = cnt[i, :]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        vi = v.astype(int)
        vals, counts = np.unique(vi, return_counts=True)
        mode[i] = float(vals[np.argmax(counts)])
        mean[i] = float(np.mean(v))
        med[i]  = float(np.median(v))
        mx[i]   = float(np.max(v))
        frac_ge3[i] = float(np.mean(v >= 3))
    return mode, mean, med, mx, frac_ge3


def plot_ber_vs_activity(per_mc: pd.DataFrame, out_png: str):
    """
    Paper-ready plot:
      - Bars (left): # MC instances per bin
      - Line (right): mean BER (%) per bin
      - Safe-zone shading: contiguous bins from left with BER==0
      - No markers/labels for BER==0 bins
    """
    df = per_mc.copy()
    df = df.dropna(subset=["cnt_mean", "ber"])

    # Bin by mean oscillation count
    df["bin"] = pd.cut(
        df["cnt_mean"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=True,
        include_lowest=True
    )

    g = df.groupby("bin")["ber"]
    count_by_bin = g.count().reindex(BIN_LABELS).fillna(0).astype(int)
    ber_mean = (g.mean().reindex(BIN_LABELS).fillna(0.0) * 100.0)  # %

    x = np.arange(len(BIN_LABELS))

    plt.figure(figsize=(7.4, 3.8), dpi=300)
    ax1 = plt.gca()

    # ---- Safe-zone shading: contiguous BER==0 bins from the left ----
    safe_last = -1
    for i, v in enumerate(ber_mean.values):
        if np.isclose(v, 0.0):
            safe_last = i
        else:
            break
    if safe_last >= 0:
        ax1.axvspan(-0.5, safe_last + 0.5, color="0.95", zorder=0)

    # ---- Bars ----
    bar_colors = ["0.82", "0.72", "0.62", "0.52"]
    bars = ax1.bar(
        x, count_by_bin.values,
        width=0.72,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.9,
        zorder=2
    )
    # ---- FIX: add headroom so bar labels don’t hit top border ----
    ymax = np.nanmax(count_by_bin.values)
    ax1.set_ylim(0, ymax * 1.12)

    ax1.set_xlabel("Mean transient oscillation count (binned)", fontsize=13)
    ax1.set_ylabel("Number of MC instances", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels(BIN_LABELS, fontsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.30, zorder=1)

    # Count labels (slightly lower to avoid title overlap)
    ymax = max(1, int(count_by_bin.max()))
    for b, val in zip(bars, count_by_bin.values):
        if np.isnan(val) or val == 0:
            continue

        h = b.get_height()
        xloc = b.get_x() + b.get_width() / 2

        # If bar is very tall, place label inside
        
        ax1.text(
            xloc, h + 0.02 * ymax,
            f"{int(val)}",
            ha="center", va="bottom",
            fontsize=11)

    # ---- BER line on secondary axis ----
    ax2 = ax1.twinx()
    ax2.set_ylabel("BER (%)", fontsize=13)
    ax2.tick_params(axis="y", labelsize=12)

    # Plot line through all bins, BUT only show markers where BER>0
    y = ber_mean.values
    ax2.plot(x, y, color="black", linewidth=2.2, zorder=3)

    idx_pos = np.where(y > 0)[0]
    ax2.plot(
        x[idx_pos], y[idx_pos],
        linestyle="none",
        marker="o",
        markersize=5.5,
        color="black",
        zorder=4
    )

    # BER labels only for BER>0
    ber_ymax = max(1.0, float(np.max(y)))
    for xi, yi in zip(x[idx_pos], y[idx_pos]):
        ax2.text(
            xi,
            yi + 0.06 * ber_ymax,
            f"{yi:.2f}",
            ha="center", va="bottom",
            fontsize=10
        )

    ax2.set_ylim(0, max(1.0, ber_ymax * 1.25))

    plt.title("BER vs. Transient Oscillation Activity", fontsize=14, pad=6)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"Wrote figure: {out_png}")

# -------- MAIN --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_transient_counter.py <csv> [--plot]")
        sys.exit(1)

    path = sys.argv[1]
    do_plot = ("--plot" in sys.argv[2:])

    df = parse_csv_blocks(path)
    if df.empty:
        raise SystemExit("No valid data parsed. Check output names: E_Out0_iter#, E_q*_iter#")

    n_mc = int(df["mc"].max())
    n_iter = int(df["iter"].max()) + 1

    sigs = [PUF_SIG] + COUNTER_BITS
    tensor = build_tensor(df, n_mc, n_iter, sigs)

    # bits
    puf_bits = to_bits(tensor[PUF_SIG], THRESH_V)
    q_bits = {q: to_bits(tensor[q], THRESH_V) for q in COUNTER_BITS}

    # PUF majority/stability/strict + BER
    maj, stability, strict = puf_majority_and_stability(puf_bits)
    ber = ber_per_mc(puf_bits, maj)  # fraction

    # Counter stats
    cnt = counter_values(q_bits)
    cnt_mode, cnt_mean, cnt_med, cnt_max, frac_ge3 = per_mc_counter_stats(cnt)

    per_mc = pd.DataFrame({
        "mc": np.arange(1, n_mc + 1),
        "maj": maj,
        "stability": stability,
        "strict": strict,
        "ber": ber,                  # fraction
        "cnt_mode": cnt_mode,
        "cnt_mean": cnt_mean,
        "cnt_median": cnt_med,
        "cnt_max": cnt_max,
        "cnt_frac_ge3": frac_ge3,
    })

    # Summary
    print("=== Summary ===")
    print(f"MC instances:        {n_mc}")
    print(f"Iterations (reads):  {n_iter}")
    print(f"Threshold (V):       {THRESH_V}")
    print()
    print(f"Overall stability mean: {np.nanmean(stability):.6f}")
    print(f"Overall strict mean:    {np.nanmean(strict):.6f}")
    print(f"Overall BER mean:       {np.nanmean(ber)*100:.6f}%")

    # Save per-mc
    out_csv = "per_mc_transient.csv"
    per_mc.to_csv(out_csv, index=False)
    print(f"\nWrote per-MC table: {out_csv}")

    # Optional plot
    if do_plot:
        out_png = "ber_vs_osc_activity.png"
        plot_ber_vs_activity(per_mc, out_png)


if __name__ == "__main__":
    main()