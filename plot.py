#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf_step(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    xs = np.sort(x)
    ys = np.arange(1, xs.size + 1) / xs.size
    xs = np.concatenate(([xs[0]], xs))
    ys = np.concatenate(([0.0], ys))
    return xs, ys

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_cdf_safe.py per_mc_transient.csv")
        sys.exit(1)

    df = pd.read_csv(sys.argv[1])

    metric_col = "cnt_mean"   # change if you want cnt_max / cnt_frac_ge3 etc.
    ber_col    = "ber"

    df = df.dropna(subset=[metric_col, ber_col]).copy()
    x_all = df[metric_col].astype(float).to_numpy()
    ber   = df[ber_col].astype(float).to_numpy()

    xmin = float(np.min(x_all))
    xmax = float(np.max(x_all))

    flaky_mask = ber > 0
    x_flaky = x_all[flaky_mask]

    # ---- contiguous safe threshold from the left ----
    # Find smallest x where BER>0. Shade everything to the left of that.
    if np.any(flaky_mask):
        x_first_flaky = float(np.min(x_all[flaky_mask]))
        t_safe = x_first_flaky  # boundary where errors start appearing
    else:
        t_safe = xmax  # no errors at all

    # ECDFs
    xs_all, ys_all = ecdf_step(x_all)

    if x_flaky.size > 0:
        xs_f, ys_f = ecdf_step(x_flaky)
        # extend flaky CDF to full x-range visually
        if xs_f[0] > xmin:
            xs_f = np.concatenate(([xmin], xs_f))
            ys_f = np.concatenate(([0.0], ys_f))
    else:
        xs_f = np.array([xmin, xmax], dtype=float)
        ys_f = np.array([0.0, 0.0], dtype=float)

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)

    # Shade safe region (contiguous from left)
    ax.axvspan(xmin, t_safe, color="0.92", zorder=0, label="BER = 0 (contiguous safe zone)")

    ax.step(xs_all, ys_all, where="post", color="black", linewidth=2.6, label="All MC instances")
    ax.step(xs_f, ys_f, where="post", color="#B22222", linestyle="--",
            linewidth=2.6, label="Flaky instances (BER > 0)")

    # boundary marker
    ax.axvline(t_safe, color="0.35", linestyle=":", linewidth=2.0)
    ax.text(t_safe, 0.03, f"  T_safe = {t_safe:.2f}", rotation=90,
            va="bottom", ha="left", color="0.35")

    ax.set_title("CDF of Transient Oscillation Metric")
    ax.set_xlabel("Mean transient oscillation count")
    ax.set_ylabel("Cumulative fraction of MC instances")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0.0, 1.02)

    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.legend(loc="lower right", frameon=True)

    out = "cdf_transient_metric_safe.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()