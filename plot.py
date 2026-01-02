#!/usr/bin/env python3
"""
Simplified, paper-style CDF figure (Option A):
- Only ONE ECDF curve (all MC instances)
- Shade the contiguous BER=0 region from left
- Mark T_safe boundary (first appearance of BER>0)

Usage:
  python3 plot_cdf_threshold.py per_mc_transient.csv
  python3 plot_cdf_threshold.py per_mc_transient.csv --metric cnt_mean --ber ber
  python3 plot_cdf_threshold.py per_mc_transient.csv --force_t 2.0
"""

import argparse
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
    # start at y=0 with the first x
    xs = np.concatenate(([xs[0]], xs))
    ys = np.concatenate(([0.0], ys))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="per_mc_transient.csv")
    ap.add_argument("--metric", default="cnt_mean", help="metric column (default: cnt_mean)")
    ap.add_argument("--ber", default="ber", help="BER column (fraction, default: ber)")
    ap.add_argument("--force_t", type=float, default=None,
                    help="Force T_safe to this value instead of auto (e.g., 2.0)")
    ap.add_argument("--out", default="cdf_transient_metric_threshold.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.metric not in df.columns or args.ber not in df.columns:
        raise SystemExit(f"Missing columns. Need metric='{args.metric}' and ber='{args.ber}' in CSV.")

    df = df.dropna(subset=[args.metric, args.ber]).copy()
    x_all = df[args.metric].astype(float).to_numpy()
    ber   = df[args.ber].astype(float).to_numpy()

    if x_all.size == 0:
        raise SystemExit("No valid rows after dropna.")

    xmin = float(np.min(x_all))
    xmax = float(np.max(x_all))

    # --- Determine contiguous BER=0 safe boundary from the left ---
    # T_safe := first x where BER>0 appears (conservative boundary).
    flaky_mask = ber > 0
    if args.force_t is not None:
        t_safe = float(args.force_t)
    else:
        if np.any(flaky_mask):
            t_safe = float(np.min(x_all[flaky_mask]))
        else:
            t_safe = xmax  # no errors at all

    # ECDF for all MC instances
    xs_all, ys_all = ecdf_step(x_all)

    # ---------- Styling: clean / TCAS-friendly ----------
    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(9.6, 5.6), dpi=200)

    # Shade contiguous BER=0 region
    ax.axvspan(xmin, t_safe, color="0.92", zorder=0, label="BER = 0 (contiguous safe zone)")

    # Single ECDF curve
    ax.step(xs_all, ys_all, where="post", color="black", linewidth=2.8, label="All MC instances")

    # Boundary line + annotation (minimal)
    ax.axvline(t_safe, color="0.35", linestyle="--", linewidth=2.0)
    ax.text(t_safe, 1.01, f"T_safe ≈ {t_safe:.2f}",
            ha="center", va="bottom", color="0.35", fontsize=14)

    ax.set_title("CDF of Transient Oscillation Metric")
    ax.set_xlabel("Mean transient oscillation count")
    ax.set_ylabel("Cumulative fraction of MC instances")
    # ax.set_xlim(xmin, xmax)
    ax.set_xlim(0, 3.5)
    ax.set_ylim(0.0, 1.02)

    # Light y-grid only (clean)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.grid(False, axis="x")

    # Put legend bottom-right; small and clean
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight")
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()