#!/usr/bin/env python3
import re
import sys
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
THRESH_V = 0.4
PUF_SIG = "Out0"
COUNTER_BITS = ["q0", "q1", "q2"]  # LSB -> MSB
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


def build_tensor(df, n_mc, n_iter, sigs):
    tensor = {}
    for sig in sigs:
        mat = np.full((n_mc, n_iter), np.nan)
        sub = df[df["sig"].str.lower() == sig.lower()]
        for _, r in sub.iterrows():
            mc, it = int(r["mc"]), int(r["iter"])
            if 1 <= mc <= n_mc and 0 <= it < n_iter:
                mat[mc-1, it] = r["value"]
        tensor[sig] = mat
    return tensor


# -------- BIT & RELIABILITY --------
def to_bits(mat, thresh):
    return np.where(np.isnan(mat), np.nan, (mat > thresh).astype(float))


def puf_reliability(bits):
    ones  = np.nansum(bits == 1, axis=1)
    zeros = np.nansum(bits == 0, axis=1)
    valid = ones + zeros

    maj = np.where(ones >= zeros, 1.0, 0.0)
    matches = np.where(np.isnan(bits), np.nan, (bits == maj[:, None]).astype(float))
    stability = np.nanmean(matches, axis=1)

    strict = np.full(bits.shape[0], np.nan)
    for i in range(bits.shape[0]):
        b = bits[i, :]
        b = b[~np.isnan(b)]
        strict[i] = 1.0 if b.size > 0 and np.all(b == b[0]) else 0.0

    return maj, stability, strict


# -------- COUNTER --------
def counter_values(q_bits):
    q0, q1, q2 = q_bits["q0"], q_bits["q1"], q_bits["q2"]
    valid = ~(np.isnan(q0) | np.isnan(q1) | np.isnan(q2))
    cnt = np.full(q0.shape, np.nan)
    cnt[valid] = q0[valid] + 2*q1[valid] + 4*q2[valid]
    return cnt


# -------- MAIN --------
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_transient_counter.py <csv>")
        sys.exit(1)

    df = parse_csv_blocks(sys.argv[1])
    if df.empty:
        raise SystemExit("No valid data parsed.")

    n_mc   = int(df["mc"].max())
    n_iter = int(df["iter"].max()) + 1

    tensor = build_tensor(df, n_mc, n_iter, [PUF_SIG] + COUNTER_BITS)

    puf_bits = to_bits(tensor[PUF_SIG], THRESH_V)
    q_bits   = {q: to_bits(tensor[q], THRESH_V) for q in COUNTER_BITS}

    maj, stability, strict = puf_reliability(puf_bits)
    flaky = ((stability < 1.0) & ~np.isnan(stability)).astype(int)

    cnt = counter_values(q_bits)

    # ---- per-MC stats ----
    cnt_mode = np.full(n_mc, np.nan)
    cnt_mean = np.full(n_mc, np.nan)
    cnt_max  = np.full(n_mc, np.nan)

    for i in range(n_mc):
        v = cnt[i, :]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        vals, counts = np.unique(v.astype(int), return_counts=True)
        cnt_mode[i] = vals[np.argmax(counts)]
        cnt_mean[i] = np.mean(v)
        cnt_max[i]  = np.max(v)

    per_mc = pd.DataFrame({
        "mc": np.arange(1, n_mc+1),
        "stability": stability,
        "strict": strict,
        "flaky": flaky,
        "cnt_mode": cnt_mode,
        "cnt_mean": cnt_mean,
        "cnt_max": cnt_max
    })

    print("\n=== SUMMARY ===")
    print(f"MC instances: {n_mc}")
    print(f"Reads/MC:     {n_iter}")
    print(f"Mean stability: {np.nanmean(stability):.4f}")
    print(f"Strict stable:  {np.nanmean(strict):.4f}")
    print(f"Flaky MCs:      {np.sum(flaky)}")

    # ---- MODE TABLE ----
    print("\n=== By COUNTER MODE ===")
    grp = per_mc.dropna().groupby("cnt_mode")["flaky"].agg(total="count", flaky="sum")
    grp["stable"] = grp["total"] - grp["flaky"]
    grp["flaky_frac"] = grp["flaky"] / grp["total"]
    print(grp)

    # ---- MEAN TABLE ----
    print("\n=== By COUNTER MEAN (binned) ===")
    per_mc["mean_bin"] = pd.cut(
        per_mc["cnt_mean"],
        bins=[0, 1.5, 2.5, np.inf],
        labels=["<1.5", "1.5–2.5", "≥2.5"]
    )
    grp = per_mc.dropna().groupby("mean_bin")["flaky"].agg(total="count", flaky="sum")
    grp["stable"] = grp["total"] - grp["flaky"]
    grp["flaky_frac"] = grp["flaky"] / grp["total"]
    print(grp)

    # ---- MAX TABLE ----
    print("\n=== By COUNTER MAX ===")
    grp = per_mc.dropna().groupby("cnt_max")["flaky"].agg(total="count", flaky="sum")
    grp["stable"] = grp["total"] - grp["flaky"]
    grp["flaky_frac"] = grp["flaky"] / grp["total"]
    print(grp)

    per_mc.to_csv("per_mc_transient.csv", index=False)
    print("\nWrote per_mc_transient.csv")


if __name__ == "__main__":
    main()
