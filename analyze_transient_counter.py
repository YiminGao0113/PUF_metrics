#!/usr/bin/env python3
import re
import sys
import numpy as np
import pandas as pd

# --- config (edit here if needed) ---
THRESH_V = 0.4
PUF_SIG = "Out0"
COUNTER_BITS = ["q0", "q1", "q2", "q3"]  # LSB -> MSB (q3 is MSB)
# -----------------------------------

PARAM_RE = re.compile(r"Parameters:\s*mc_iteration\s*=\s*(\d+)", re.IGNORECASE)
OUT_RE = re.compile(r"^E_(Out0|q0|q1|q2|q3)_iter(\d+)$", re.IGNORECASE)

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
            val_s = parts[3]

            m2 = OUT_RE.match(output)
            if not m2:
                continue

            sig = m2.group(1)
            it = int(m2.group(2))

            try:
                val = float(val_s)
            except ValueError:
                val = np.nan

            rows.append((current_mc, sig, it, val))

    df = pd.DataFrame(rows, columns=["mc", "sig", "iter", "value"])
    return df

def build_tensor(df: pd.DataFrame, n_mc: int, n_iter: int, sigs):
    """
    tensor[sig] -> (n_mc, n_iter) float with NaNs
    mc is 1-based in file
    """
    tensor = {}
    for sig in sigs:
        mat = np.full((n_mc, n_iter), np.nan, dtype=float)
        sub = df[df["sig"].str.lower() == sig.lower()]
        for _, r in sub.iterrows():
            mc = int(r["mc"])
            it = int(r["iter"])
            if 1 <= mc <= n_mc and 0 <= it < n_iter:
                mat[mc - 1, it] = float(r["value"])
        tensor[sig] = mat
    return tensor

def to_bits(mat: np.ndarray, thresh: float) -> np.ndarray:
    # returns float array {0,1,nan}
    return np.where(np.isnan(mat), np.nan, (mat > thresh).astype(float))

def puf_reliability(bits: np.ndarray):
    """
    bits: (n_mc, n_iter) {0,1,nan}
    """
    ones = np.nansum(bits == 1, axis=1)
    zeros = np.nansum(bits == 0, axis=1)
    valid = ones + zeros

    maj = np.where(ones >= zeros, 1.0, 0.0)  # ties -> 1
    matches = np.where(np.isnan(bits), np.nan, (bits == maj[:, None]).astype(float))
    stability = np.nanmean(matches, axis=1)  # fraction matching majority

    strict = np.full(bits.shape[0], np.nan, dtype=float)
    for i in range(bits.shape[0]):
        b = bits[i, :]
        b = b[~np.isnan(b)]
        if b.size == 0:
            strict[i] = np.nan
        else:
            strict[i] = 1.0 if np.all(b == b[0]) else 0.0

    missing = int(np.sum(valid == 0))
    partial = int(np.sum((valid > 0) & (valid < bits.shape[1])))

    return maj, stability, strict, missing, partial

def counter_values(q_bits_dict):
    """
    q_bits_dict: q0..q3 each (n_mc,n_iter) {0,1,nan}
    returns cnt (n_mc,n_iter) float with nan if any bit nan in that read
    """
    q0, q1, q2 = (q_bits_dict["q0"], q_bits_dict["q1"], q_bits_dict["q2"])
    valid = ~(np.isnan(q0) | np.isnan(q1) | np.isnan(q2))
    cnt = np.full(q0.shape, np.nan, dtype=float)
    cnt[valid] = (q0[valid] + 2*q1[valid] + 4*q2[valid])
    return cnt

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_transient_counter.py <nominal_counter.csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = parse_csv_blocks(path)
    if df.empty:
        raise SystemExit("Parsed 0 rows. Check Output names: E_Out0_iter#, E_q*_iter#")

    n_mc = int(df["mc"].max())
    n_iter = int(df["iter"].max()) + 1

    sigs = [PUF_SIG] + COUNTER_BITS
    tensor = build_tensor(df, n_mc, n_iter, sigs)

    # bits
    puf_bits = to_bits(tensor[PUF_SIG], THRESH_V)
    q_bits = {q: to_bits(tensor[q], THRESH_V) for q in COUNTER_BITS}

    # PUF reliability
    maj, stability, strict, missing, partial = puf_reliability(puf_bits)
    unstable_any_flip = int(np.sum((stability < 1.0) & ~np.isnan(stability)))

    print("=== Summary ===")
    print(f"MC instances:        {n_mc}")
    print(f"Iterations (reads):  {n_iter}")
    print(f"Threshold (V):       {THRESH_V}")
    print()
    print(f"Overall reliability (majority stability mean): {np.nanmean(stability):.6f}")
    print(f"Overall reliability (strict all-same mean):    {np.nanmean(strict):.6f}")
    print()
    print(f"Unstable MC (any flip):             {unstable_any_flip}")
    print(f"Missing MC rows (0 valid samples):  {missing}")
    print(f"Partially missing MC rows:          {partial}")

    # flaky definition: ANY flip in PUF across reads (within a given MC)
    flaky = ((stability < 1.0) & ~np.isnan(stability)).astype(int)

    # counter per read
    cnt = counter_values(q_bits)  # (n_mc,n_iter)
    cnt_mode = np.full(n_mc, np.nan, dtype=float)

    for i in range(n_mc):
        v = cnt[i, :]
        v = v[~np.isnan(v)]
        if v.size == 0:
            continue
        # mode
        vals, counts = np.unique(v.astype(int), return_counts=True)
        cnt_mode[i] = float(vals[np.argmax(counts)])

    per_mc = pd.DataFrame({
        "mc_index_1based": np.arange(1, n_mc+1),
        "puf_majority_bit": maj,
        "puf_stability": stability,
        "puf_strict_all_same": strict,
        "flaky": flaky,
        "counter_mode": cnt_mode,
    })

    # overall counter distribution by read (flatten all valid reads)
    flat_cnt = cnt.reshape(-1)
    flat_cnt = flat_cnt[~np.isnan(flat_cnt)].astype(int)

    print("\n=== Counter distribution (ALL valid reads) ===")
    if flat_cnt.size == 0:
        print("No valid counter samples found (q0..q3 missing).")
    else:
        vc = pd.Series(flat_cnt).value_counts().sort_index()
        for k, c in vc.items():
            print(f"counter={k:2d}  count={int(c)}")

    # counter distribution split by stable vs flaky (use per-MC counter_mode)
    print("\n=== Counter distribution by MC (mode), stable vs flaky ===")
    sub = per_mc.dropna(subset=["counter_mode"]).copy()
    sub["counter_mode"] = sub["counter_mode"].astype(int)

    grp = sub.groupby("counter_mode")["flaky"].agg(
        total="count",
        flaky_count="sum"
    )
    grp["stable"] = grp["total"] - grp["flaky_count"]
    grp["flaky_frac"] = grp["flaky_count"] / grp["total"]
    grp = grp[["stable", "flaky_count", "total", "flaky_frac"]]

    if grp.empty:
        print("No valid per-MC counter_mode values.")
    else:
        print("counter | stable | flaky | total | flaky_frac")
        for idx, r in grp.iterrows():
            print(f"{idx:7d} | {int(r['stable']):6d} | {int(r['flaky_count']):5d} | {int(r['total']):5d} | {r['flaky_frac']:.3f}")

    # write per-mc table next to input file
    out_path = "per_mc_transient.csv"
    per_mc.to_csv(out_path, index=False)
    print(f"\nWrote per-MC table: {out_path}")

if __name__ == "__main__":
    main()
