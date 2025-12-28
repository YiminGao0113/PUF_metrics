import sys
import re
import numpy as np
import pandas as pd

# ================= CONFIG =================
PUF_SIG = "Out0"
PUF_THRESH = 0.4
BIT_THRESH = 0.4

CORNERS = ["Nominal", "ll", "lh", "hl", "hh"]
QS = ["q0", "q1", "q2", "q3"]  # q3 = MSB
# =========================================

PARAM_RE = re.compile(r"Parameters:\s*mc_iteration\s*=\s*(\d+)", re.IGNORECASE)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def bit(v, thr):
    if np.isnan(v):
        return np.nan
    return 1 if v > thr else 0


def counter_from_bits(q0, q1, q2, q3):
    if any(np.isnan([q0, q1, q2, q3])):
        return np.nan
    return (int(q3) << 3) | (int(q2) << 2) | (int(q1) << 1) | int(q0)


def print_counter_distribution(df, corner):
    flip_col = f"flip_{corner}"
    cnt_col = f"cnt_{corner}"

    print(f"\n=== Counter distribution @ {corner} ===")
    print("counter | stable | flaky | total | flaky_frac")

    vals = sorted(df[cnt_col].dropna().unique())
    if not vals:
        print("(no data)")
        return

    for c in vals:
        sub = df[df[cnt_col] == c]
        flaky = int(np.sum(sub[flip_col] == 1))
        stable = int(np.sum(sub[flip_col] == 0))
        total = stable + flaky
        frac = flaky / total if total > 0 else np.nan
        print(f"{int(c):>7} | {stable:>6} | {flaky:>5} | {total:>5} | {frac:>9.3f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python corner_reliability_counter.py corner_counter.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # -------- Parse CSV --------
    rows = []
    current_mc = None
    header = None
    idx = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = PARAM_RE.match(line)
            if m:
                current_mc = int(m.group(1))
                continue

            if line.startswith("Point,Test,Output"):
                header = [h.strip() for h in line.split(",")]
                for k in ["Output"] + CORNERS:
                    if k not in header:
                        raise SystemExit(f"Missing column {k}")
                    idx[k] = header.index(k)
                continue

            if current_mc is None or header is None:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) <= max(idx.values()):
                continue

            out = parts[idx["Output"]]
            if not out.startswith("E_"):
                continue

            sig = out[2:]
            vals = {c: safe_float(parts[idx[c]]) for c in CORNERS}

            rows.append((current_mc, sig, *[vals[c] for c in CORNERS]))

    df = pd.DataFrame(rows, columns=["mc", "sig"] + CORNERS)
    if df.empty:
        raise SystemExit("No data parsed.")

    # -------- Organize per MC --------
    per_mc = {}
    for r in df.itertuples(index=False):
        per_mc.setdefault(r.mc, {})[r.sig] = {c: getattr(r, c) for c in CORNERS}

    records = []

    for mc, d in per_mc.items():
        if PUF_SIG not in d or any(q not in d for q in QS):
            continue

        rec = {"mc": mc}

        # PUF bits
        puf_bits = {c: bit(d[PUF_SIG][c], PUF_THRESH) for c in CORNERS}

        for c in CORNERS:
            rec[f"puf_{c}"] = puf_bits[c]
            if c != "Nominal":
                rec[f"flip_{c}"] = (
                    np.nan if np.isnan(puf_bits["Nominal"]) or np.isnan(puf_bits[c])
                    else int(puf_bits["Nominal"] != puf_bits[c])
                )

        # Counters
        for c in CORNERS:
            qbits = [bit(d[q][c], BIT_THRESH) for q in QS]
            rec[f"cnt_{c}"] = counter_from_bits(*qbits)

        records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        raise SystemExit("No complete MC rows.")

    # -------- Reliability --------
    print("\n=== Corner Reliability ===")
    print(f"MC instances: {len(out)}")

    for c in CORNERS:
        if c == "Nominal":
            continue
        flip_rate = np.nanmean(out[f"flip_{c}"])
        print(f"{c}: flip_rate={flip_rate:.6f}  reliability={1-flip_rate:.6f}")

    # Worst-case flip
    flip_cols = [f"flip_{c}" for c in CORNERS if c != "Nominal"]
    out["flip_any"] = out[flip_cols].max(axis=1)

    wc = np.nanmean(out["flip_any"])
    print(f"\nWorst-case (ANY corner) reliability: {1-wc:.6f}")

    # -------- Counter vs Flakiness --------
    for c in CORNERS:
        if c != "Nominal":
            print_counter_distribution(out, c)

    # -------- MC indices that flip --------
    for c in CORNERS:
        if c == "Nominal":
            continue
        bad = out[out[f"flip_{c}"] == 1][
            ["mc", f"cnt_Nominal", f"cnt_{c}", "puf_Nominal", f"puf_{c}"]
        ]
        print(f"\n=== MC indices that flip @ {c} ===")
        print(bad.to_string(index=False) if len(bad) else "(none)")


if __name__ == "__main__":
    main()
