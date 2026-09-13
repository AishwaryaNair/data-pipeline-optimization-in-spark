"""Turn the raw run records into the article's evidence.

    python report.py --bucket BUCKET

Deliberately reports distributions, not single numbers. The whole point of the
experiment is that a before/after pair cannot distinguish a real effect from
run-to-run variance, so every observation is shown alongside the median, and
each block's baseline/salted pair is compared directly.

Reads results/benchmark_raw.csv, which analyze.py builds from GCS.
"""

import argparse
import csv
import statistics as st
from pathlib import Path

import config

CELLS = [("uniform", "baseline"), ("uniform", "salted"),
         ("skewed", "baseline"), ("skewed", "salted")]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        if r.get("warmup", "").lower() == "true":
            continue
        rec = dict(r)
        for k, v in r.items():
            try:
                rec[k] = float(v)
            except (TypeError, ValueError):
                pass
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Path(config.REPO) / "results" / "benchmark_raw.csv"))
    args = ap.parse_args()

    rows = load(Path(args.csv))
    by_cell = {c: sorted([r for r in rows if (r["workload"], r["variant"]) == c],
                         key=lambda r: r.get("block", 0)) for c in CELLS}

    n = {c: len(v) for c, v in by_cell.items()}
    print(f"Measured runs per cell: {n}\n")

    # --- every observation -------------------------------------------------
    print("WALL TIME - every observation (seconds)")
    print(f"{'cell':<20s}" + "".join(f"{'b'+str(i):>8s}" for i in range(1, 6))
          + f"{'median':>10s}{'min':>8s}{'max':>8s}{'spread':>9s}")
    print("-" * 83)
    medians = {}
    for c in CELLS:
        vals = [r["wall_s"] for r in by_cell[c]]
        if not vals:
            continue
        med = st.median(vals)
        medians[c] = med
        spread = (max(vals) - min(vals)) / med * 100
        label = f"{c[0]}/{c[1]}"
        print(f"{label:<20s}" + "".join(f"{v:>8.2f}" for v in vals)
              + f"{med:>10.2f}{min(vals):>8.2f}{max(vals):>8.2f}{spread:>8.1f}%")

    # --- paired per-block deltas ------------------------------------------
    print("\nPAIRED DELTAS  (salted - baseline) / baseline, per block")
    print(f"{'workload':<12s}" + "".join(f"{'b'+str(i):>9s}" for i in range(1, 6))
          + f"{'median':>10s}")
    print("-" * 67)
    for wl in ("uniform", "skewed"):
        base = {r.get("block"): r["wall_s"] for r in by_cell[(wl, "baseline")]}
        salt = {r.get("block"): r["wall_s"] for r in by_cell[(wl, "salted")]}
        deltas = []
        cells = []
        for b in range(1, 6):
            if b in base and b in salt and base[b]:
                d = (salt[b] - base[b]) / base[b] * 100
                deltas.append(d)
                cells.append(f"{d:>+8.1f}%")
            else:
                cells.append(f"{'-':>9s}")
        med = f"{st.median(deltas):>+9.1f}%" if deltas else f"{'-':>10s}"
        print(f"{wl:<12s}" + "".join(cells) + med)
        if deltas:
            signs = {d > 0 for d in deltas}
            note = ("SIGN FLIPS between blocks" if len(signs) > 1
                    else "consistent direction")
            print(f"{'':<12s}  -> {note}")

    # --- structural metrics -----------------------------------------------
    print("\nSTRUCTURAL METRICS (medians) - deterministic, unlike wall time")
    keys = [
        ("shuffle max/p50 (bytes)", "_byte_ratio", 1.0, "{:>13.2f}"),
        ("largest task read (MB)", "shuffle_read_max_bytes_task", 1e6, "{:>13.2f}"),
        ("typical task read (MB)", "shuffle_read_p50_bytes_task", 1e6, "{:>13.2f}"),
        ("shuffle total (MB)", "shuffle_read_bytes_total", 1e6, "{:>13.1f}"),
        ("task max/p50 (time)", "skew_ratio_max_over_p50", 1.0, "{:>13.2f}"),
        ("executor CPU (s)", "executor_cpu_time_ms_total", 1000.0, "{:>13.1f}"),
    ]
    print(f"{'metric':<26s}" + "".join(f"{c[0][:4]+'/'+c[1][:4]:>13s}" for c in CELLS))
    print("-" * 78)
    for label, key, div, fmt in keys:
        line = f"{label:<26s}"
        for c in CELLS:
            vals = [r[key] for r in by_cell[c]
                    if isinstance(r.get(key), float)]
            line += fmt.format(st.median(vals) / div) if vals else f"{'-':>13s}"
        print(line)

    # --- correctness -------------------------------------------------------
    checksums = {r.get("result_checksum") for r in rows if r.get("result_checksum")}
    txns = {r.get("result_transaction_count") for r in rows
            if r.get("result_transaction_count")}
    passed = sum(1 for r in rows if str(r.get("correctness_verified")).lower() == "true")

    print("\nCORRECTNESS")
    print(f"  {passed}/{len(rows)} measured runs passed")
    if txns:
        vals = ", ".join(f"{int(t):,}" for t in sorted(txns))
        print(f"  transaction_count = {vals} "
              f"{'in every run' if len(txns) == 1 else '<- DIFFERS, investigate'}")
    if checksums:
        print(f"  result checksum = {'identical across all variants' if len(checksums) == 1 else 'DIFFERS: ' + str(checksums)}")
        print(f"    {next(iter(checksums))}")

    bad = [r["run_id"] for r in rows if str(r.get("single_execution")).lower() == "false"]
    print(f"  single-execution guard: "
          f"{'all runs clean' if not bad else 'FAILED: ' + str(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
