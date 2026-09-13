"""Pull run records and event logs from GCS, parse them, emit the results table.

    python analyze.py --bucket BUCKET                 # every run found
    python analyze.py --bucket BUCKET --runs a b c    # specific run ids

Writes results/benchmark_raw.csv (one row per run, every metric) and prints a
comparison. Run records live in object storage; event logs are downloaded to a
local cache because the parser reads them as files.

Kept separate from run_one.py deliberately: measurement and analysis should not
share a process, so re-analysing never risks touching a measurement.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import config
import parse_eventlogs

GCLOUD = shutil.which("gcloud") or "gcloud"
CACHE = Path.home() / ".cache" / "dzone-eventlogs"


def _sh(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def list_runs(bucket: str) -> list[str]:
    out = _sh([GCLOUD, "storage", "ls", f"{bucket}/results/runs/"])
    return [
        line.rstrip("/").split("/")[-1]
        for line in out.replace("\r", "").splitlines()
        if line.strip().endswith("/")
    ]


def load_record(bucket: str, run_id: str) -> dict | None:
    raw = _sh([GCLOUD, "storage", "cat", f"{bucket}/results/runs/{run_id}/*"])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def fetch_log(bucket: str, app_id: str) -> Path | None:
    """Download an event log once and cache it locally."""
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / app_id
    if local.exists():
        return local
    _sh([GCLOUD, "storage", "cp", "-r", f"{bucket}/eventlogs/{app_id}", str(CACHE)])
    return local if local.exists() else None


ORDER = [
    ("uniform", "baseline"),
    ("uniform", "salted"),
    ("skewed", "baseline"),
    ("skewed", "salted"),
]

ROWS = [
    ("wall time (s)", "wall_s", 1.0, "{:>12.2f}"),
    ("executor CPU (s)", "executor_cpu_time_ms_total", 1000.0, "{:>12.1f}"),
    ("executor runtime (s)", "executor_run_time_ms_total", 1000.0, "{:>12.1f}"),
    ("", None, 1.0, ""),
    ("task ms  p50", "task_ms_p50", 1.0, "{:>12.1f}"),
    ("task ms  p95", "task_ms_p95", 1.0, "{:>12.1f}"),
    ("task ms  max", "task_ms_max", 1.0, "{:>12.1f}"),
    ("max / p50  (time)", "skew_ratio_max_over_p50", 1.0, "{:>12.2f}"),
    ("", None, 1.0, ""),
    ("shuffle read p50 (MB)", "shuffle_read_p50_bytes_task", 1e6, "{:>12.2f}"),
    ("shuffle read MAX (MB)", "shuffle_read_max_bytes_task", 1e6, "{:>12.2f}"),
    ("max / p50  (bytes)", "_byte_ratio", 1.0, "{:>12.2f}"),
    ("shuffle read total (MB)", "shuffle_read_bytes_total", 1e6, "{:>12.1f}"),
    ("shuffle write total (MB)", "shuffle_write_bytes_total", 1e6, "{:>12.1f}"),
    ("", None, 1.0, ""),
    ("disk spill (MB)", "disk_spill_bytes_total", 1e6, "{:>12.1f}"),
    ("GC time (s)", "gc_time_ms_total", 1000.0, "{:>12.1f}"),
    ("join stage tasks", "join_stage_tasks", 1.0, "{:>12.0f}"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--runs", nargs="*", help="specific run ids; default is all")
    args = ap.parse_args()

    bucket = args.bucket if args.bucket.startswith("gs://") else f"gs://{args.bucket}"
    run_ids = args.runs or list_runs(bucket)
    if not run_ids:
        print("No runs found.")
        return 1

    records = []
    for run_id in run_ids:
        rec = load_record(bucket, run_id)
        if not rec:
            print(f"  !! no record for {run_id}")
            continue
        log = fetch_log(bucket, rec["app_id"])
        if log:
            try:
                rec.update(parse_eventlogs.parse_run(log))
            except Exception as exc:  # noqa: BLE001 - keep the timing regardless
                print(f"  !! parse failed for {run_id}: {exc}")
                rec["parse_error"] = str(exc)
        p50 = rec.get("shuffle_read_p50_bytes_task") or 0
        rec["_byte_ratio"] = (
            round(rec.get("shuffle_read_max_bytes_task", 0) / p50, 2) if p50 else None
        )
        records.append(rec)

    # --- raw CSV -----------------------------------------------------------
    out = Path(config.REPO) / "results"
    out.mkdir(parents=True, exist_ok=True)

    # Correctness must survive into the CSV as scalars. An earlier version
    # dropped dict-valued columns when flattening, which silently discarded the
    # result checksum - so the report could not show that salting preserved the
    # answer, the single most important validity claim in the experiment.
    expected = None
    flat = []
    for r in records:
        row = {k: v for k, v in r.items() if not isinstance(v, dict)}
        ck = r.get("result_checksum") or {}
        if ck:
            row["result_groups"] = ck.get("groups")
            row["result_transaction_count"] = ck.get("total_transactions")
            row["result_total_sales"] = ck.get("total_sales")
            # A validation-SUMMARY checksum: a hash of {groups,
            # total_transactions, total_sales}, not of the 12 output rows. It
            # catches a changed total, not a redistribution between groups
            # that preserves the totals. The column keeps its historical name.
            #
            # Deterministic: sorted keys, stable separators, so the same
            # summary always hashes identically regardless of dict ordering.
            row["result_checksum"] = hashlib.sha256(
                json.dumps(ck, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()[:16]
            if expected is None:
                expected = row["result_checksum"]
            # True means "agrees with the first run's validation summary".
            row["correctness_verified"] = row["result_checksum"] == expected
        flat.append(row)
    fields: list[str] = []
    for r in flat:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (out / "benchmark_raw.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)

    # --- comparison table --------------------------------------------------
    cells = {}
    for wl, var in ORDER:
        matches = [
            r
            for r in records
            if r.get("workload") == wl and r.get("variant") == var and not r.get("warmup")
        ]
        if matches:
            cells[(wl, var)] = matches[-1]  # most recent

    present = [k for k in ORDER if k in cells]
    if not present:
        print("No measured runs to compare.")
        return 1

    head = f"{'metric':<26s}" + "".join(
        f"{wl[:4] + '/' + var[:4]:>13s}" for wl, var in present
    )
    print()
    print(head)
    print("-" * len(head))
    for label, key, div, fmt in ROWS:
        if key is None:
            print()
            continue
        line = f"{label:<26s}"
        for k in present:
            v = cells[k].get(key)
            line += fmt.format(v / div) if isinstance(v, (int, float)) else f"{'-':>12s}"
        print(line)

    # Correctness: salting must not change the answer.
    print()
    sums = {k: cells[k].get("result_checksum") for k in present}
    distinct = {json.dumps(v, sort_keys=True) for v in sums.values() if v}
    if len(distinct) == 1:
        print(f"[ok]   all runs agree on the result: {next(iter(distinct))}")
    elif distinct:
        print("[BAD]  runs disagree on the result - salting changed the answer:")
        for k, v in sums.items():
            print(f"         {k}: {v}")

    print(f"\nWrote {out / 'benchmark_raw.csv'} ({len(records)} runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
