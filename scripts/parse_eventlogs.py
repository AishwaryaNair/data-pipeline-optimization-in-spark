"""Extract task-level metrics from Spark event logs.

Wall-clock duration alone cannot distinguish "faster because it did less work"
from "faster because the work was spread more evenly". These metrics can, and
Spark already records them — no custom instrumentation required.

The stage with the largest shuffle read is treated as the join stage; that is
where skew manifests, and reporting whole-run percentiles would dilute it with
hundreds of trivial tasks from other stages.
"""

import gzip
import io
import json
from pathlib import Path

import numpy as np


def _open_log(path: Path):
    """Open an event-log file, transparently decompressing if needed.

    Spark 4 writes rolling event logs and compresses them with zstd by
    default. spark_session.py disables compression, but a reader running with
    different defaults would still produce .zstd, so handle both.
    """
    if path.suffix == ".zstd":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise RuntimeError(
                f"{path.name} is zstd-compressed but the `zstandard` package "
                f"is not installed. Either `pip install zstandard`, or set "
                f"spark.eventLog.compress=false and re-run."
            ) from exc
        fh = path.open("rb")
        return io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(fh),
            encoding="utf-8",
            errors="replace",
        )
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def _iter_events(eventlog_dir: Path):
    # Two shapes in the wild:
    #   Spark 4 (local)      rolling logs in an eventlog_v2_<appId>/ directory
    #   Spark 3.5 (Dataproc) a single file named after the application
    # `appstatus_*` is a zero-byte completion marker, not JSON.
    if eventlog_dir.is_file():
        files = [eventlog_dir]
    else:
        files = [
            p
            for p in eventlog_dir.rglob("*")
            if p.is_file() and not p.name.startswith("appstatus")
        ]
    if not files:
        raise FileNotFoundError(f"no event log found under {eventlog_dir}")
    for path in sorted(files):
        with _open_log(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _task_record(ev: dict) -> dict | None:
    info = ev.get("Task Info") or {}
    metrics = ev.get("Task Metrics") or {}
    if not metrics or info.get("Failed"):
        return None

    sr = metrics.get("Shuffle Read Metrics") or {}
    sw = metrics.get("Shuffle Write Metrics") or {}
    shuffle_read = sr.get("Remote Bytes Read", 0) + sr.get("Local Bytes Read", 0)

    return {
        "stage_id": ev.get("Stage ID"),
        "run_time_ms": metrics.get("Executor Run Time", 0),
        # Spark reports CPU time in nanoseconds.
        "cpu_time_ms": metrics.get("Executor CPU Time", 0) / 1e6,
        "gc_time_ms": metrics.get("JVM GC Time", 0),
        "shuffle_read_bytes": shuffle_read,
        "shuffle_write_bytes": sw.get("Shuffle Bytes Written", 0),
        "memory_spill_bytes": metrics.get("Memory Bytes Spilled", 0),
        "disk_spill_bytes": metrics.get("Disk Bytes Spilled", 0),
    }


def parse_run(eventlog_dir: Path) -> dict:
    tasks, aqe_skew, aqe_updates, sql_executions = [], False, 0, 0

    for ev in _iter_events(Path(eventlog_dir)):
        name = ev.get("Event", "")
        if name == "SparkListenerTaskEnd":
            rec = _task_record(ev)
            if rec:
                tasks.append(rec)
        elif name.endswith("SparkListenerSQLExecutionStart"):
            # Totals below sum every task in the application. If the job ran
            # the query more than once, those totals are multiples of the real
            # figure - a mistake that leaves wall time untouched and is
            # therefore easy to miss. Counted here so it cannot pass silently.
            sql_executions += 1
        elif "AdaptiveExecutionUpdate" in name:
            aqe_updates += 1
            # AQE labels a shuffle read it split for skew; if the substring
            # never appears, AQE's skew-join rule did not engage.
            if "skewed" in json.dumps(ev.get("physicalPlanDescription", "")):
                aqe_skew = True

    if not tasks:
        raise ValueError(f"no successful tasks found in {eventlog_dir}")

    def totals(key):
        return sum(t[key] for t in tasks)

    # Join stage = the stage that read the most shuffle data.
    by_stage: dict[int, list] = {}
    for t in tasks:
        by_stage.setdefault(t["stage_id"], []).append(t)
    join_stage_id = max(
        by_stage, key=lambda s: sum(t["shuffle_read_bytes"] for t in by_stage[s])
    )
    # Stages moving at least half the join stage's shuffle bytes. Exactly one
    # is expected; more means the query was executed more than once.
    join_reads = sum(t["shuffle_read_bytes"] for t in by_stage[join_stage_id])
    heavy_stages = sum(
        1
        for s, ts in by_stage.items()
        if join_reads and sum(t["shuffle_read_bytes"] for t in ts) >= join_reads * 0.5
    )

    stage_tasks = by_stage[join_stage_id]
    durs = np.array([t["run_time_ms"] for t in stage_tasks], dtype=float)
    reads = np.array([t["shuffle_read_bytes"] for t in stage_tasks], dtype=float)

    p50 = float(np.percentile(durs, 50))
    p95 = float(np.percentile(durs, 95))
    dmax = float(durs.max())

    return {
        # Metric scope, stated explicitly so the article never mixes them:
        #   *_total            summed over EVERY task in the application
        #   task_ms_*, *_task  the join stage only (largest shuffle read)
        # Duplicate-execution guard. Counting SQLExecutionStart events is not a
        # usable signal: spark.read.parquet() registers one of its own for
        # schema/footer reading, so a correct single-action job still reports
        # two. Counting heavy shuffle stages is reliable - the query has one
        # join, so exactly one stage should move a large share of the bytes. If
        # the query ran twice there are two such stages, and every *_total
        # below is a multiple of the true figure.
        # Exactly one, not "at most one": zero heavy stages would mean the join
        # never shuffled, which is as much a broken run as two executions.
        "sql_executions": sql_executions,
        "heavy_shuffle_stages": heavy_stages,
        "single_execution": heavy_stages == 1,
        "num_tasks_total": len(tasks),
        "join_stage_id": int(join_stage_id),
        "join_stage_tasks": len(stage_tasks),
        "task_ms_p50": round(p50, 1),
        "task_ms_p95": round(p95, 1),
        "task_ms_max": round(dmax, 1),
        # The simplest honest skew indicator: how much worse is the worst task
        # than the typical one.
        "skew_ratio_max_over_p50": round(dmax / p50, 2) if p50 > 0 else None,
        "shuffle_read_max_bytes_task": int(reads.max()),
        "shuffle_read_p50_bytes_task": int(np.percentile(reads, 50)),
        "shuffle_read_bytes_total": int(totals("shuffle_read_bytes")),
        "shuffle_write_bytes_total": int(totals("shuffle_write_bytes")),
        "memory_spill_bytes_total": int(totals("memory_spill_bytes")),
        "disk_spill_bytes_total": int(totals("disk_spill_bytes")),
        # Compute-work proxy. NOT a dollar figure: these runs are local.
        "executor_run_time_ms_total": int(totals("run_time_ms")),
        "executor_cpu_time_ms_total": int(totals("cpu_time_ms")),
        "gc_time_ms_total": int(totals("gc_time_ms")),
        "aqe_plan_updates": aqe_updates,
        "aqe_skew_split_detected": aqe_skew,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(parse_run(Path(sys.argv[1])), indent=2))
