"""Execute exactly one measured run.

The entry point of one Dataproc batch, submitted by submit_cloud.py. Every
run is a fresh JVM, which is the point: a warm JVM is measurably faster, so
sharing one across runs would hand JIT warm-up to whichever variant ran later.

    run_one.py --workload skewed --variant salted --run-id r001 \
        --data-root gs://BUCKET/data --results-root gs://BUCKET/results \
        --eventlog-root gs://BUCKET/eventlogs
"""

import argparse
import json
import time
from pathlib import Path

import config
import spark_session


def _write_record(spark, record: dict, run_id: str) -> str:
    """Persist the run record to object storage."""
    dest = f"{config.RESULTS_ROOT}/runs/{run_id}"
    payload = json.dumps(record, indent=2)

    # Spark already has the GCS connector configured; using it avoids adding a
    # cloud SDK dependency to the job image.
    spark.createDataFrame([(payload,)], "json string").coalesce(1).write.mode(
        "overwrite"
    ).text(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True, choices=config.WORKLOADS)
    ap.add_argument("--variant", required=True, choices=config.VARIANTS)
    ap.add_argument("--query", default=config.DEFAULT_QUERY, choices=config.QUERIES)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--warmup", action="store_true")
    # Required: there is no default root, and an unset one would silently
    # resolve to paths like "/sales_uniform".
    ap.add_argument("--data-root", required=True, help="gs:// URI holding the datasets")
    ap.add_argument("--results-root", required=True, help="gs:// URI for run records")
    ap.add_argument("--eventlog-root", required=True, help="gs:// URI for event logs")
    args = ap.parse_args()

    config.set_roots(
        data_root=args.data_root,
        results_root=args.results_root,
        eventlog_root=args.eventlog_root,
    )

    # Spark requires spark.eventLog.dir to already exist. Object storage has no
    # real directories, so a per-run path cannot be created ahead of the
    # session. Every run therefore shares one pre-created root and Spark writes
    # eventlog_v2_<appId>/ inside it - `app_id` below is what maps a log back
    # to its run.
    eventlog_dir = config.EVENTLOG_ROOT
    spark = spark_session.build(f"bench-{args.run_id}", eventlog_dir=eventlog_dir)

    import pipeline

    df = pipeline.build(spark, args.workload, args.variant, args.query)

    # Plan is materialised BEFORE the clock starts, so query planning is not
    # counted as execution time.
    plan = df._jdf.queryExecution().executedPlan().toString()

    # The experiment is void if Spark broadcast the dimension: there would be
    # no shuffle on the join and therefore no skew. autoBroadcastJoinThreshold
    # is set to -1 to prevent it, but assert rather than trust.
    if "SortMergeJoin" not in plan:
        print("PLAN ASSERTION FAILED - expected SortMergeJoin, got:\n")
        print(plan[:4000])
        spark.stop()
        return 2

    # EXACTLY ONE measured benchmark action. A small post-measurement Spark
    # write persists the run metadata further down; it happens after wall_s is
    # recorded and is far too small to register as a heavy shuffle stage, but
    # it does contribute to the application-wide *_total metrics.
    #
    # An earlier version timed a noop write and then called collect() for the
    # validation summary - which re-executed the whole query, so the event log
    # contained two heavy executions and every summed metric (shuffle bytes,
    # executor CPU) came out doubled. Wall time was unaffected, which is what
    # made it easy to miss.
    #
    # collect() is safe as the timed action here because the result is 12 rows
    # (4 regions x 3 segments): no meaningful driver transfer, and unlike
    # count() it cannot be satisfied by a cheaper plan. One execution, one
    # timing, and the validation summary falls out of the same action.
    t0 = time.perf_counter()
    result = df.collect()
    wall_s = time.perf_counter() - t0

    rows = sorted(
        (r["region"], r["segment"], r["transaction_count"], float(r["total_sales"]))
        for r in result
    )
    # Validation SUMMARY, not a hash of the full result. It aggregates the 12
    # output rows, so it detects a changed total but not a redistribution
    # between groups that preserves the totals. analyze.py hashes this summary
    # and stores it as result_checksum.
    checksum = {
        "groups": len(rows),
        "total_transactions": sum(r[2] for r in rows),
        "total_sales": round(sum(r[3] for r in rows), 2),
    }
    print(f"  validation summary: {checksum}")

    conf = spark.sparkContext.getConf()
    record = {
        "run_id": args.run_id,
        "workload": args.workload,
        "variant": args.variant,
        "query": args.query,
        "rep": args.rep,
        "block": args.block,
        # wall_s measures ONLY the single Spark action inside this process.
        # It deliberately excludes gcloud submission, serverless provisioning,
        # driver startup and teardown, which are control-plane variance rather
        # than pipeline performance. The submitter records that separately as
        # batch_elapsed_s.
        "warmup": args.warmup,
        "wall_s": round(wall_s, 4),
        "app_id": spark.sparkContext.applicationId,
        "eventlog_dir": eventlog_dir,
        "plan_has_smj": True,
        "result_checksum": checksum,
        "fact_rows": config.FACT_ROWS,
        "salt_buckets": config.SALT_BUCKETS,
        # Recorded per run so the article can state the configuration from
        # evidence rather than from intent.
        "spark_version": spark.version,
        "shuffle_partitions": conf.get("spark.sql.shuffle.partitions", "?"),
        "aqe_enabled": conf.get("spark.sql.adaptive.enabled", "?"),
        "broadcast_threshold": conf.get("spark.sql.autoBroadcastJoinThreshold", "?"),
        "executor_instances": conf.get("spark.executor.instances", "?"),
        "executor_cores": conf.get("spark.executor.cores", "?"),
        "executor_memory": conf.get("spark.executor.memory", "?"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    dest = _write_record(spark, record, args.run_id)
    spark.stop()

    tag = "warmup" if args.warmup else f"rep{args.rep}"
    print(f"{args.workload:<8s} {args.variant:<9s} {tag:<7s} {wall_s:7.2f}s  -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
