"""Submit the benchmark to Dataproc Serverless, one batch per run.

Runs from your laptop; each execution becomes its own Spark batch. That is
deliberate - one batch per run gives every measurement a cold JVM and its own
event log, which is the same guarantee the local runner gets from spawning a
subprocess per run.

    python submit_cloud.py --upload            # push scripts to GCS
    python submit_cloud.py --generate          # build the datasets
    python submit_cloud.py --smoke             # one run, end to end
    python submit_cloud.py                     # the full 24-run matrix

Every run uses identical Spark properties. Executor count, cores and memory
are pinned and autoscaling is disabled: if the cluster resized between runs we
would be measuring elasticity rather than the optimization.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config

# On Windows gcloud is a .cmd wrapper, which CreateProcess will not launch by
# bare name. Resolve the full path once.
GCLOUD = shutil.which("gcloud") or "gcloud"

# Not hard-coded: this repo is published alongside the article, and a reader
# reproducing it needs their own project and bucket. Supply via --project /
# --bucket, or export GCP_PROJECT / GCS_BUCKET.
PROJECT = os.environ.get("GCP_PROJECT", "")
REGION = os.environ.get("GCP_REGION", "us-central1")
BUCKET = os.environ.get("GCS_BUCKET", "")

SCRIPTS = DATA_ROOT = RESULTS_ROOT = EVENTLOG_ROOT = ""


def _set_target(project: str, bucket: str, region: str) -> None:
    global PROJECT, BUCKET, REGION, SCRIPTS, DATA_ROOT, RESULTS_ROOT, EVENTLOG_ROOT
    PROJECT = project or PROJECT
    REGION = region or REGION
    BUCKET = (bucket or BUCKET).rstrip("/")
    if BUCKET and not BUCKET.startswith("gs://"):
        BUCKET = f"gs://{BUCKET}"
    SCRIPTS = f"{BUCKET}/scripts"
    DATA_ROOT = f"{BUCKET}/data"
    RESULTS_ROOT = f"{BUCKET}/results"
    EVENTLOG_ROOT = f"{BUCKET}/eventlogs"

# Shipped alongside the entry point so the job can import them.
DEPS = ["config.py", "spark_session.py", "pipeline.py"]

# Pin the runtime. Left unspecified, gcloud warns the default "may change at
# any time" - which would quietly invalidate the reproducibility claim if a new
# default landed partway through the matrix. 2.2.x was verified working here;
# the exact patch version is recorded per run from `batches describe`.
RUNTIME_VERSION = "2.2"

# Properties that MUST be in effect for a run to be valid, checked against what
# the service actually applied. A silently-dropped property would leave AQE on
# and broadcast joins allowed, producing plausible-looking numbers that measure
# nothing.
#
# Keys are matched by suffix, because the service prefixes them by namespace
# ("spark:spark.sql...", "dataproc:dataproc.tier") and the prefix is not
# something to hard-code.
REQUIRED_APPLIED = {
    "dataproc.tier": "standard",
    "spark.dataproc.engine": "default",
    "spark.sql.adaptive.enabled": "false",
    "spark.sql.autoBroadcastJoinThreshold": "-1",
    "spark.dynamicAllocation.enabled": "false",
    "spark.executor.instances": "2",
    "spark.executor.cores": "4",
    "spark.executor.memory": "16g",
    "spark.driver.cores": "4",
    "spark.sql.shuffle.partitions": "200",
}

# spark.driver.memory is submitted but deliberately absent above: it is not
# among the properties `gcloud dataproc batches describe` was observed to
# return, and a guard that always fails is worse than no guard. Add it once
# it is confirmed present in a describe response.


def _prop(props: dict, name: str):
    """Look up a property by its unprefixed name."""
    for k, v in props.items():
        if k == name or k.endswith(f":{name}") or k.endswith(f".{name}"):
            return v
    return None

# Seconds to wait between submissions so a finished batch releases its CPU
# quota before the next one asks for it.
SUBMIT_PAUSE_S = 30

# Pinned for every run.
#
# Comma-separated, gcloud's default. Do NOT use the `^;^` alternate-delimiter
# form here: on Windows the carets are consumed by cmd before gcloud sees them,
# the first property arrives as ";spark.executor.instances", and gcloud
# silently drops it as a non-Spark property - which would unpin the compute
# configuration without failing the job. None of these values contain a comma,
# so the default delimiter is correct anyway.
SPARK_PROPERTIES = ",".join(
    [
        # Pin the tier. `standard` is currently the default, but defaults can
        # change and a silently-premium run would put Lightning Engine in the
        # path - a second optimization layer underneath the one being measured.
        # Pinning also keeps spark.dataproc.engine=default.
        "dataproc.tier=standard",
        # --- fixed compute. The whole experiment depends on this not moving.
        #
        # 4 driver + 2 x 4 executor = 12 vCPU per run. Sized against this
        # project's CPUS_ALL_REGIONS quota of 32: a finished batch does not
        # release its quota instantly, so at 20 vCPU/run the next submission
        # collided with its own predecessor and failed. At 12, two overlapping
        # runs still fit inside 32 with room to spare.
        "spark.executor.instances=2",
        "spark.executor.cores=4",
        "spark.executor.memory=16g",
        "spark.driver.cores=4",
        "spark.driver.memory=16g",
        # Serverless Spark autoscales by default, which would silently change
        # executor count mid-benchmark and make runs incomparable.
        "spark.dynamicAllocation.enabled=false",
        # --- the two deliberate overrides, same as local. Neither is a
        # production recommendation; both hold the plan constant.
        "spark.sql.adaptive.enabled=false",
        "spark.sql.autoBroadcastJoinThreshold=-1",
        f"spark.sql.shuffle.partitions={config.SHUFFLE_PARTITIONS}",
        # Keep event logs readable without a decompression dependency.
        "spark.eventLog.compress=false",
    ]
)


def sh(cmd: list[str], check: bool = True) -> int:
    print("  $ " + " ".join(cmd[:6]) + (" ..." if len(cmd) > 6 else ""))
    proc = subprocess.run(cmd, text=True)
    if check and proc.returncode != 0:
        print(f"  !! exited {proc.returncode}")
    return proc.returncode


def upload() -> int:
    """Push the source files the batches need."""
    src = config.REPO / "scripts"
    # submit_cloud.py and analyze.py are uploaded too, not because batches run
    # them, but so the orchestrator itself can be pulled down and run from
    # Cloud Shell - which removes the local machine from the experiment
    # entirely.
    files = [
        str(src / f)
        for f in DEPS
        + [
            "generate_data.py",
            "run_one.py",
            "submit_cloud.py",
            "analyze.py",
            "parse_eventlogs.py",
        ]
    ]
    rc = sh([GCLOUD, "storage", "cp", *files, f"{SCRIPTS}/"])
    if rc != 0:
        return rc

    # Spark refuses to start if spark.eventLog.dir does not exist, and object
    # storage has no directories to create. Writing any object under the prefix
    # makes the GCS connector report it as an existing directory.
    keep = Path(config.REPO) / "results" / ".keep"
    keep.parent.mkdir(parents=True, exist_ok=True)
    keep.write_text("placeholder so the eventlog prefix exists\n", encoding="utf-8")
    return sh([GCLOUD, "storage", "cp", str(keep), f"{EVENTLOG_ROOT}/.keep"])


def submit(entry: str, batch_id: str, job_args: list[str], wait: bool = True) -> int:
    """Fire a batch. With --async gcloud returns as soon as it is accepted.

    Blocking submission ties the run's fate to a local streaming process that
    must survive minutes; killing it loses the tracking even though the batch
    completes fine in the cloud. Orchestration should depend on batch state,
    not on a CLI subprocess staying alive.
    """
    py_files = ",".join(f"{SCRIPTS}/{d}" for d in DEPS)
    cmd = [
        GCLOUD, "dataproc", "batches", "submit", "pyspark",
        f"{SCRIPTS}/{entry}",
        f"--batch={batch_id}",
        f"--project={PROJECT}",
        f"--region={REGION}",
        f"--version={RUNTIME_VERSION}",
        f"--deps-bucket={BUCKET}",
        f"--py-files={py_files}",
        f"--properties={SPARK_PROPERTIES}",
    ]
    if not wait:
        cmd.append("--async")
    cmd += ["--", *job_args]
    rc = sh(cmd)
    if wait or rc != 0:
        return rc
    return 0 if wait_for(batch_id) == "SUCCEEDED" else 1


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}


def wait_for(batch_id: str, poll_s: int = 15, timeout_s: int = 1800) -> str:
    """Poll until the batch reaches a terminal state."""
    waited = 0
    last = ""
    while waited < timeout_s:
        state = _sh_out(
            [
                GCLOUD, "dataproc", "batches", "describe", batch_id,
                f"--region={REGION}", f"--project={PROJECT}",
                "--format=value(state)",
            ]
        ).strip()
        if state != last:
            print(f"    {batch_id}: {state or '...'}")
            last = state
        if state in TERMINAL:
            return state
        time.sleep(poll_s)
        waited += poll_s
    return "TIMEOUT"


def _sh_out(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def describe(batch_id: str) -> dict:
    """Fetch what the service ACTUALLY applied, plus resource usage."""
    out = subprocess.run(
        [
            GCLOUD, "dataproc", "batches", "describe", batch_id,
            f"--region={REGION}", f"--project={PROJECT}", "--format=json",
        ],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return {}
    return json.loads(out.stdout or "{}")


def verify_applied(batch_id: str) -> tuple[bool, dict]:
    """Confirm the experiment's controls survived into the running job.

    gcloud will silently drop a malformed --properties string, leaving the job
    running on defaults. That produces numbers that look fine and mean nothing,
    so every run is checked rather than trusted.
    """
    info = describe(batch_id)
    props = (info.get("runtimeConfig") or {}).get("properties") or {}
    usage = (info.get("runtimeInfo") or {}).get("approximateUsage") or {}

    applied = {name: _prop(props, name) for name in REQUIRED_APPLIED}
    mismatches = {
        name: (applied[name] if applied[name] is not None else "<missing>")
        for name, want in REQUIRED_APPLIED.items()
        if str(applied[name]).lower() != want
    }

    captured = {
        "runtime_version": (info.get("runtimeConfig") or {}).get("version"),
        "milli_dcu_seconds": usage.get("milliDcuSeconds"),
        "shuffle_storage_gb_seconds": usage.get("shuffleStorageGbSeconds"),
        "executor_memory": _prop(props, "spark.executor.memory"),
        "driver_cores": _prop(props, "spark.driver.cores"),
        "lightning_engine": _prop(props, "spark.dataproc.lightningEngine.runtime"),
        **{k.replace(".", "_"): v for k, v in applied.items()},
    }

    print("  applied config:")
    for name in REQUIRED_APPLIED:
        want = REQUIRED_APPLIED[name]
        got = applied[name]
        mark = "ok " if str(got).lower() == want else "BAD"
        print(f"    [{mark}] {name:<44s} {got}")

    if mismatches:
        print(f"  !! properties NOT applied as requested: {mismatches}")
        return False, captured
    return True, captured


def generate(fact_rows: int) -> int:
    return submit(
        "generate_data.py",
        f"gen-{int(time.time())}",
        ["--data-root", DATA_ROOT, "--fact-rows", str(fact_rows)],
    )


def run(spec: dict, session: str) -> str | None:
    tag = "warm" if spec["warmup"] else f"b{spec.get('block', spec['rep'])}"
    run_id = f"{session}-{spec['workload']}-{spec['variant']}-{tag}"
    job_args = [
        "--workload", spec["workload"],
        "--variant", spec["variant"],
        "--run-id", run_id,
        "--rep", str(spec["rep"]),
        "--block", str(spec.get("block", 0)),
        "--data-root", DATA_ROOT,
        "--results-root", RESULTS_ROOT,
        "--eventlog-root", EVENTLOG_ROOT,
    ]
    if spec["warmup"]:
        job_args.append("--warmup")

    # Batch ids must be unique, lowercase, and <= 63 chars.
    batch_id = run_id.lower()[:63]
    t0 = time.time()
    rc = submit("run_one.py", batch_id, job_args, wait=False)
    batch_elapsed = round(time.time() - t0, 1)

    # Quota exhaustion is transient - the previous batch is still releasing
    # its vCPUs. Wait it out and retry once with a fresh id rather than losing
    # the cell.
    if rc != 0:
        print(f"  retrying {batch_id} in 90s (likely transient quota)")
        time.sleep(90)
        batch_id = f"{batch_id[:57]}-r2"
        t0 = time.time()
        rc = submit("run_one.py", batch_id, job_args, wait=False)
        batch_elapsed = round(time.time() - t0, 1)

    if rc != 0:
        return None

    ok, captured = verify_applied(batch_id)
    # Submission-to-completion, including provisioning and teardown. Reported
    # separately from wall_s so control-plane variance never contaminates the
    # pipeline measurement.
    captured.update(
        {"run_id": run_id, "batch_id": batch_id, "batch_elapsed_s": batch_elapsed, **spec}
    )
    _record_batch(batch_id, captured)
    if not ok:
        print("  !! run is NOT valid - experimental controls were not applied")
        return None
    dcu = captured.get("milli_dcu_seconds")
    if dcu:
        print(f"  compute: {int(dcu)/1000:.0f} DCU-seconds")
    return run_id


def _record_batch(batch_id: str, data: dict) -> None:
    """Persist the run's cloud-side facts to GCS, not the local disk.

    Object storage is the ledger. Each completed run writes its own immutable
    record, so an orchestrator dying partway through loses nothing: finished
    runs stay persisted, and the combined CSV is rebuilt from these files.
    """
    tmp = Path(config.REPO) / "results" / "batches"
    tmp.mkdir(parents=True, exist_ok=True)
    local = tmp / f"{batch_id}.json"
    local.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    sh([GCLOUD, "storage", "cp", str(local), f"{RESULTS_ROOT}/raw/{batch_id}.json"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true", help="push scripts to GCS")
    ap.add_argument("--generate", action="store_true", help="build the datasets")
    ap.add_argument(
        "--smoke", action="store_true", help="one run only, to validate the setup"
    )
    ap.add_argument(
        "--pair",
        action="store_true",
        help="baseline on uniform then skewed, nothing else. The staging gate: "
        "prove the same query runs on both datasets and that the skewed one "
        "actually produces straggler behaviour, before any optimization exists.",
    )
    ap.add_argument(
        "--once", action="store_true", help="run exactly one cell and stop"
    )
    ap.add_argument(
        "--recover", metavar="BATCH_ID",
        help="rebuild the record for a batch that succeeded but was not tracked",
    )
    ap.add_argument("--workload", choices=config.WORKLOADS, default="uniform")
    ap.add_argument("--variant", choices=config.VARIANTS, default="baseline")
    ap.add_argument("--block", type=int, help="block label for a single --once run")
    ap.add_argument("--fact-rows", type=int, default=config.FACT_ROWS)
    ap.add_argument("--reps", type=int, default=config.MEASURED_RUNS)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--from-block", type=int, default=1, help="resume an interrupted matrix"
    )
    ap.add_argument(
        "--to-block", type=int, help="stop after this block; run one block at a time"
    )
    ap.add_argument("--project", help="GCP project id (or $GCP_PROJECT)")
    ap.add_argument("--bucket", help="GCS bucket (or $GCS_BUCKET)")
    ap.add_argument("--region", help="default us-central1 (or $GCP_REGION)")
    args = ap.parse_args()

    _set_target(args.project, args.bucket, args.region)
    if not args.dry_run and (not PROJECT or not BUCKET):
        print(
            "Set the target first:\n"
            "  --project PROJECT_ID --bucket BUCKET_NAME\n"
            "or export GCP_PROJECT and GCS_BUCKET."
        )
        return 2

    if args.upload:
        return upload()

    if args.generate:
        print(f"Generating {args.fact_rows:,} fact rows into {DATA_ROOT}")
        return generate(args.fact_rows)

    if args.recover:
        # A batch that SUCCEEDED in the cloud is a valid observation even if
        # the local orchestrator died before recording it. Rebuild the record
        # from the batch itself rather than paying to run it again - but only
        # if every frozen control still verifies. Anything unverifiable is
        # reported, never silently filled in.
        print(f"Recovering {args.recover}")
        state = _sh_out(
            [
                GCLOUD, "dataproc", "batches", "describe", args.recover,
                f"--region={REGION}", f"--project={PROJECT}", "--format=value(state)",
            ]
        ).strip()
        print(f"  state: {state}")
        if state != "SUCCEEDED":
            print("  not SUCCEEDED - cannot recover, re-run this cell")
            return 1
        ok, captured = verify_applied(args.recover)
        captured["batch_id"] = args.recover
        captured["recovered"] = True
        _record_batch(args.recover, captured)
        if not ok:
            print("  !! controls did not verify - mark INCOMPLETE and re-run")
            return 1
        print("  recovered and verified")
        return 0

    if args.once:
        # --block matters when re-running a single cell lost to an interrupted
        # matrix: without it the run is labelled b1 and collides with an
        # existing observation instead of filling the gap.
        spec = {
            "workload": args.workload,
            "variant": args.variant,
            "rep": args.block or 1,
            "block": args.block or 1,
            "warmup": False,
        }
        print(f"=== single execution: {args.workload} {args.variant} ===")
        return 0 if run(spec, time.strftime("%m%d-%H%M%S")) else 1

    if args.pair:
        session = time.strftime("%m%d-%H%M%S")
        ids = []
        for workload in config.WORKLOADS:  # uniform, then skewed
            spec = {
                "workload": workload,
                "variant": "baseline",
                "rep": 1,
                "warmup": False,
            }
            print(f"\n=== {workload} baseline ===")
            rid = run(spec, session)
            if not rid:
                print(f"  !! {workload} baseline failed")
                return 1
            ids.append(rid)
            time.sleep(SUBMIT_PAUSE_S)
        print("\nBoth baseline runs complete:")
        for i in ids:
            print(f"  {i}")
        return 0

    if args.smoke:
        session = time.strftime("%m%d-%H%M%S")
        spec = {"workload": "skewed", "variant": "baseline", "rep": 1, "warmup": False}
        print("Smoke test: one skewed baseline run")
        return 0 if run(spec, session) else 1

    plan = config.run_order(args.reps)

    # Resume support. A long matrix can be interrupted (we lost one to local
    # memory pressure); re-running completed blocks would waste money and, more
    # importantly, mix runs from two sessions into one cell.
    if args.from_block > 1 or args.to_block:
        hi = args.to_block or args.reps
        plan = [
            s
            for s in plan
            if not s["warmup"] and args.from_block <= s["block"] <= hi
        ]
        print(f"Blocks {args.from_block}-{hi}: {len(plan)} runs "
              f"(warm-ups skipped, already done)\n")
    print(f"{len(plan)} batches to submit ({args.reps * 4} measured)\n")
    if args.dry_run:
        for i, s in enumerate(plan, 1):
            tag = "warm" if s["warmup"] else f"rep{s['rep']}"
            print(f"  {i:>3}. {s['workload']:<8s} {s['variant']:<9s} {tag}")
        return 0

    session = time.strftime("%m%d-%H%M%S")
    started = time.time()
    ok, failed = [], []
    for i, spec in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {spec['workload']} {spec['variant']}")
        rid = run(spec, session)
        (ok if rid else failed).append(rid or f"{spec}")
        if i < len(plan):
            time.sleep(SUBMIT_PAUSE_S)

    print(f"\nSubmitted {len(ok)} ok, {len(failed)} failed "
          f"in {(time.time()-started)/60:.1f} min")
    print(f"Results:    {RESULTS_ROOT}/runs/")
    print(f"Event logs: {EVENTLOG_ROOT}/")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
