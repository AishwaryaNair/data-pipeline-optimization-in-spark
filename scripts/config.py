"""Single source of truth for every experiment parameter.

Nothing else in this project hard-codes a size, seed, or path. Every number
the article reports comes from here or from the measurements in results/.

The published measurements were produced on Dataproc Serverless (runtime
2.2.86, Spark 3.5.3). This file carries the cloud configuration only.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- Where data lives -------------------------------------------------------
# Roots are plain strings, not Path objects, because they are gs:// URIs
# rather than filesystem paths. Each entry point calls set_roots() after
# parsing its arguments; there is no default, so a run cannot silently read
# from somewhere other than the bucket it was told to use.
DATA_ROOT = ""
EVENTLOG_ROOT = ""
RESULTS_ROOT = str(REPO / "results")


def set_roots(data_root=None, results_root=None, eventlog_root=None) -> None:
    """Point the experiment at a storage location."""
    global DATA_ROOT, RESULTS_ROOT, EVENTLOG_ROOT
    if data_root:
        DATA_ROOT = data_root.rstrip("/")
    if results_root:
        RESULTS_ROOT = results_root.rstrip("/")
    if eventlog_root:
        EVENTLOG_ROOT = eventlog_root.rstrip("/")

# --- Dataset shape ----------------------------------------------------------
# Starting point. The real target is RUNTIME, not row count: a baseline run
# between 30s and 120s is long enough for stage/task differences to show and
# short enough to repeat 20+ times. If the baseline lands outside that band,
# change FACT_ROWS, re-run, then FREEZE it - changing dataset size mid-
# experiment invalidates every measurement taken before the change.
FACT_ROWS = 20_000_000
DIM_ROWS = 100_000

# Both workloads carry an identical row count, identical schema, and the same
# number of distinct join keys. Only the *distribution* of customer_id differs.
WORKLOADS = ("uniform", "skewed")

# --- Skew model -------------------------------------------------------------
# One hot customer absorbs SKEW_FRACTION of all sales. Deliberately severe, so
# the effect is unambiguous rather than arguable.
#
#   uniform : customer_id = (sale_id % 100_000) + 1  -> ~50 sales per customer
#   skewed  : customer 42 gets 40% = 2,000,000 sales
#             the other 60% spread over the remaining customers (~30 each)
HOT_CUSTOMER_ID = 42
SKEW_FRACTION = 0.40

# --- Optimization -----------------------------------------------------------
# Salt bucket count. More buckets spread the hot key further but replicate the
# dimension proportionally - that trade-off is part of the article's argument.
SALT_BUCKETS = 8

VARIANTS = ("baseline", "salted")

# --- Queries ----------------------------------------------------------------
# `simple` isolates join behaviour: scan -> join -> aggregate, no filters. Used
# to confirm skew actually shows up before any measurements are collected. A
# filter that accidentally removes most of the hot key's rows is an easy way to
# spend hours wondering where the skew went.
#
# `full` adds the production-like filter and is what the article reports.
QUERIES = ("simple", "full")
DEFAULT_QUERY = "full"

# Filter values for the `full` query. Chosen to keep ~80% of rows, so the hot
# key survives the filter.
STATUS_FILTER = "COMPLETE"
DATE_FROM = "2026-01-01"
DATE_TO = "2026-06-30"

# --- Column distributions ---------------------------------------------------
# Identical across both workloads. If these differed the datasets would not be
# comparable and any runtime difference could be caused by them rather than by
# skew. generate_data.py verifies this rather than assuming it.
STATUS_WEIGHTS = {"COMPLETE": 0.80, "PENDING": 0.10, "CANCELLED": 0.10}
AMOUNT_MIN, AMOUNT_MAX = 10.0, 500.0
QUANTITY_MIN, QUANTITY_MAX = 1, 5
DATE_START = "2026-01-01"
DATE_SPAN_DAYS = 181  # Jan 1 - Jun 30 2026

REGIONS = ("East", "West", "North", "South")
SEGMENTS = ("Consumer", "SMB", "Enterprise")
TIERS = ("Bronze", "Silver", "Gold", "Platinum")

# --- Spark ------------------------------------------------------------------
# Master, executor count and executor memory are owned by Dataproc Serverless
# and supplied per batch; see config/spark.properties for what was applied.
SHUFFLE_PARTITIONS = 200

# Two settings are deliberately overridden, and the article must say so
# explicitly. Neither is a production recommendation - both exist to hold the
# execution plan constant so the only thing varying is the data distribution.
#
#   autoBroadcastJoinThreshold = -1
#       Otherwise Spark sees a 100K-row dimension, broadcasts it, and the join
#       never shuffles - which removes the very skew being studied.
#
#   adaptive.enabled = false
#       Otherwise AQE detects runtime conditions and rewrites the plan (
#       coalescing partitions, splitting skewed ones) mid-flight, which is a
#       second uncontrolled variable on top of the one under test.
DISABLE_BROADCAST_JOIN = True
DISABLE_AQE = True

# --- Protocol ---------------------------------------------------------------
WARMUP_RUNS = 1
MEASURED_RUNS = 5

# --- Determinism ------------------------------------------------------------
# Same seed + same partition count reproduces the data byte-for-byte, which is
# why 500MB of Parquet never needs to enter git.
SEEDS = {
    "skew_selector": 17,
    "amount": 31,
    "quantity": 33,
    "status": 35,
    "event_date": 37,
    "dim_attrs": 41,
}


# Counterbalanced block design. Each block runs all four cells once; the order
# rotates between blocks so no treatment systematically occupies the first or
# last position, where startup effects and end-of-session drift live. Within a
# block, a workload's baseline and salted runs stay close together, which is
# what makes the paired per-block comparison meaningful.
UB, US = ("uniform", "baseline"), ("uniform", "salted")
SB, SS = ("skewed", "baseline"), ("skewed", "salted")

BLOCK_ORDERS = [
    [UB, US, SB, SS],
    [SS, SB, US, UB],
    [US, UB, SS, SB],
    [SB, SS, UB, US],
    [UB, US, SB, SS],
]


def run_order(reps: int = MEASURED_RUNS) -> list[dict]:
    """The execution order, shared by the local and cloud runners.

    One discarded warm-up per cell, then `reps` counterbalanced blocks.

    Lives in config (which has no third-party imports) so the cloud submitter
    can use it without pulling in numpy.
    """
    plan = [
        {"workload": w, "variant": v, "rep": 0, "block": 0, "warmup": True}
        for w in WORKLOADS
        for v in VARIANTS
    ]
    for block in range(1, reps + 1):
        order = BLOCK_ORDERS[(block - 1) % len(BLOCK_ORDERS)]
        for workload, variant in order:
            plan.append(
                {
                    "workload": workload,
                    "variant": variant,
                    "rep": block,
                    "block": block,
                    "warmup": False,
                }
            )
    return plan


def sales_path(workload: str) -> str:
    assert workload in WORKLOADS, workload
    return f"{DATA_ROOT}/sales_{workload}"


def customers_path() -> str:
    return f"{DATA_ROOT}/customers"
