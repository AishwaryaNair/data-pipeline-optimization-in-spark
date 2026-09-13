"""Generate the `customers` dimension and the two `sales` fact tables.

Submitted as a Dataproc batch by submit_cloud.py --generate.

Produces, under the --data-root given:
    customers/       100K rows
    sales_uniform/   20M rows, ~200 sales per customer
    sales_skewed/    20M rows, customer 42 holds 40% (~8M rows)

Both fact tables have identical row counts, identical schemas, the same number
of distinct join keys, and statistically identical values in every column
except `customer_id`. That last property is what makes the comparison valid,
so the script verifies it rather than assuming it.

Everything is generated in Spark from `spark.range`, never collected to the
driver - generating 20 million rows in Python and calling createDataFrame
would be far slower and would not scale.
"""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config
import spark_session


def _pick(col, values):
    """Map a 0..1 random column onto a list of values, evenly."""
    idx = (col * len(values)).cast("int")
    expr = F.lit(values[-1])
    for i in reversed(range(len(values) - 1)):
        expr = F.when(idx == i, F.lit(values[i])).otherwise(expr)
    return expr


def build_customers(spark) -> DataFrame:
    """Dimension: customer_id, region, segment, customer_tier.

    customer_id runs 1..DIM_ROWS to match the (sale_id % DIM_ROWS) + 1 key
    used on the fact side.
    """
    df = spark.range(1, config.DIM_ROWS + 1).withColumnRenamed("id", "customer_id")
    r = F.rand(config.SEEDS["dim_attrs"])
    return (
        df.withColumn("region", _pick(r, config.REGIONS))
        .withColumn("segment", _pick(F.rand(config.SEEDS["dim_attrs"] + 1), config.SEGMENTS))
        .withColumn("customer_tier", _pick(F.rand(config.SEEDS["dim_attrs"] + 2), config.TIERS))
    )


def build_sales(spark, workload: str) -> DataFrame:
    """Fact table. Identical across workloads except for `customer_id`."""
    df = spark.range(0, config.FACT_ROWS).withColumnRenamed("id", "sale_id")

    # Even spread: 20,000,000 / 100,000 = exactly 200 sales per customer.
    even_key = (F.col("sale_id") % config.DIM_ROWS) + 1

    if workload == "uniform":
        customer_id = even_key
    elif workload == "skewed":
        # 40% of rows to the hot customer; the rest keep the even spread.
        # The hot customer also picks up its ~120 rows from the even spread,
        # which is immaterial against ~8,000,000.
        customer_id = F.when(
            F.rand(config.SEEDS["skew_selector"]) < config.SKEW_FRACTION,
            F.lit(config.HOT_CUSTOMER_ID),
        ).otherwise(even_key)
    else:
        raise ValueError(workload)

    status_r = F.rand(config.SEEDS["status"])
    complete, pending = (
        config.STATUS_WEIGHTS["COMPLETE"],
        config.STATUS_WEIGHTS["COMPLETE"] + config.STATUS_WEIGHTS["PENDING"],
    )

    amount_span = config.AMOUNT_MAX - config.AMOUNT_MIN
    qty_span = config.QUANTITY_MAX - config.QUANTITY_MIN + 1

    return (
        df.withColumn("customer_id", customer_id.cast("bigint"))
        .withColumn(
            "event_date",
            F.date_add(
                F.lit(config.DATE_START).cast("date"),
                (F.rand(config.SEEDS["event_date"]) * config.DATE_SPAN_DAYS).cast("int"),
            ),
        )
        .withColumn(
            "amount",
            (F.lit(config.AMOUNT_MIN) + F.rand(config.SEEDS["amount"]) * amount_span)
            .cast("decimal(12,2)"),
        )
        .withColumn(
            "quantity",
            (F.lit(config.QUANTITY_MIN) + F.rand(config.SEEDS["quantity"]) * qty_span)
            .cast("int"),
        )
        .withColumn(
            "status",
            F.when(status_r < complete, F.lit("COMPLETE"))
            .when(status_r < pending, F.lit("PENDING"))
            .otherwise(F.lit("CANCELLED")),
        )
        .select(
            "sale_id", "customer_id", "event_date", "amount", "quantity", "status"
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        required=True,
        help="gs:// URI to write the datasets under. No default.",
    )
    ap.add_argument("--fact-rows", type=int, default=config.FACT_ROWS)
    args = ap.parse_args()

    config.set_roots(data_root=args.data_root)
    config.FACT_ROWS = args.fact_rows

    spark = spark_session.build("generate-data")
    print(f"data root : {config.DATA_ROOT}")
    print(f"fact rows : {config.FACT_ROWS:,}\n")

    # mode("overwrite") replaces any previous output, on local disk or on
    # object storage alike - no filesystem-specific cleanup needed.
    build_customers(spark).write.mode("overwrite").parquet(config.customers_path())
    print(f"customers    {config.DIM_ROWS:>12,} rows")

    for workload in config.WORKLOADS:
        build_sales(spark, workload).write.mode("overwrite").parquet(
            config.sales_path(workload)
        )
        print(f"sales_{workload:<7s}{config.FACT_ROWS:>12,} rows")

    # --- verification -------------------------------------------------------
    print("\nVerification")
    failures = []
    frames = {w: spark.read.parquet(str(config.sales_path(w))) for w in config.WORKLOADS}

    stats = {}
    for workload, df in frames.items():
        top = (
            df.groupBy("customer_id").count().orderBy(F.desc("count")).limit(1).collect()
        )[0]
        agg = df.agg(
            F.count(F.lit(1)).alias("rows"),
            F.countDistinct("customer_id").alias("keys"),
            F.avg("amount").alias("avg_amount"),
            F.avg("quantity").alias("avg_qty"),
            F.avg(F.when(F.col("status") == "COMPLETE", 1.0).otherwise(0.0)).alias("pct_complete"),
        ).collect()[0]
        stats[workload] = agg
        print(
            f"  {workload:<8s} rows={agg['rows']:,}  keys={agg['keys']:,}  "
            f"top_customer={top['customer_id']} holds {top['count']:,} "
            f"({top['count']/agg['rows']:.1%})"
        )

    u, s = stats["uniform"], stats["skewed"]

    if u["rows"] != s["rows"]:
        failures.append(f"row counts differ: {u['rows']} vs {s['rows']}")
    else:
        print(f"  [ok]   both workloads carry {u['rows']:,} rows")

    # A small gap is expected and harmless: the hot key absorbs SKEW_FRACTION
    # of the rows, so fewer remain to cover the key space and some keys land
    # with zero rows. That effect vanishes as FACT_ROWS grows (at 20M there are
    # ~120 rows per key even after the hot key takes 40%). A LARGE gap means
    # the two workloads are no longer joining against comparable key spaces.
    key_gap = abs(u["keys"] - s["keys"]) / u["keys"]
    if key_gap > 0.01:
        failures.append(
            f"distinct key counts differ by {key_gap:.1%}: "
            f"{u['keys']:,} vs {s['keys']:,}. With this few rows the hot key "
            f"starves the long tail - raise FACT_ROWS."
        )
    else:
        print(
            f"  [ok]   distinct customer_ids match within {key_gap:.2%} "
            f"({u['keys']:,} vs {s['keys']:,})"
        )

    # Everything except customer_id must match, or a runtime difference could
    # be caused by the data rather than by the skew.
    for label, a, b, tol in [
        ("avg amount", float(u["avg_amount"]), float(s["avg_amount"]), 0.01),
        ("avg quantity", float(u["avg_qty"]), float(s["avg_qty"]), 0.01),
        ("pct COMPLETE", float(u["pct_complete"]), float(s["pct_complete"]), 0.005),
    ]:
        if abs(a - b) / max(abs(a), 1e-9) > tol:
            failures.append(f"{label} differs between workloads: {a:.4f} vs {b:.4f}")
        else:
            print(f"  [ok]   {label} matches ({a:.3f} vs {s and b:.3f})")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        spark.stop()
        return 1

    print("\nData generation complete.")
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
