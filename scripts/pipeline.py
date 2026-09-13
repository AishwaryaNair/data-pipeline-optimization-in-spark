"""The pipeline under test, in two implementations.

Shape (identical for both variants):

    scan sales -> [filter] -> join customers -> shuffle -> aggregate -> sink

The ONLY difference between `baseline` and `salted` is how the join is
expressed. Same input, same filter, same aggregation, same sink.

Two query shapes:
    simple - scan, join, aggregate. No filters. Used to confirm skew shows up
             before collecting measurements: a filter that accidentally
             removes the hot key's rows is an easy way to lose hours.
    full   - adds the production-like status and date filter. What the article
             reports.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

import config


def _read(spark, workload: str) -> tuple[DataFrame, DataFrame]:
    sales = spark.read.parquet(str(config.sales_path(workload)))
    customers = spark.read.parquet(str(config.customers_path()))
    return sales, customers


def _filter(sales: DataFrame, query: str) -> DataFrame:
    """Production-like predicate, or none at all for the `simple` query.

    The filter keeps ~80% of rows and does not target customer_id, so the hot
    key survives it in proportion.
    """
    if query == "simple":
        return sales
    return sales.filter(
        (F.col("status") == config.STATUS_FILTER)
        & F.col("event_date").between(
            F.lit(config.DATE_FROM).cast("date"),
            F.lit(config.DATE_TO).cast("date"),
        )
    )


def _aggregate(joined: DataFrame) -> DataFrame:
    """Rollup by region and segment - both from the dimension side."""
    return joined.groupBy("region", "segment").agg(
        F.count(F.lit(1)).alias("transaction_count"),
        F.sum("amount").alias("total_sales"),
        F.avg("amount").alias("avg_transaction_value"),
        F.sum("quantity").alias("total_quantity"),
    )


def baseline(spark, workload: str, query: str) -> DataFrame:
    """Ordinary join. What an engineer writes before anyone mentions skew."""
    sales, customers = _read(spark, workload)
    joined = _filter(sales, query).join(customers, "customer_id")
    return _aggregate(joined)


def salted(spark, workload: str, query: str) -> DataFrame:
    """Selective key salting - the hot key only.

    The hot customer's rows are spread across SALT_BUCKETS partitions instead
    of all landing in one. Every other customer keeps salt 0 and is untouched.

    Selective, not blanket. Salting every key would mean replicating all
    100,000 dimension rows 8 times (800,000 rows) to keep the join correct -
    work that has nothing to do with the skew being fixed, and which would make
    the optimization look worse than it is. Here the dimension grows by 7 rows:

        customer 42          -> 8 rows, salt 0..7
        all other customers  -> 1 row,  salt 0
        100,000 rows         -> 100,007 rows

    Fact-side salt is derived from hash(sale_id) rather than rand(): rand is
    only reproducible for a fixed partitioning, while a hash gives every row
    the same bucket on every run regardless of how Spark splits the input.
    Identical work across repetitions is the whole point of the protocol.
    """
    sales, customers = _read(spark, workload)
    filtered = _filter(sales, query)

    hot = F.lit(config.HOT_CUSTOMER_ID)
    buckets = F.lit(config.SALT_BUCKETS)

    sales_salted = filtered.withColumn(
        "salt",
        F.when(
            F.col("customer_id") == hot, F.pmod(F.hash(F.col("sale_id")), buckets)
        ).otherwise(F.lit(0)),
    )

    customers_salted = customers.withColumn(
        "salt",
        F.when(
            F.col("customer_id") == hot,
            F.array(*[F.lit(i) for i in range(config.SALT_BUCKETS)]),
        ).otherwise(F.array(F.lit(0))),
    ).withColumn("salt", F.explode(F.col("salt")))

    joined = sales_salted.join(customers_salted, ["customer_id", "salt"])
    return _aggregate(joined)


VARIANT_FNS = {"baseline": baseline, "salted": salted}


def build(spark, workload: str, variant: str, query: str = config.DEFAULT_QUERY) -> DataFrame:
    return VARIANT_FNS[variant](spark, workload, query)
