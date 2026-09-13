"""Spark session construction, identical for every run except the event-log path.

Keeping this in one place is what makes "same Spark configuration for
comparable runs" an enforced property rather than an intention.
"""

from pathlib import Path

import config


def build(app_name: str, eventlog_dir: str | None = None):
    """Create a SparkSession with the experiment's configuration.

    Master, executor count and executor memory are owned by Dataproc
    Serverless and supplied per batch, so they are not set here.

    Two settings are deliberately non-default, and the article states both
    explicitly. Neither is a production recommendation; both exist to hold the
    execution plan constant so the only thing varying between runs is the
    distribution of the join key.
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", config.SHUFFLE_PARTITIONS)
        .config("spark.ui.showConsoleProgress", "false")
    )

    if config.DISABLE_BROADCAST_JOIN:
        # Without this, Spark sees a 100K-row dimension, broadcasts it, and the
        # join never shuffles - removing the very skew being studied.
        builder = builder.config("spark.sql.autoBroadcastJoinThreshold", -1)

    if config.DISABLE_AQE:
        # Without this, AQE rewrites the plan at runtime - coalescing
        # partitions and splitting skewed ones - which is a second
        # uncontrolled variable on top of the one under test.
        builder = builder.config("spark.sql.adaptive.enabled", "false")

    if eventlog_dir:
        if not eventlog_dir.startswith("gs://"):
            p = Path(eventlog_dir)
            p.mkdir(parents=True, exist_ok=True)
            eventlog_dir = p.resolve().as_uri()
        builder = (
            builder.config("spark.eventLog.enabled", "true")
            .config("spark.eventLog.dir", eventlog_dir)
            # Spark 4 compresses event logs with zstd by default. Off keeps
            # them readable with no extra dependency; they are only a few MB.
            .config("spark.eventLog.compress", "false")
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
