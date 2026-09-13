# Data Pipeline Optimization in Spark

Companion repository for a controlled Apache Spark data-pipeline
optimization experiment.

The repository contains:

- synthetic data generation code
- Spark benchmark implementation
- Google Cloud batch submission tooling
- Spark event-log analysis
- the exact benchmark configuration
- raw and measured experiment results

## Article

The methodology, results, interpretation, and conclusions are discussed
in the accompanying DZone article.

[Article link will be added after publication.]

## Repository Structure

- `config/` — human-readable snapshot of the frozen configuration, recorded
  from the executed runs. Nothing reads these files at runtime; the executable
  values live in `scripts/config.py` and in `SPARK_PROPERTIES` inside
  `scripts/submit_cloud.py`.
- `scripts/` — experiment and analysis code
- `results/` — experiment output used for the article

## Running

Set:

```bash
export GCP_PROJECT="<project>"
export GCS_BUCKET="<bucket>"
export GCP_REGION="us-central1"
```

Upload the experiment scripts to the bucket — the batches are submitted from
there, so this must come first on a fresh bucket:

```bash
python scripts/submit_cloud.py --upload
```

Generate the datasets:

```bash
python scripts/submit_cloud.py --generate --fact-rows 20000000
```

Submit the benchmark matrix:

```bash
python scripts/submit_cloud.py
```

Analyze results:

```bash
python scripts/analyze.py --bucket $GCS_BUCKET
python scripts/report.py
```

Requires a GCP project with billing enabled and the Dataproc API on.
`submit_cloud.py` uses only the Python standard library plus the `gcloud`
CLI; `analyze.py` and `report.py` need the packages in `requirements.txt`.

## Results

- `results/benchmark_raw.csv` — all 24 runs, including the 4 discarded
  warm-ups (`warmup=true`)
- `results/benchmark_measured.csv` — the 20 measured runs used for the
  article, 5 per condition across 4 conditions

Measurements were produced on Dataproc Serverless runtime 2.2.86
(Spark 3.5.3). The applied configuration is in `config/`.
