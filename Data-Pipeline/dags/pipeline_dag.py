from dotenv import load_dotenv
load_dotenv()

"""
pipeline_dag.py
---------------
CourseWeave AI — Master Airflow DAG

Pipeline Flow:
    acquire → preprocess → validate → detect_anomalies
    → bias_detection → dvc_version → load_db → pipeline_report

GCS Bucket : courseweave-ai-data
DVC Remote : gs://courseweave-ai-data/dvc-storage
Alerts     : Slack webhook (SLACK_WEBHOOK_URL in .env)

Fix Notes:
- Uses local temp JSON files instead of XCom for large DataFrames
- Avoids Airflow 3.x XCom 404 API error
- Fairlearn bias detection added
- timezone.utc used instead of deprecated utcnow()
"""

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# ─────────────────────────────────────────
# Temp file paths for inter-task data passing
# ─────────────────────────────────────────

BASE_DIR            = Path(__file__).parent.parent
TEMP_DIR            = BASE_DIR / "data" / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

RAW_COURSES_PATH     = TEMP_DIR / "raw_courses.json"
RAW_PREREQS_PATH     = TEMP_DIR / "raw_prerequisites.json"
CLEAN_COURSES_PATH   = TEMP_DIR / "cleaned_courses.json"
CLEAN_PREREQS_PATH   = TEMP_DIR / "cleaned_prerequisites.json"
STATS_REPORT_PATH    = BASE_DIR / "data" / "processed" / "stats_report.json"
BIAS_REPORT_PATH     = BASE_DIR / "data" / "processed" / "bias_report.json"
PIPELINE_REPORT_PATH = BASE_DIR / "data" / "processed" / "pipeline_report.json"

# ─────────────────────────────────────────
# Default DAG arguments
# ─────────────────────────────────────────

default_args = {
    "owner":            "courseweave-team",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 1, 28),
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

# ─────────────────────────────────────────
# Slack alert helper
# ─────────────────────────────────────────

def send_slack_alert(message: str, is_error: bool = False) -> None:
    """Send a message to Slack via webhook."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        logging.warning("SLACK_WEBHOOK_URL not set — skipping Slack alert.")
        return

    icon    = "🚨" if is_error else "✅"
    payload = {"text": f"{icon} *CourseWeave Pipeline*\n{message}"}

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Slack alert sent successfully.")
    except Exception as e:
        logging.error("Failed to send Slack alert: %s", e)


# ─────────────────────────────────────────
# Task 1: Acquire Data
# ─────────────────────────────────────────

def task_acquire_data(**kwargs) -> None:
    """
    Pull raw data from GCS bucket into local data/raw/ directory.
    Saves DataFrames to temp JSON files for next task.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))

    from google.cloud import storage
    from acquire_data import acquire_data

    bucket_name = "courseweave-ai-data"
    local_raw   = BASE_DIR / "data" / "raw"
    local_raw.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    files_to_download = ["raw/courses.csv", "raw/prerequisites.csv"]
    for gcs_path in files_to_download:
        blob       = bucket.blob(gcs_path)
        file_name  = Path(gcs_path).name
        local_path = local_raw / file_name

        if blob.exists():
            blob.download_to_filename(str(local_path))
            logging.info("Downloaded %s from GCS to %s", gcs_path, local_path)
        else:
            logging.warning("File not found in GCS: %s", gcs_path)

    raw = acquire_data()
    raw["courses"].to_json(RAW_COURSES_PATH)
    raw["prerequisites"].to_json(RAW_PREREQS_PATH)

    logging.info(
        "Acquisition complete — %d courses, %d prerequisites.",
        len(raw["courses"]), len(raw["prerequisites"])
    )
    send_slack_alert(
        f"Task 1 ✅ *Acquire* complete — "
        f"{len(raw['courses'])} courses, {len(raw['prerequisites'])} prerequisites."
    )


# ─────────────────────────────────────────
# Task 2: Preprocess Data
# ─────────────────────────────────────────

def task_preprocess_data(**kwargs) -> None:
    """Clean and normalize raw DataFrames."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from preprocess_data import preprocess_data

    raw = {
        "courses":       pd.read_json(RAW_COURSES_PATH),
        "prerequisites": pd.read_json(RAW_PREREQS_PATH),
    }

    cleaned = preprocess_data(raw)

    cleaned["courses"].to_json(CLEAN_COURSES_PATH)
    cleaned["prerequisites"].to_json(CLEAN_PREREQS_PATH)

    logging.info(
        "Preprocessing complete — %d courses, %d prerequisites.",
        len(cleaned["courses"]), len(cleaned["prerequisites"])
    )
    send_slack_alert(
        f"Task 2 ✅ *Preprocess* complete — "
        f"{len(cleaned['courses'])} courses after cleaning."
    )


# ─────────────────────────────────────────
# Task 3: Validate Data + Generate Stats
# ─────────────────────────────────────────

def task_validate_data(**kwargs) -> None:
    """
    Run schema validation + generate statistics report.
    Saves stats_report.json locally and uploads to GCS.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from validate_data import validate_data
    from google.cloud import storage

    cleaned = {
        "courses":       pd.read_json(CLEAN_COURSES_PATH),
        "prerequisites": pd.read_json(CLEAN_PREREQS_PATH),
    }

    try:
        stats = validate_data(cleaned)
    except ValueError as e:
        send_slack_alert(f"Task 3 🚨 *Validation FAILED*\n{str(e)}", is_error=True)
        raise

    client = storage.Client()
    bucket = client.bucket("courseweave-ai-data")
    blob   = bucket.blob("processed/stats_report.json")
    blob.upload_from_filename(str(STATS_REPORT_PATH))
    logging.info("Stats report uploaded to GCS processed/stats_report.json")

    send_slack_alert(
        f"Task 3 ✅ *Validate* complete — "
        f"{stats['courses']['total_courses']} courses validated. "
        f"Stats saved to GCS."
    )


# ─────────────────────────────────────────
# Task 4: Anomaly Detection + Slack Alerts
# ─────────────────────────────────────────

def task_detect_anomalies(**kwargs) -> None:
    """
    Run anomaly detection against PostgreSQL.
    Sends Slack alert if hard anomalies found.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from detect_anomalies import detect_anomalies

    results = detect_anomalies()

    total_anomalies = (
        len(results["missing_prerequisites"]) +
        len(results["circular_prerequisites"])
    )

    if total_anomalies > 0:
        msg = (
            f"Task 4 🚨 *Anomaly Detection* — {total_anomalies} hard anomalies found!\n"
            f"• Missing prerequisites: {len(results['missing_prerequisites'])}\n"
            f"• Circular prerequisites: {len(results['circular_prerequisites'])}\n"
            f"Check Airflow logs for details."
        )
        send_slack_alert(msg, is_error=True)
        raise ValueError(f"Anomaly detection failed — {total_anomalies} anomalies found.")
    else:
        send_slack_alert("Task 4 ✅ *Anomaly Detection* complete — no anomalies found.")


# ─────────────────────────────────────────
# Task 5: Bias Detection
# ─────────────────────────────────────────

def task_bias_detection(**kwargs) -> None:
    """
    Detect data bias by slicing across:
    1. Program coverage — are all programs equally represented?
    2. Credit distribution — are credits fairly distributed across programs?
    3. Fairlearn MetricFrame — course coverage fairness analysis
    """
    from google.cloud import storage

    courses = pd.read_json(CLEAN_COURSES_PATH)

    bias_report = {
        "generated_at":        datetime.now(timezone.utc).isoformat(),
        "program_coverage":    {},
        "credit_distribution": {},
        "bias_flags":          [],
    }

    # ── 1. Program Coverage ──────────────────────────────
    total_courses     = len(courses)
    program_counts    = courses["program_code"].value_counts().to_dict()
    expected_programs = {"MS_DAE", "MS_DS", "MS_CS", "MS_DA", "MS_IS"}

    for program in expected_programs:
        count      = program_counts.get(program, 0)
        percentage = round((count / total_courses) * 100, 2) if total_courses else 0
        bias_report["program_coverage"][program] = {
            "count":      count,
            "percentage": percentage,
        }

        if percentage < 10.0:
            flag = f"LOW COVERAGE: {program} has only {percentage}% of courses ({count} courses)"
            bias_report["bias_flags"].append(flag)
            logging.warning("BIAS FLAG: %s", flag)

        if count == 0:
            flag = f"MISSING PROGRAM: {program} has NO courses in dataset"
            bias_report["bias_flags"].append(flag)
            logging.error("BIAS FLAG: %s", flag)

    # ── 2. Credit Distribution ───────────────────────────
    for program, group in courses.groupby("program_code"):
        avg_credits = round(float(group["credits"].mean()), 2)
        bias_report["credit_distribution"][program] = {
            "avg_credits": avg_credits,
            "min_credits": int(group["credits"].min()),
            "max_credits": int(group["credits"].max()),
        }

    avg_values = [v["avg_credits"] for v in bias_report["credit_distribution"].values()]
    if avg_values and (max(avg_values) - min(avg_values)) > 1.0:
        flag = (
            f"CREDIT IMBALANCE: Credit avg ranges from "
            f"{min(avg_values)} to {max(avg_values)} across programs"
        )
        bias_report["bias_flags"].append(flag)
        logging.warning("BIAS FLAG: %s", flag)

    # ── 3. Fairlearn Analysis ────────────────────────────
    try:
        from fairlearn.metrics import MetricFrame

        courses["is_core"] = (courses["course_type"] == "Core").astype(int)

        metric_frame = MetricFrame(
            metrics={"course_count": lambda y_true, y_pred: len(y_pred)},
            y_true=courses["is_core"],
            y_pred=courses["is_core"],
            sensitive_features=courses["program_code"]
        )

        fairlearn_results = metric_frame.by_group.to_dict()
        bias_report["fairlearn_by_group"] = fairlearn_results
        logging.info("Fairlearn coverage by program: %s", fairlearn_results)

    except Exception as e:
        logging.warning("Fairlearn analysis skipped: %s", e)

    # ── Save bias report locally and upload to GCS ───────
    BIAS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BIAS_REPORT_PATH, "w") as f:
        json.dump(bias_report, f, indent=2)

    client = storage.Client()
    bucket = client.bucket("courseweave-ai-data")
    blob   = bucket.blob("processed/bias_report.json")
    blob.upload_from_filename(str(BIAS_REPORT_PATH))
    logging.info("Bias report uploaded to GCS processed/bias_report.json")

    if bias_report["bias_flags"]:
        flags_text = "\n".join(f"• {f}" for f in bias_report["bias_flags"])
        send_slack_alert(
            f"Task 5 ⚠️ *Bias Detection* — {len(bias_report['bias_flags'])} bias flags found!\n"
            f"{flags_text}",
            is_error=True
        )
    else:
        send_slack_alert(
            "Task 5 ✅ *Bias Detection* complete — "
            "no significant bias found across programs."
        )

    logging.info("Bias detection complete — %d flags.", len(bias_report["bias_flags"]))


# ─────────────────────────────────────────
# Task 6: DVC Versioning
# ─────────────────────────────────────────

def task_dvc_versioning(**kwargs) -> None:
    """
    Version control the validated data using DVC.
    Pushes data to GCS DVC remote: gs://courseweave-ai-data/dvc-storage
    """
    repo_root  = BASE_DIR
    data_paths = [
        "data/raw/courses.csv",
        "data/raw/prerequisites.csv",
        "data/processed/stats_report.json",
        "data/processed/bias_report.json",
    ]

    try:
        for data_path in data_paths:
            full_path = repo_root / data_path
            if full_path.exists():
                result = subprocess.run(
                    ["/opt/anaconda3/bin/dvc", "add", data_path],
                    cwd=str(repo_root),
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logging.warning("DVC add warning for %s: %s", data_path, result.stderr)
                else:
                    logging.info("DVC tracked: %s", data_path)

        push_result = subprocess.run(
            ["/opt/anaconda3/bin/dvc", "push"],
            cwd=str(repo_root),
            capture_output=True,
            text=True
        )
        if push_result.returncode != 0:
            raise RuntimeError(f"DVC push failed: {push_result.stderr}")

        logging.info("DVC push complete: %s", push_result.stdout)

        run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        git_commands = [
            ["git", "add", "*.dvc", ".dvcignore"],
            ["git", "commit", "-m", f"chore: DVC data versioning — pipeline run {run_date}"],
        ]
        for cmd in git_commands:
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True
            )
            if result.returncode not in (0, 1):
                logging.warning("Git command output: %s", result.stderr)

        send_slack_alert("Task 6 ✅ *DVC Versioning* complete — data pushed to GCS remote.")

    except Exception as e:
        send_slack_alert(f"Task 6 🚨 *DVC Versioning FAILED*\n{str(e)}", is_error=True)
        raise


# ─────────────────────────────────────────
# Task 7: Load to PostgreSQL
# ─────────────────────────────────────────

def task_load_data(**kwargs) -> None:
    """Load cleaned data into Cloud SQL PostgreSQL."""
    import sys
    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from load_data import load_data

    cleaned = {
        "courses":       pd.read_json(CLEAN_COURSES_PATH),
        "prerequisites": pd.read_json(CLEAN_PREREQS_PATH),
    }

    try:
        counts = load_data(cleaned)
        logging.info("Load complete: %s", counts)
        send_slack_alert(
            f"Task 7 ✅ *Load DB* complete — "
            f"{counts['courses_inserted']} courses, "
            f"{counts['prerequisites_inserted']} prerequisites inserted."
        )
    except Exception as e:
        send_slack_alert(f"Task 7 🚨 *Load DB FAILED*\n{str(e)}", is_error=True)
        raise


# ─────────────────────────────────────────
# Task 8: Pipeline Report
# ─────────────────────────────────────────

def task_pipeline_report(**kwargs) -> None:
    """
    Generate a final pipeline summary report.
    Saves to GCS reports/ folder and sends Slack summary.
    """
    from google.cloud import storage

    stats = {}
    bias  = {}

    if STATS_REPORT_PATH.exists():
        with open(STATS_REPORT_PATH) as f:
            stats = json.load(f)

    if BIAS_REPORT_PATH.exists():
        with open(BIAS_REPORT_PATH) as f:
            bias = json.load(f)

    report = {
        "pipeline_run_at": datetime.now(timezone.utc).isoformat(),
        "status":          "SUCCESS",
        "data_stats":      stats,
        "bias_summary": {
            "flags_count": len(bias.get("bias_flags", [])),
            "flags":       bias.get("bias_flags", []),
        },
    }

    PIPELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    client = storage.Client()
    bucket = client.bucket("courseweave-ai-data")
    blob   = bucket.blob(f"reports/pipeline_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json")
    blob.upload_from_filename(str(PIPELINE_REPORT_PATH))
    logging.info("Pipeline report uploaded to GCS reports/")

    total_courses = stats.get("courses", {}).get("total_courses", "N/A")
    bias_flags    = len(bias.get("bias_flags", []))

    send_slack_alert(
        f"🎉 *CourseWeave Pipeline Complete!*\n"
        f"• Total courses processed: {total_courses}\n"
        f"• Bias flags: {bias_flags}\n"
        f"• Report saved to GCS reports/\n"
        f"• Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    logging.info("Pipeline report complete.")


# ─────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────

with DAG(
    dag_id="courseweave_data_pipeline",
    default_args=default_args,
    description="CourseWeave AI — End-to-end data pipeline",
    schedule="@weekly",
    catchup=False,
    tags=["courseweave", "data-pipeline", "mlops"],
) as dag:

    acquire_task = PythonOperator(
        task_id="acquire_data",
        python_callable=task_acquire_data,
    )

    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=task_preprocess_data,
    )

    validate_task = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    anomaly_task = PythonOperator(
        task_id="detect_anomalies",
        python_callable=task_detect_anomalies,
    )

    bias_task = PythonOperator(
        task_id="bias_detection",
        python_callable=task_bias_detection,
    )

    dvc_task = PythonOperator(
        task_id="dvc_versioning",
        python_callable=task_dvc_versioning,
    )

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=task_load_data,
    )

    report_task = PythonOperator(
        task_id="pipeline_report",
        python_callable=task_pipeline_report,
    )

    # ── Task execution order ──────────────────────────────
    acquire_task >> preprocess_task >> validate_task >> anomaly_task >> bias_task >> dvc_task >> load_task >> report_task
