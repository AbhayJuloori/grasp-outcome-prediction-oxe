"""Airflow DAG for the grasp outcome pipeline.

Picked up by an Airflow 2.x deployment that has this repo checked out; Airflow
itself is not a project dependency. Set GRASP_REPO if the repo lives elsewhere.
"""
import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

REPO = os.environ.get("GRASP_REPO", str(Path(__file__).resolve().parents[1]))
PY = os.environ.get("GRASP_PYTHON", f"{REPO}/.venv/bin/python")
ENV = {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1", "PATH": os.environ.get("PATH", "")}


def step(task_id: str, command: str) -> BashOperator:
    return BashOperator(task_id=task_id, bash_command=f"cd {REPO} && {command}", env=ENV)


with DAG(
    dag_id="grasp_outcome_pipeline",
    schedule="@weekly",
    start_date=datetime(2026, 11, 1),
    catchup=False,
    tags=["robotics", "ml"],
) as dag:
    download = step("download", "scripts/download_data.sh")
    ingest = step("ingest", f"{PY} -m grasp.ingest")
    build = step("warehouse", f"{PY} -m grasp.warehouse")
    train = step("train_kuka", f"{PY} -m grasp.train_kuka")
    uci = step("uci", f"{PY} -m grasp.uci")
    explain = step("explain", f"{PY} -m grasp.explain")
    report = step("report", f"{PY} -m grasp.report")

    download >> ingest >> build >> [train, uci]
    train >> explain
    [explain, uci] >> report
