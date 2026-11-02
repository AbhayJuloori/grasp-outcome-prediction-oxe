"""Build the DuckDB telemetry warehouse and run SQL data quality checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
DEFAULT_PARAMS = {"max_step_m": 0.25, "min_cmd_m": 0.01, "ws_margin_m": 0.05}


def render(sql: str, **params) -> str:
    for key, value in params.items():
        sql = sql.replace("{" + key + "}", str(value))
    return sql


def build(parquet_dir: Path, db_path: Path, **overrides) -> duckdb.DuckDBPyConnection:
    params = {**DEFAULT_PARAMS, **overrides, "parquet": parquet_dir.as_posix()}
    con = duckdb.connect(str(db_path))
    for name in ("01_marts.sql", "02_quality_checks.sql"):
        con.execute(render((SQL_DIR / name).read_text(), **params))
    return con


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", type=Path, default=Path("data/parquet"))
    ap.add_argument("--db", type=Path, default=Path("data/warehouse.duckdb"))
    args = ap.parse_args()
    con = build(args.parquet, args.db)
    print(con.sql("SELECT * FROM dim_cell"))
    print(con.sql("SELECT * FROM dq_summary"))
    print(con.sql("SELECT count(*) AS clean_episodes, avg(success) AS success_rate FROM episodes_clean"))


if __name__ == "__main__":
    main()
