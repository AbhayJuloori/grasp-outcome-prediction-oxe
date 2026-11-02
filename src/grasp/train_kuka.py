"""Early grasp success prediction on kuka with leave-one-cell-out validation.

Each fold holds out every episode from one physical robot cell, so the score
measures transfer to a cell the model has never seen. Hyperparameters are
fixed up front; nothing is tuned on held-out folds.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# torch and xgboost each bundle their own libomp on macOS; with more than one
# OpenMP thread the two runtimes deadlock or crash. The data is small, so one
# thread costs little.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import duckdb
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from grasp import cnn
from grasp.features import load_window, sequence_tensor, tabular_features

SEED = 7
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=2, eval_metric="logloss")


def make_xgb() -> XGBClassifier:
    return XGBClassifier(**XGB_PARAMS, random_state=SEED, n_jobs=1)


def make_logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=0.5))


def predict_fold(name: str, X: pd.DataFrame, seq: np.ndarray, y: np.ndarray,
                 tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    if name == "prevalence":
        return np.full(te.sum(), y[tr].mean())
    if name == "logreg":
        return make_logreg().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    if name == "xgboost":
        return make_xgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    if name == "cnn1d":
        return cnn.fit_predict(seq[tr], y[tr], seq[te], seed=SEED)
    raise ValueError(name)


def retry_operating_point(y: np.ndarray, p_success: np.ndarray, min_precision: float = 0.9) -> dict:
    """Highest failure recall at which flagged episodes are failures >= min_precision of the time."""
    fail = 1 - y
    order = np.argsort(p_success)  # most confident failures first
    hits = np.cumsum(fail[order])
    flagged = np.arange(1, len(y) + 1)
    precision = hits / flagged
    ok = np.where(precision >= min_precision)[0]
    if len(ok) == 0:
        return {"failure_precision": None, "failure_recall": 0.0, "threshold": None}
    i = ok.max()
    return {"failure_precision": float(precision[i]), "failure_recall": float(hits[i] / fail.sum()),
            "threshold_p_success": float(p_success[order][i]), "flagged_share": float(flagged[i] / len(y))}


def evaluate(con: duckdb.DuckDBPyConnection, k: int, models: list[str], out_dir: Path) -> list[dict]:
    episodes, steps = load_window(con, k)
    Xdf = tabular_features(episodes, steps, k)
    X = Xdf.to_numpy(dtype=np.float32)
    seq = sequence_tensor(episodes, steps, k)
    y = episodes.success.to_numpy()
    cells = episodes.cell_id.to_numpy()
    rows, oof_all = [], {"episode_id": episodes.episode_id, "cell_id": cells, "success": y}
    for name in models:
        oof = np.zeros(len(y))
        fold_auc, fold_ap = [], []
        for cell in np.unique(cells):
            te = cells == cell
            tr = ~te
            oof[te] = predict_fold(name, X, seq, y, tr, te)
            fold_auc.append(roc_auc_score(y[te], oof[te]) if name != "prevalence" else 0.5)
            fold_ap.append(average_precision_score(y[te], oof[te]))
        row = {
            "k": k, "model": name, "n_episodes": int(len(y)), "success_rate": float(y.mean()),
            "roc_auc_mean": float(np.mean(fold_auc)), "roc_auc_std": float(np.std(fold_auc)),
            "pr_auc_mean": float(np.mean(fold_ap)), "pr_auc_std": float(np.std(fold_ap)),
            "roc_auc_pooled": float(roc_auc_score(y, oof)) if name != "prevalence" else 0.5,
            "pr_auc_pooled": float(average_precision_score(y, oof)),
            **{f"retry_{key}": v for key, v in retry_operating_point(y, oof).items()},
        }
        rows.append(row)
        oof_all[name] = oof
        with mlflow.start_run(run_name=f"kuka_{name}_k{k}"):
            mlflow.log_params({"task": "kuka_grasp_success", "model": name, "k_steps": k,
                               "cv": "leave_one_cell_out", "n_cells": int(len(np.unique(cells))),
                               "seed": SEED, **({f"xgb_{p}": v for p, v in XGB_PARAMS.items()}
                                                 if name == "xgboost" else {})})
            mlflow.log_metrics({m: v for m, v in row.items()
                                if isinstance(v, float) and m not in ("success_rate",)})
        print(f"k={k} {name:10s} AUC {row['roc_auc_mean']:.3f}±{row['roc_auc_std']:.3f} "
              f"PR-AUC {row['pr_auc_mean']:.3f}±{row['pr_auc_std']:.3f} (base {y.mean():.3f})")
    pd.DataFrame(oof_all).to_parquet(out_dir / f"kuka_oof_k{k}.parquet", index=False)
    if k == 3:
        Xdf.assign(episode_id=episodes.episode_id.values, cell_id=cells, success=y).to_parquet(
            out_dir / "kuka_features_k3.parquet", index=False)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/warehouse.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--models", nargs="+", default=["prevalence", "logreg", "xgboost", "cnn1d"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("grasp-outcome-kuka")
    con = duckdb.connect(str(args.db), read_only=True)
    rows = [r for k in args.ks for r in evaluate(con, k, args.models, args.out)]
    (args.out / "kuka_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
