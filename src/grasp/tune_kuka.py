"""Nested leave-one-cell-out tuning check for the kuka XGBoost model.

The outer loop holds out one robot cell, exactly like train_kuka.py. Inside
each outer fold, a small XGBoost grid is scored by leave-one-cell-out on the
remaining six cells only; the winning config is refit on those six cells and
scored once on the held-out cell. The held-out cell never influences the
choice. Both the fixed config from train_kuka.py and the tuned model are run
with the same three seeds so the comparison is not a lucky draw.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see train_kuka.py

import duckdb
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from grasp.features import load_window, tabular_features

FIXED = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, min_child_weight=2)  # same as train_kuka.XGB_PARAMS
GRID = [dict(FIXED, max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=w)
        for d, lr, n, w in itertools.product([2, 3, 4], [0.03, 0.1], [150, 400], [2, 8])]
SEEDS = [7, 11, 23]


def fit_predict(params: dict, seed: int, X, y, tr, te) -> np.ndarray:
    m = XGBClassifier(**params, eval_metric="logloss", random_state=seed, n_jobs=1)
    return m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


def inner_score(params: dict, seed: int, X, y, cells, outer_tr) -> float:
    """Mean ROC AUC over leave-one-cell-out folds inside the outer training cells."""
    aucs = []
    for cell in np.unique(cells[outer_tr]):
        te = outer_tr & (cells == cell)
        tr = outer_tr & (cells != cell)
        aucs.append(roc_auc_score(y[te], fit_predict(params, seed, X, y, tr, te)))
    return float(np.mean(aucs))


def outer_fold(cell, seed: int, X, y, cells) -> dict:
    te = cells == cell
    tr = ~te
    scores = [inner_score(p, seed, X, y, cells, tr) for p in GRID]
    best = GRID[int(np.argmax(scores))]
    p_tuned = fit_predict(best, seed, X, y, tr, te)
    p_fixed = fit_predict(FIXED, seed, X, y, tr, te)
    return {"cell": int(cell), "seed": seed, "best": best, "best_inner_auc": max(scores),
            "tuned_auc": roc_auc_score(y[te], p_tuned), "tuned_ap": average_precision_score(y[te], p_tuned),
            "fixed_auc": roc_auc_score(y[te], p_fixed), "fixed_ap": average_precision_score(y[te], p_fixed)}


def summarize(folds: list[dict], key: str) -> dict:
    """Per seed: mean over the 7 held-out cells. Then mean and sd of that across seeds."""
    per_seed = [np.mean([f[key] for f in folds if f["seed"] == s]) for s in SEEDS]
    per_cell_sd = [np.std([f[key] for f in folds if f["seed"] == s]) for s in SEEDS]
    return {"mean": float(np.mean(per_seed)), "sd_across_seeds": float(np.std(per_seed)),
            "sd_across_cells": float(np.mean(per_cell_sd)), "per_seed": [float(v) for v in per_seed]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/warehouse.duckdb"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/kuka_tuning.json"))
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    con = duckdb.connect(str(args.db), read_only=True)
    result = {"grid_size": len(GRID), "seeds": SEEDS, "fixed_params": FIXED, "by_k": {}}
    for k in args.ks:
        episodes, steps = load_window(con, k)
        X = tabular_features(episodes, steps, k).to_numpy(dtype=np.float32)
        y = episodes.success.to_numpy()
        cells = episodes.cell_id.to_numpy()
        folds = Parallel(n_jobs=args.jobs)(delayed(outer_fold)(c, s, X, y, cells)
                                           for s in SEEDS for c in np.unique(cells))
        res = {m: summarize(folds, m) for m in ["fixed_auc", "tuned_auc", "fixed_ap", "tuned_ap"]}
        res["folds"] = folds
        result["by_k"][str(k)] = res
        print(f"k={k} fixed AUC {res['fixed_auc']['mean']:.4f} (seed sd {res['fixed_auc']['sd_across_seeds']:.4f}) "
              f"tuned AUC {res['tuned_auc']['mean']:.4f} (seed sd {res['tuned_auc']['sd_across_seeds']:.4f}) | "
              f"fixed PR {res['fixed_ap']['mean']:.4f} tuned PR {res['tuned_ap']['mean']:.4f}")
    args.out.write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
