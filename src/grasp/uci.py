"""Failure mode classification on UCI Robot Execution Failures (LP1-LP5).

Each sample is 15 force/torque readings (Fx Fy Fz Tx Ty Tz) taken right after
a failure was detected. The sets are tiny (47-164 samples), so results are
reported as mean and spread over repeated stratified folds, never one split.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see train_kuka.py
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from grasp import cnn

SEED = 7
N_READINGS, N_CHANNELS = 15, 6


def parse_lp(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return X with shape (n, 15, 6) and the class label of each sample."""
    X, y, rows, label = [], [], [], None
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1 and not parts[0].lstrip("-").isdigit():
            if label is not None:
                X.append(rows)
                y.append(label)
            label, rows = parts[0], []
        else:
            rows.append([float(v) for v in parts])
    if label is not None:
        X.append(rows)
        y.append(label)
    arr = np.asarray(X, dtype=np.float32)
    if arr.shape[1:] != (N_READINGS, N_CHANNELS):
        raise ValueError(f"{path}: unexpected sample shape {arr.shape}")
    return arr, y


def summary_features(X: np.ndarray) -> np.ndarray:
    t = np.arange(X.shape[1])
    slope = np.polyfit(t, X.transpose(1, 0, 2).reshape(X.shape[1], -1), 1)[0].reshape(len(X), -1)
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, 0], X[:, -1],
                           slope, np.abs(X).max(1)], axis=1)


def merge_rare(y: list[str], min_count: int = 2) -> tuple[list[str], list[str]]:
    counts = Counter(y)
    rare = sorted(c for c, n in counts.items() if n < min_count)
    return [("other" if c in rare else c) for c in y], rare


def evaluate_lp(path: Path) -> list[dict]:
    X, labels = parse_lp(path)
    labels, rare = merge_rare(labels)
    classes = sorted(set(labels))
    y = np.array([classes.index(c) for c in labels])
    counts = Counter(labels)
    n_splits = max(2, min(5, min(counts.values())))
    notes = []
    if n_splits < 5:
        notes.append(f"{n_splits}-fold because the smallest class has {min(counts.values())} samples")
    if rare:
        notes.append(f"merged singleton classes into 'other': {rare}")
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=SEED)
    F = summary_features(X)
    seq = X.transpose(0, 2, 1)
    results = []
    for name in ("majority", "xgboost", "cnn1d"):
        f1s, accs = [], []
        for tr, te in cv.split(F, y):
            if name == "majority":
                pred = DummyClassifier(strategy="most_frequent").fit(F[tr], y[tr]).predict(F[te])
            elif name == "xgboost":
                # Folds can miss a class, so re-index labels to what the fold has seen.
                seen = np.unique(y[tr])
                remap = {c: i for i, c in enumerate(seen)}
                model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                      subsample=0.9, random_state=SEED, n_jobs=1)
                model.fit(F[tr], np.array([remap[c] for c in y[tr]]))
                pred = seen[model.predict(F[te])]
            else:
                proba = cnn.fit_predict(seq[tr], y[tr], seq[te], n_classes=len(classes),
                                        epochs=150, seed=SEED)
                pred = proba.argmax(1) if proba.ndim == 2 else (proba > 0.5).astype(int)
            f1s.append(f1_score(y[te], pred, average="macro", labels=np.unique(y), zero_division=0))
            accs.append(accuracy_score(y[te], pred))
        row = {"lp": path.stem.upper(), "model": name, "n_samples": int(len(y)),
               "n_classes": len(classes), "class_counts": dict(sorted(counts.items())),
               "n_splits": n_splits, "macro_f1_mean": float(np.mean(f1s)),
               "macro_f1_std": float(np.std(f1s)), "accuracy_mean": float(np.mean(accs)),
               "accuracy_std": float(np.std(accs)), "notes": "; ".join(notes)}
        results.append(row)
        with mlflow.start_run(run_name=f"uci_{row['lp']}_{name}"):
            mlflow.log_params({"task": "uci_failure_mode", "lp": row["lp"], "model": name,
                               "cv": f"repeated_stratified_{n_splits}x3", "seed": SEED})
            mlflow.log_metrics({k: row[k] for k in ("macro_f1_mean", "macro_f1_std",
                                                    "accuracy_mean", "accuracy_std")})
        print(f"{row['lp']} {name:9s} n={len(y):3d} classes={len(classes)} "
              f"macro-F1 {row['macro_f1_mean']:.3f}±{row['macro_f1_std']:.3f} "
              f"acc {row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/uci"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("uci-robot-failures")
    rows = [r for p in sorted(args.raw.glob("lp*.data")) for r in evaluate_lp(p)]
    (args.out / "uci_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
