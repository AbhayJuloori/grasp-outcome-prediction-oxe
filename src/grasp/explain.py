"""SHAP attribution for the kuka XGBoost model at k=3.

This is post hoc attribution on a model fit to all eligible episodes, used to
interpret which early-window signals drive predictions. It is not an
evaluation; held-out scores come from train_kuka.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see train_kuka.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from grasp.train_kuka import make_xgb


def main(features: Path = Path("artifacts/kuka_features_k3.parquet"),
         out_json: Path = Path("artifacts/shap_kuka_k3.json"),
         out_png: Path = Path("docs/assets/shap_kuka_k3.png")) -> None:
    df = pd.read_parquet(features)
    X = df.drop(columns=["episode_id", "cell_id", "success"])
    model = make_xgb().fit(X, df.success)
    values = shap.TreeExplainer(model).shap_values(X)
    mean_abs = np.abs(values).mean(0)
    top = np.argsort(mean_abs)[::-1][:15]
    rows = []
    for i in top:
        x = X.iloc[:, i].to_numpy(dtype=float)
        ok = np.isfinite(x)
        corr = np.corrcoef(x[ok], values[ok, i])[0, 1] if x[ok].std() > 0 else 0.0
        rows.append({"feature": X.columns[i], "mean_abs_shap": float(mean_abs[i]),
                     "direction": "higher -> more success" if corr > 0 else "higher -> more failure"})
    out_json.write_text(json.dumps(rows, indent=2))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, X, max_display=12, show=False)
    plt.title("SHAP: early-window drivers of grasp success (k=3)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    for r in rows[:8]:
        print(f"{r['feature']:24s} {r['mean_abs_shap']:.3f}  {r['direction']}")


if __name__ == "__main__":
    main()
