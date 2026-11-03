"""Render static charts and a metrics bundle for the GitHub Pages site."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                             roc_curve)

ART, ASSETS = Path("artifacts"), Path("docs/assets")
COLORS = {"logreg": "#8a8f98", "xgboost": "#2f6fb0", "cnn1d": "#d0802f", "majority": "#c4c7cc"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def roc_by_k(kuka: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (model, g) in enumerate(kuka[kuka.model != "prevalence"].groupby("model")):
        ax.errorbar(g.k + (i - 1) * 0.06, g.roc_auc_mean, yerr=g.roc_auc_std, marker="o",
                    capsize=3, label=model, color=COLORS[model])
    ax.axhline(0.5, ls="--", color="#999", lw=1, label="chance")
    ax.set(xlabel="control steps observed (k)", ylabel="ROC AUC, held-out cell",
           title="Grasp success from the first k steps", xticks=sorted(kuka.k.unique()))
    ax.legend(frameon=False, ncol=4, loc="lower center")
    fig.tight_layout()
    fig.savefig(ASSETS / "roc_auc_by_k.png", dpi=150)


def curves_k3() -> None:
    """Pooled out-of-fold ROC and PR curves at k=3 (each episode scored by the fold that held out its cell)."""
    oof = pd.read_parquet(ART / "kuka_oof_k3.parquet")
    y = oof.success.to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for model in ["logreg", "xgboost", "cnn1d"]:
        fpr, tpr, _ = roc_curve(y, oof[model])
        prec, rec, _ = precision_recall_curve(y, oof[model])
        ax1.plot(fpr, tpr, color=COLORS[model], label=f"{model} ({roc_auc_score(y, oof[model]):.3f})")
        ax2.plot(rec, prec, color=COLORS[model], label=f"{model} ({average_precision_score(y, oof[model]):.3f})")
    ax1.plot([0, 1], [0, 1], ls="--", color="#999", lw=1)
    ax2.axhline(y.mean(), ls="--", color="#999", lw=1, label=f"base rate ({y.mean():.3f})")
    ax1.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC, k=3 (pooled out-of-fold)")
    ax2.set(xlabel="recall (success)", ylabel="precision (success)", title="Precision-recall, k=3", ylim=(0, 1.02))
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "roc_pr_curves_k3.png", dpi=150)


def per_cell(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    oof = pd.read_parquet(ART / "kuka_oof_k3.parquet")
    auc = oof.groupby("cell_id").apply(lambda d: roc_auc_score(d.success, d.xgboost),
                                       include_groups=False).rename("xgb_auc_k3")
    cells = con.sql("SELECT cell_id, count(*) AS n_episodes, avg(success) AS success_rate, "
                    "avg(n_steps) AS mean_steps FROM episodes GROUP BY 1 ORDER BY 1").df()
    cells = cells.merge(auc.reset_index(), on="cell_id")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(cells.cell_id.astype(str), cells.xgb_auc_k3, color=COLORS["xgboost"])
    ax.axhline(0.5, ls="--", color="#999", lw=1)
    for x, v in zip(range(len(cells)), cells.xgb_auc_k3):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
    ax.set(xlabel="held-out robot cell", ylabel="ROC AUC", ylim=(0.4, 0.85),
           title="XGBoost (k=3) on each unseen cell")
    fig.tight_layout()
    fig.savefig(ASSETS / "per_cell_auc_k3.png", dpi=150)
    return cells


def dq_chart(dq: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 2.4))
    ax.barh(dq.check_name, dq.episodes_flagged, color="#8a8f98")
    for y, v in enumerate(dq.episodes_flagged):
        ax.text(v + 0.3, y, str(v), va="center", fontsize=9)
    ax.set(xlabel="episodes flagged", title="SQL data quality checks (checks with zero hits omitted)")
    fig.tight_layout()
    fig.savefig(ASSETS / "dq_summary.png", dpi=150)


def uci_chart(uci: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    lps = sorted(uci.lp.unique())
    x = np.arange(len(lps))
    for i, model in enumerate(["majority", "xgboost", "cnn1d"]):
        g = uci[uci.model == model].set_index("lp").loc[lps]
        ax.bar(x + (i - 1) * 0.27, g.macro_f1_mean, 0.27, yerr=g.macro_f1_std, capsize=2,
               label=model, color=COLORS[model])
    ax.set(xticks=x, xticklabels=lps, ylabel="macro F1 (mean ± sd over folds)", ylim=(0, 1.05),
           title="UCI failure mode classification")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSETS / "uci_macro_f1.png", dpi=150)


def records(df: pd.DataFrame) -> list[dict]:
    """DataFrame rows as dicts with NaN turned into null, so the JSON stays valid."""
    return df.astype(object).where(df.notna(), None).to_dict("records")


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    kuka = pd.DataFrame(json.loads((ART / "kuka_results.json").read_text()))
    uci = pd.DataFrame(json.loads((ART / "uci_results.json").read_text()))
    con = duckdb.connect("data/warehouse.duckdb", read_only=True)
    dq = con.sql("SELECT * FROM dq_summary").df()
    counts = con.sql("SELECT (SELECT count(*) FROM episodes) AS episodes, "
                     "(SELECT count(*) FROM episodes_clean) AS clean, "
                     "(SELECT count(*) FROM steps) AS steps, "
                     "(SELECT avg(success) FROM episodes) AS success_rate").df().iloc[0]
    roc_by_k(kuka)
    curves_k3()
    cells = per_cell(con)
    dq_chart(dq)
    uci_chart(uci)
    bundle = {"counts": {k: float(v) for k, v in counts.items()}, "kuka": records(kuka),
              "uci": records(uci), "dq": records(dq), "cells": records(cells),
              "shap": json.loads((ART / "shap_kuka_k3.json").read_text())}
    (ASSETS / "metrics.json").write_text(json.dumps(bundle, indent=2, default=float))
    print(cells.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
