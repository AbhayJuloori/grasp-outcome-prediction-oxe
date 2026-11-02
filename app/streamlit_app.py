"""Grasp fleet dashboard: failure rate, cycle length and anomaly flags per robot cell.

Run with ``make dashboard`` after the pipeline has built data/warehouse.duckdb.
"""
import json
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "warehouse.duckdb"
ART = ROOT / "artifacts"

st.set_page_config(page_title="Grasp fleet dashboard", layout="wide")


@st.cache_resource
def connect():
    return duckdb.connect(str(DB), read_only=True)


@st.cache_data
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return connect().execute(sql, list(params)).df()


def load_json(name: str):
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else []


st.title("Grasp fleet dashboard")
st.caption("Open X-Embodiment kuka (QT-Opt) subset, one row per grasp attempt.")

kpi = query("""
    SELECT count(*) AS episodes, avg(success) AS success_rate, avg(n_steps) AS mean_steps,
           (SELECT count(DISTINCT episode_id) FROM dq_results) AS flagged
    FROM episodes""").iloc[0]
c = st.columns(5)
c[0].metric("Episodes", f"{int(kpi.episodes):,}")
c[1].metric("Success rate", f"{kpi.success_rate:.1%}")
c[2].metric("Failure rate", f"{1 - kpi.success_rate:.1%}")
c[3].metric("Mean cycle (steps)", f"{kpi.mean_steps:.1f}")
c[4].metric("Episodes with DQ flags", int(kpi.flagged))

st.subheader("Per robot cell")
cells = query("""
    SELECT e.cell_id, count(*) AS episodes, 1 - avg(e.success) AS failure_rate,
           avg(e.n_steps) AS mean_cycle_steps, count(DISTINCT d.episode_id) AS anomaly_flags
    FROM episodes e LEFT JOIN dq_results d USING (episode_id)
    GROUP BY e.cell_id ORDER BY e.cell_id""")
left, right = st.columns([2, 3])
left.dataframe(cells.style.format({"failure_rate": "{:.1%}", "mean_cycle_steps": "{:.1f}"}),
               hide_index=True)
right.bar_chart(cells.set_index("cell_id")[["failure_rate"]])

st.subheader("Data quality checks")
st.dataframe(query("SELECT * FROM dq_summary"), hide_index=True)

st.subheader("Early grasp success models (leave one cell out)")
results = pd.DataFrame(load_json("kuka_results.json"))
if not results.empty:
    st.dataframe(results[["k", "model", "n_episodes", "roc_auc_mean", "roc_auc_std",
                          "pr_auc_mean", "retry_failure_recall"]].round(3), hide_index=True)
shap_rows = pd.DataFrame(load_json("shap_kuka_k3.json"))
if not shap_rows.empty:
    st.markdown("**Top SHAP drivers (k=3)**")
    st.dataframe(shap_rows.round(3), hide_index=True)

st.subheader("Episode trace")
ids = query("SELECT episode_id FROM episodes ORDER BY episode_id").episode_id.tolist()
eid = st.selectbox("Episode", ids)
trace = query("SELECT step_idx, ee_z, height_to_bottom, gripper_closed FROM steps "
              "WHERE episode_id = ? ORDER BY step_idx", (eid,))
meta = query("SELECT success, n_steps, cell_id FROM episodes WHERE episode_id = ?", (eid,)).iloc[0]
st.caption(f"cell {meta.cell_id} · {meta.n_steps} steps · {'success' if meta.success else 'failure'}")
st.line_chart(trace.set_index("step_idx"))
