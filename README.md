# Early grasp outcome prediction on Open X-Embodiment (kuka) + UCI robot failures

Can a pick-and-place arm tell, a few control steps into a grasp, that the
attempt is going to fail, early enough to abort and retry? This repo builds a
telemetry warehouse from real robot episodes, checks it for data problems, and
trains models that predict grasp success from the first `k` steps of an
episode. A second task classifies failure modes from force/torque readings.

**Live summary page:** https://abhayjuloori.github.io/grasp-outcome-prediction-oxe/

## Data

| Source | What | Used here |
|---|---|---|
| [Open X-Embodiment](https://robotics-transformer-x.github.io) `kuka` (QT-Opt) | Real Kuka arm bin-picking episodes in RLDS/TFRecord format, with an episode-level `success` flag | 3 of 1,024 shards (00000, 00341, 00682): **1,709 episodes, 25,534 steps** across **7 robot cells** |
| [UCI Robot Execution Failures](https://archive.ics.uci.edu/dataset/138) | 15 force/torque readings (Fx Fy Fz Tx Ty Tz) after a failure, 5 learning problems | All **463 samples** (LP1-LP5) |

Per-step telemetry kept from kuka: end effector pose (xyz + quaternion), gripper
closed state, height above the bin floor, commanded translation/rotation,
gripper command. Camera images are skipped. The kuka subset has **no joint
angle readings**; the arm state is the end effector pose.

A "robot cell" is identified from the calibrated workspace bounds stored with
each episode; QT-Opt was collected on several physical Kuka setups and each has
its own bin calibration.

## Pipeline

```
scripts/download_data.sh   public HTTPS download (about 2.5 GB)
grasp.ingest               TFRecord -> Parquet (decodes zlib-compressed tensors, no TensorFlow needed)
grasp.warehouse            DuckDB marts (dim_cell, episodes, steps) + SQL data quality checks
grasp.train_kuka           early-window features, logistic regression / XGBoost / 1D CNN, leave-one-cell-out CV
grasp.uci                  failure mode classifiers, repeated stratified CV
grasp.explain              SHAP on the XGBoost model
grasp.report               charts + metrics.json for the Pages site
dags/grasp_pipeline.py     Airflow DAG wiring the steps above
app/streamlit_app.py       fleet dashboard (failure rate, cycle length, anomaly flags per cell)
```

Every model run is logged to MLflow (`sqlite:///mlflow.db`, experiments
`grasp-outcome-kuka` and `uci-robot-failures`).

## Data quality checks

`sql/02_quality_checks.sql` runs on every build. kuka has no wall-clock
timestamps, so dropped frames are detected through step index gaps, missing
first/last markers, row count mismatches, and end effector jumps.

| Check | Episodes flagged | Action |
|---|---:|---|
| step index gap, bad first/last markers, row count mismatch | 0 | drop |
| non-finite values, non-unit quaternion | 0 | drop |
| success flag disagrees with sparse reward | 0 | drop |
| EE jump > 0.25 m with no commanded motion (`ee_jump_uncommanded`) | 10 | **drop** |
| EE jump > 0.25 m that the policy commanded (`ee_large_commanded_move`) | 23 | keep, flag |
| EE outside workspace bounds (+5 cm) | 0 | keep, flag |

The jump check was split in two after looking at the data: all 23 commanded
large moves are failed grasps, so they are real behaviour. Dropping them as
"bad data" would have removed easy failures and inflated the scores.

## Evaluation design

- **Prediction point:** after the first `k` control steps. Only episodes that
  are still running at step `k` are scored (`n_steps > k`).
- **Leakage controls:** episode length is not a feature (the policy stops early
  after a grasp it believes worked, so length alone is very predictive), and
  the `terminate_episode` action is excluded. A test checks that editing steps
  at or after `k` never changes the features.
- **Split:** leave one robot cell out (7 folds). The score is how well the
  model transfers to a cell it has not seen. Hyperparameters are fixed before
  evaluation; nothing is tuned on held-out folds.
- **Metrics:** ROC AUC and PR AUC, mean ± sd across the 7 held-out cells. Base
  success rate is about 0.35, so a no-skill PR AUC is about 0.35.

## Results

### Grasp success from the first k steps (kuka, leave one cell out)

| k | Model | ROC AUC | PR AUC |
|---:|---|---:|---:|
| 3 | prevalence baseline | 0.500 | 0.363 ± 0.085 |
| 3 | logistic regression | 0.687 ± 0.041 | 0.560 ± 0.078 |
| 3 | XGBoost | **0.705 ± 0.040** | **0.563 ± 0.075** |
| 3 | 1D CNN (PyTorch) | 0.678 ± 0.024 | 0.546 ± 0.065 |
| 4 | logistic regression | 0.696 ± 0.036 | 0.564 ± 0.066 |
| 4 | XGBoost | **0.723 ± 0.042** | **0.602 ± 0.077** |
| 4 | 1D CNN (PyTorch) | 0.703 ± 0.025 | 0.593 ± 0.061 |

Full sweep for k = 2..5 is in `artifacts/kuka_results.json` and on the Pages site.
Per held-out cell, XGBoost at k=3 ranges from 0.648 to 0.758 ROC AUC.

**Retry trigger.** Flagging the most confident predicted failures at k=3 so
that at least 90% of flagged grasps really fail catches about **24% of all
failures** (18% of attempts flagged). That is the operating point a "stop and
regrasp" rule would use.

SHAP on the XGBoost model points mostly at the commanded vertical motion in the
first steps (an arm that is not descending toward the object tends to fail)
and at where the gripper sits in the bin.

### Failure mode classification (UCI, macro F1, 5-fold x 3 repeats)

| Problem | Samples | Classes | Majority | XGBoost | 1D CNN |
|---|---:|---:|---:|---:|---:|
| LP1 failures in approach to grasp | 88 | 4 | 0.139 | **0.935 ± 0.056** | 0.766 ± 0.116 |
| LP2 failures in transfer | 47 | 5 | 0.120 | **0.606 ± 0.173** | 0.550 ± 0.139 |
| LP3 position after transfer failure | 47 | 4 | 0.149 | **0.771 ± 0.144** | 0.699 ± 0.143 |
| LP4 failures in approach to ungrasp | 117 | 3 | 0.254 | 0.924 ± 0.068 | **0.926 ± 0.054** |
| LP5 failures in motion with part | 164 | 5 | 0.089 | **0.727 ± 0.078** | 0.672 ± 0.063 |

The spread matters more than the mean here: with 47 samples, one fold can swing
macro F1 by more than 0.15.

## Limitations

- There is no shared public benchmark for early grasp success prediction on
  this subset, so the numbers above are only comparable to the baselines in
  this repo.
- Three shards (about 0.3% of kuka) were used to keep the download small.
  Cells are a proxy for "scene": the shards do not record which physical robot
  or object set was used beyond the workspace calibration.
- Time is measured in control steps, not seconds; RLDS kuka does not store
  timestamps.
- The model only sees proprioception and commands. Adding the wrist camera
  would likely help a lot and is the obvious next step.
- UCI sets are tiny and classes are imbalanced; treat the per-problem scores as
  rough.

## How to run

```bash
make setup        # Python 3.11 venv via uv
make all          # download, ingest, warehouse, train, uci, explain, report (about 5 minutes on a laptop CPU)
make test         # pytest suite, synthetic data, no network
make dashboard    # Streamlit fleet dashboard
```

On macOS, `OMP_NUM_THREADS=1` is set by the Makefile and the scripts: torch and
xgboost each ship their own OpenMP runtime and deadlock when both use threads.

## Layout

```
sql/                DuckDB marts and data quality checks
src/grasp/          ingest, warehouse, features, cnn, train_kuka, uci, explain, report
tests/              pytest suite on synthetic episodes
app/                Streamlit dashboard
dags/               Airflow DAG
docs/               GitHub Pages site and charts
artifacts/          results JSON and out-of-fold predictions
```
