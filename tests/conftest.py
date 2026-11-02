import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pytest

from grasp import warehouse

WS = [0.47, -0.16, 0.07, 0.62, -0.16, 0.07, 0.47, 0.24, 0.07]


def make_episode(eid: str, cell: int, n: int, success: int, rng) -> tuple[dict, pd.DataFrame]:
    t = np.arange(n)
    steps = pd.DataFrame({
        "episode_id": eid, "step_idx": t.astype(np.int32),
        "ee_x": 0.5 + 0.01 * t, "ee_y": 0.0 + 0.005 * t, "ee_z": 0.15 - 0.01 * t,
        "ee_qx": 0.0, "ee_qy": 1.0, "ee_qz": 0.0, "ee_qw": 0.0,
        "gripper_closed": (t >= n - 2).astype(float), "height_to_bottom": 0.1 - 0.01 * t,
        **{f"act_xyz_{i}": rng.normal(0, 0.02, n) for i in range(3)},
        **{f"act_rot_{i}": 0.0 for i in range(3)},
        "act_gripper": 0.0, "act_terminate_0": 0, "act_terminate_1": 1, "act_terminate_2": 0,
        "act_base_0": 0.0, "act_base_1": 0.0, "act_base_rot": 0.0,
        "reward": 0.0, "is_first": (t == 0).astype(int), "is_last": (t == n - 1).astype(int),
        "is_terminal": (t == n - 1).astype(int),
    })
    if success:
        steps.loc[n - 2, "reward"] = 1.0
    ws = list(WS)
    ws[0] = ws[6] = WS[0] + 0.01 * cell
    ep = {"episode_id": eid, "n_steps": n, "success": success, "instruction": "pick anything",
          "task_id": 1.0, **{f"ws_bound_{i}": v for i, v in enumerate(ws)},
          "shard": "00000", "record_idx": 0}
    return ep, steps


@pytest.fixture
def landing(tmp_path):
    rng = np.random.default_rng(0)
    eps, steps = [], []
    for i in range(12):
        ep, st = make_episode(f"ep{i:02d}", cell=i % 3, n=4 + i % 5, success=i % 2, rng=rng)
        eps.append(ep)
        steps.append(st)
    # ep00: uncommanded 0.4 m jump at step 2 -> dropped
    s0 = steps[0]
    s0.loc[2, "ee_x"] += 0.4
    s0.loc[1, ["act_xyz_0", "act_xyz_1", "act_xyz_2"]] = 0.0
    # ep01: commanded 0.4 m move -> flagged, kept
    s1 = steps[1]
    s1.loc[2:, "ee_z"] += 0.4
    s1.loc[1, ["act_xyz_0", "act_xyz_1", "act_xyz_2"]] = [0.0, 0.0, 0.3]
    # ep03: success flag set but no reward -> label mismatch
    steps[3]["reward"] = 0.0
    pq = tmp_path / "parquet"
    pq.mkdir()
    pd.DataFrame(eps).to_parquet(pq / "episodes_00000.parquet", index=False)
    pd.concat(steps, ignore_index=True).to_parquet(pq / "steps_00000.parquet", index=False)
    return pq


@pytest.fixture
def con(landing, tmp_path):
    c = warehouse.build(landing, tmp_path / "wh.duckdb")
    yield c
    c.close()
