"""Early-window features for grasp outcome prediction.

The prediction is made after the first ``k`` control steps of an episode, so
only those steps are visible. Two things are deliberately excluded because
they leak the outcome: the episode length (the QT-Opt policy stops early after
a grasp it believes succeeded) and the ``terminate_episode`` action. Episodes
that end at or before step ``k`` are not eligible: there is nothing left to
predict or retry.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

SEQ_CHANNELS = [
    "ee_x_rel", "ee_y_rel", "ee_z", "height_to_bottom", "gripper_closed",
    "act_xyz_0", "act_xyz_1", "act_xyz_2", "act_rot_2", "act_gripper",
]


def load_window(con: duckdb.DuckDBPyConnection, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (episodes, steps) restricted to eligible episodes and their first k steps."""
    episodes = con.execute(
        "SELECT episode_id, cell_id, success, ws_x_min, ws_x_max, ws_y_min, ws_y_max "
        "FROM episodes_clean WHERE n_steps > ? ORDER BY episode_id",
        [k],
    ).df()
    steps = con.execute(
        "SELECT s.* FROM steps s JOIN episodes_clean e USING (episode_id) "
        "WHERE e.n_steps > ? AND s.step_idx < ? ORDER BY s.episode_id, s.step_idx",
        [k, k],
    ).df()
    steps = steps.merge(episodes[["episode_id", "ws_x_min", "ws_x_max", "ws_y_min", "ws_y_max"]],
                        on="episode_id")
    # Position relative to the cell's bin, so cells with different calibration compare.
    steps["ee_x_rel"] = (steps.ee_x - steps.ws_x_min) / (steps.ws_x_max - steps.ws_x_min)
    steps["ee_y_rel"] = (steps.ee_y - steps.ws_y_min) / (steps.ws_y_max - steps.ws_y_min)
    return episodes, steps


def tabular_features(episodes: pd.DataFrame, steps: pd.DataFrame, k: int) -> pd.DataFrame:
    """Flatten per-step channels plus a few window summaries into one row per episode."""
    wide = steps.pivot(index="episode_id", columns="step_idx", values=SEQ_CHANNELS)
    wide.columns = [f"{c}_t{t}" for c, t in wide.columns]
    g = steps.groupby("episode_id")
    summary = pd.DataFrame({
        "path_len": g.step_dist.sum(min_count=1).fillna(0.0),
        "min_height": g.height_to_bottom.min(),
        "mean_height": g.height_to_bottom.mean(),
        "gripper_closed_any": g.gripper_closed.max(),
        "gripper_cmd_sum": g.act_gripper.sum(),
        "cmd_xyz_mean": g.apply(lambda d: np.sqrt(d.act_xyz_0**2 + d.act_xyz_1**2 + d.act_xyz_2**2).mean(),
                                include_groups=False),
        "z_drop": g.ee_z.first() - g.ee_z.last(),
    })
    feats = wide.join(summary)
    return feats.loc[episodes.episode_id].reset_index(drop=True)


def sequence_tensor(episodes: pd.DataFrame, steps: pd.DataFrame, k: int) -> np.ndarray:
    """Array of shape (n_episodes, n_channels, k) for the 1D CNN."""
    arr = steps.set_index(["episode_id", "step_idx"])[SEQ_CHANNELS]
    out = np.zeros((len(episodes), len(SEQ_CHANNELS), k), dtype=np.float32)
    for i, eid in enumerate(episodes.episode_id):
        out[i] = arr.loc[eid].to_numpy(dtype=np.float32).T
    return out
