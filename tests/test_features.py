import numpy as np

from grasp.features import SEQ_CHANNELS, load_window, sequence_tensor, tabular_features


def test_no_leaky_columns(con):
    eps, steps = load_window(con, 3)
    cols = tabular_features(eps, steps, 3).columns
    assert not any("terminate" in c or "n_steps" in c or "is_last" in c for c in cols)


def test_short_episodes_excluded(con):
    k = 5
    eps, _ = load_window(con, k)
    n = con.sql("SELECT episode_id, n_steps FROM episodes_clean").df().set_index("episode_id").n_steps
    assert (n.loc[eps.episode_id] > k).all()
    assert set(n[n <= k].index).isdisjoint(eps.episode_id)


def test_only_first_k_steps_used(con):
    k = 3
    eps, steps = load_window(con, k)
    before = tabular_features(eps, steps, k)
    con.execute("UPDATE steps SET ee_z = ee_z + 5, gripper_closed = 1 WHERE step_idx >= ?", [k])
    eps2, steps2 = load_window(con, k)
    after = tabular_features(eps2, steps2, k)
    assert np.allclose(before.to_numpy(float), after.to_numpy(float), equal_nan=True)


def test_sequence_shape(con):
    eps, steps = load_window(con, 3)
    assert sequence_tensor(eps, steps, 3).shape == (len(eps), len(SEQ_CHANNELS), 3)
