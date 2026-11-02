import numpy as np

from grasp import cnn
from grasp.train_kuka import retry_operating_point
from grasp.uci import merge_rare, parse_lp, summary_features


def test_cnn_binary_and_multiclass():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4, 5)).astype(np.float32)
    yb = (x[:, 0, 0] > 0).astype(int)
    p1 = cnn.fit_predict(x, yb, x[:7], epochs=3, seed=1)
    p2 = cnn.fit_predict(x, yb, x[:7], epochs=3, seed=1)
    assert p1.shape == (7,) and ((p1 >= 0) & (p1 <= 1)).all()
    assert np.allclose(p1, p2)
    pm = cnn.fit_predict(x, rng.integers(0, 3, 40), x[:5], n_classes=3, epochs=3)
    assert pm.shape == (5, 3) and np.allclose(pm.sum(1), 1, atol=1e-5)


def test_retry_operating_point():
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    p = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9])
    out = retry_operating_point(y, p, min_precision=0.9)
    assert out["failure_recall"] == 0.75 and out["failure_precision"] == 1.0


def test_parse_lp(tmp_path):
    rows = "\n".join("\t" + "\t".join(str(i + j) for j in range(6)) for i in range(15))
    path = tmp_path / "lp9.data"
    path.write_text(f"normal\n{rows}\n\n\ncollision\n{rows}\n\n")
    X, y = parse_lp(path)
    assert X.shape == (2, 15, 6) and y == ["normal", "collision"]
    assert X[0, 14, 5] == 19
    assert summary_features(X).shape == (2, 48)


def test_merge_rare():
    y, rare = merge_rare(["a", "a", "b", "c", "c"])
    assert rare == ["b"] and y == ["a", "a", "other", "c", "c"]
