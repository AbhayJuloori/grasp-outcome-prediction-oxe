def _flags(con, check):
    return set(con.execute("SELECT episode_id FROM dq_results WHERE check_name = ?", [check])
               .df().episode_id)


def test_cells(con):
    assert con.execute("SELECT count(*) FROM dim_cell").fetchone()[0] == 3


def test_uncommanded_jump_dropped(con):
    assert "ep00" in _flags(con, "ee_jump_uncommanded")
    clean = set(con.sql("SELECT episode_id FROM episodes_clean").df().episode_id)
    assert "ep00" not in clean


def test_commanded_move_kept(con):
    assert "ep01" in _flags(con, "ee_large_commanded_move")
    assert "ep01" not in _flags(con, "ee_jump_uncommanded")
    clean = set(con.sql("SELECT episode_id FROM episodes_clean").df().episode_id)
    assert "ep01" in clean


def test_label_mismatch(con):
    assert _flags(con, "label_reward_mismatch") == {"ep03"}


def test_no_framing_errors(con):
    assert not _flags(con, "bad_episode_markers") and not _flags(con, "step_index_gap")
