-- Data quality checks. Each row of dq_results is one (check, episode) failure.
-- kuka has no wall-clock timestamps, so frame drops are detected from step
-- index gaps, missing first/last markers and implausible end effector jumps.

CREATE OR REPLACE TABLE dq_results AS
WITH framing AS (
    SELECT episode_id,
           sum(is_first) AS n_first,
           sum(is_last) AS n_last,
           max(step_idx) + 1 AS max_idx,
           count(*) AS n_rows
    FROM steps GROUP BY episode_id
)
SELECT 'step_index_gap' AS check_name, episode_id, max(idx_gap)::DOUBLE AS value
FROM steps WHERE idx_gap > 1 GROUP BY episode_id
UNION ALL
SELECT 'bad_episode_markers', episode_id, (n_first * 10 + n_last)::DOUBLE
FROM framing WHERE n_first <> 1 OR n_last <> 1
UNION ALL
SELECT 'row_count_mismatch', f.episode_id, (f.n_rows - e.n_steps)::DOUBLE
FROM framing f JOIN episodes e USING (episode_id) WHERE f.n_rows <> e.n_steps
UNION ALL
-- A control step moves the arm a few centimetres. A jump above {max_step_m} m
-- with (almost) no commanded translation means frames were lost or the pose
-- is corrupt. Large moves that the policy did command are real behaviour and
-- are only flagged as ee_large_commanded_move.
SELECT 'ee_jump_uncommanded', episode_id, max(step_dist)
FROM steps WHERE step_dist > {max_step_m} AND prev_cmd_dist < {min_cmd_m}
GROUP BY episode_id
UNION ALL
SELECT 'ee_large_commanded_move', episode_id, max(step_dist)
FROM steps WHERE step_dist > {max_step_m} AND prev_cmd_dist >= {min_cmd_m}
GROUP BY episode_id
UNION ALL
SELECT 'ee_out_of_workspace', s.episode_id, count(*)::DOUBLE
FROM steps s JOIN episodes e USING (episode_id)
WHERE s.ee_x < e.ws_x_min - {ws_margin_m} OR s.ee_x > e.ws_x_max + {ws_margin_m}
   OR s.ee_y < e.ws_y_min - {ws_margin_m} OR s.ee_y > e.ws_y_max + {ws_margin_m}
GROUP BY s.episode_id
UNION ALL
SELECT 'quaternion_not_unit', episode_id,
       max(abs(sqrt(ee_qx*ee_qx + ee_qy*ee_qy + ee_qz*ee_qz + ee_qw*ee_qw) - 1))
FROM steps
GROUP BY episode_id
HAVING max(abs(sqrt(ee_qx*ee_qx + ee_qy*ee_qy + ee_qz*ee_qz + ee_qw*ee_qw) - 1)) > 1e-3
UNION ALL
SELECT 'non_finite_value', episode_id, count(*)::DOUBLE
FROM steps
WHERE NOT (isfinite(ee_x) AND isfinite(ee_y) AND isfinite(ee_z) AND isfinite(height_to_bottom))
GROUP BY episode_id
UNION ALL
-- Label consistency: the success flag should agree with the sparse reward.
SELECT 'label_reward_mismatch', e.episode_id, sum(s.reward)
FROM episodes e JOIN steps s USING (episode_id)
GROUP BY e.episode_id, e.success
HAVING (e.success = 1) <> (sum(s.reward) > 0);

CREATE OR REPLACE TABLE dq_summary AS
SELECT check_name, count(DISTINCT episode_id) AS episodes_flagged
FROM dq_results GROUP BY check_name ORDER BY check_name;

-- Episodes kept for modelling: no framing or corruption problems.
-- Out-of-workspace poses and commanded large moves are kept; they are real
-- robot behaviour, and dropping them would hide genuine failures.
CREATE OR REPLACE TABLE episodes_clean AS
SELECT e.*,
       coalesce(o.flagged, 0) AS flag_out_of_workspace
FROM episodes e
LEFT JOIN (SELECT episode_id, 1 AS flagged FROM dq_results
           WHERE check_name = 'ee_out_of_workspace') o USING (episode_id)
WHERE e.episode_id NOT IN (
    SELECT episode_id FROM dq_results
    WHERE check_name NOT IN ('ee_out_of_workspace', 'ee_large_commanded_move')
);
