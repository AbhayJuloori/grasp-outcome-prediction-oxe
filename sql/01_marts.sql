-- Episode and step marts built from the Parquet landing zone.
-- {parquet} is replaced with the Parquet directory at run time.

CREATE OR REPLACE TABLE raw_episodes AS
SELECT * FROM read_parquet('{parquet}/episodes_*.parquet');

CREATE OR REPLACE TABLE raw_steps AS
SELECT * FROM read_parquet('{parquet}/steps_*.parquet');

-- One robot cell per distinct workspace origin. QT-Opt was collected on
-- several physical Kuka cells; each has its own calibrated bin bounds.
CREATE OR REPLACE TABLE dim_cell AS
SELECT
    dense_rank() OVER (ORDER BY round(ws_bound_0, 3)) - 1 AS cell_id,
    round(ws_bound_0, 3) AS ws_x_origin,
    count(*) AS n_episodes
FROM raw_episodes
GROUP BY round(ws_bound_0, 3);

CREATE OR REPLACE TABLE episodes AS
SELECT
    e.episode_id,
    e.shard,
    e.record_idx,
    c.cell_id,
    e.n_steps,
    e.success,
    least(e.ws_bound_0, e.ws_bound_3) AS ws_x_min,
    greatest(e.ws_bound_0, e.ws_bound_3) AS ws_x_max,
    least(e.ws_bound_1, e.ws_bound_7) AS ws_y_min,
    greatest(e.ws_bound_1, e.ws_bound_7) AS ws_y_max
FROM raw_episodes e
JOIN dim_cell c ON round(e.ws_bound_0, 3) = c.ws_x_origin;

CREATE OR REPLACE TABLE steps AS
SELECT
    s.*,
    ee_x - lag(ee_x) OVER w AS d_x,
    ee_y - lag(ee_y) OVER w AS d_y,
    ee_z - lag(ee_z) OVER w AS d_z,
    sqrt(pow(ee_x - lag(ee_x) OVER w, 2)
       + pow(ee_y - lag(ee_y) OVER w, 2)
       + pow(ee_z - lag(ee_z) OVER w, 2)) AS step_dist,
    -- translation the policy commanded on the previous step
    lag(sqrt(act_xyz_0 * act_xyz_0 + act_xyz_1 * act_xyz_1 + act_xyz_2 * act_xyz_2)) OVER w
        AS prev_cmd_dist,
    step_idx - lag(step_idx) OVER w AS idx_gap
FROM raw_steps s
WINDOW w AS (PARTITION BY episode_id ORDER BY step_idx);
