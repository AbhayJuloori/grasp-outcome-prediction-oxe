"""Parse Open X-Embodiment `kuka` RLDS TFRecord shards into Parquet tables.

TFDS stores each RLDS episode as one ``tf.train.Example`` whose step-level
features are flattened lists (``steps/observation/gripper_closed`` holds one
value per step). We read the TFRecord framing directly and decode the protobuf
with the ``tfrecord`` package's generated classes, so TensorFlow is not needed.
Observation tensors are stored per step as zlib-compressed float32 bytes.
Camera images are skipped; only telemetry is kept.
"""
from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from tfrecord import example_pb2

# (feature key, output column prefix, width per step)
STEP_FEATURES = [
    ("steps/observation/clip_function_input/base_pose_tool_reached", "ee", 7),
    ("steps/observation/gripper_closed", "gripper_closed", 1),
    ("steps/observation/height_to_bottom", "height_to_bottom", 1),
    ("steps/action/world_vector", "act_xyz", 3),
    ("steps/action/rotation_delta", "act_rot", 3),
    ("steps/action/gripper_closedness_action", "act_gripper", 1),
    ("steps/action/terminate_episode", "act_terminate", 3),
    ("steps/action/base_displacement_vector", "act_base", 2),
    ("steps/action/base_displacement_vertical_rotation", "act_base_rot", 1),
    ("steps/reward", "reward", 1),
    ("steps/is_first", "is_first", 1),
    ("steps/is_last", "is_last", 1),
    ("steps/is_terminal", "is_terminal", 1),
]
EE_NAMES = ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]


def read_records(path: Path) -> Iterator[bytes]:
    """Yield raw record payloads from a TFRecord file (CRCs are not checked)."""
    with open(path, "rb") as fh:
        while True:
            header = fh.read(8)
            if not header:
                return
            (length,) = struct.unpack("<Q", header)
            fh.read(4)
            data = fh.read(length)
            fh.read(4)
            if len(data) != length:
                raise IOError(f"truncated record in {path}")
            yield data


def _values(feature) -> np.ndarray:
    """Decode a feature to a flat array, inflating zlib-encoded float32 tensors."""
    kind = feature.WhichOneof("kind")
    if kind == "float_list":
        return np.asarray(feature.float_list.value, dtype=np.float32)
    if kind == "int64_list":
        return np.asarray(feature.int64_list.value, dtype=np.int64)
    raw = feature.bytes_list.value
    if raw and raw[0][:1] == b"x":
        try:
            return np.concatenate(
                [np.frombuffer(zlib.decompress(b), dtype=np.float32) for b in raw]
            )
        except zlib.error:
            pass
    return np.asarray(raw, dtype=object)


def parse_episode(raw: bytes, episode_id: str) -> tuple[dict, pd.DataFrame]:
    ex = example_pb2.Example()
    ex.ParseFromString(raw)
    feats = ex.features.feature
    n_steps = len(feats["steps/is_first"].int64_list.value)

    cols: dict[str, np.ndarray] = {}
    for key, prefix, width in STEP_FEATURES:
        if key not in feats:
            continue
        vals = _values(feats[key])
        if vals.size != n_steps * width:
            raise ValueError(f"{episode_id}: {key} has {vals.size} values, expected {n_steps * width}")
        vals = vals.reshape(n_steps, width)
        if prefix == "ee":
            names = EE_NAMES
        elif width == 1:
            names = [prefix]
        else:
            names = [f"{prefix}_{i}" for i in range(width)]
        for i, name in enumerate(names):
            cols[name] = vals[:, i]

    steps = pd.DataFrame(cols)
    steps.insert(0, "step_idx", np.arange(n_steps, dtype=np.int32))
    steps.insert(0, "episode_id", episode_id)

    bounds = _values(feats["steps/observation/clip_function_input/workspace_bounds"])
    bounds = bounds[:9] if bounds.size >= 9 else np.full(9, np.nan)
    instr = feats["steps/observation/natural_language_instruction"].bytes_list.value
    task = feats["steps/observation/task_id"]
    episode = {
        "episode_id": episode_id,
        "n_steps": n_steps,
        "success": int(_values(feats["success"])[0]),
        "instruction": instr[0].decode() if instr else "",
        "task_id": float(_values(task)[0]) if len(_values(task)) else np.nan,
        **{f"ws_bound_{i}": float(v) for i, v in enumerate(bounds)},
    }
    return episode, steps


def ingest_shard(path: Path, out_dir: Path) -> int:
    shard = path.name.split("-")[-3]  # kuka-train.tfrecord-00000-of-01024
    episodes, steps = [], []
    for i, raw in enumerate(read_records(path)):
        ep, st = parse_episode(raw, f"kuka_{shard}_{i:04d}")
        ep["shard"] = shard
        ep["record_idx"] = i
        episodes.append(ep)
        steps.append(st)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episodes).to_parquet(out_dir / f"episodes_{shard}.parquet", index=False)
    pd.concat(steps, ignore_index=True).to_parquet(out_dir / f"steps_{shard}.parquet", index=False)
    return len(episodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/kuka"))
    ap.add_argument("--out", type=Path, default=Path("data/parquet"))
    args = ap.parse_args()
    shards = sorted(args.raw.glob("kuka-train.tfrecord-*"))
    if not shards:
        raise SystemExit(f"no shards in {args.raw}; run scripts/download_data.sh")
    for path in shards:
        n = ingest_shard(path, args.out)
        print(f"{path.name}: {n} episodes")


if __name__ == "__main__":
    main()
