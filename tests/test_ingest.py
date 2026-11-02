import struct
import zlib

import numpy as np
from tfrecord import example_pb2

from grasp.ingest import parse_episode, read_records


def _example(n: int) -> bytes:
    ex = example_pb2.Example()
    f = ex.features.feature

    def zl(key, arr):
        f[key].bytes_list.value.extend(
            zlib.compress(np.asarray(a, dtype=np.float32).tobytes()) for a in arr)

    zl("steps/observation/clip_function_input/base_pose_tool_reached",
       [[0.5 + i, 0.1, 0.2, 0, 1, 0, 0] for i in range(n)])
    zl("steps/observation/gripper_closed", [[float(i == n - 1)] for i in range(n)])
    zl("steps/observation/height_to_bottom", [[0.1] for _ in range(n)])
    zl("steps/observation/task_id", [[1.0] for _ in range(n)])
    zl("steps/observation/clip_function_input/workspace_bounds", [np.arange(9) for _ in range(n)])
    f["steps/observation/natural_language_instruction"].bytes_list.value.extend([b"pick anything"] * n)
    f["steps/action/world_vector"].float_list.value.extend([0.01] * 3 * n)
    f["steps/action/rotation_delta"].float_list.value.extend([0.0] * 3 * n)
    f["steps/action/gripper_closedness_action"].float_list.value.extend([0.0] * n)
    f["steps/action/terminate_episode"].int64_list.value.extend([0, 1, 0] * n)
    f["steps/action/base_displacement_vector"].float_list.value.extend([0.0] * 2 * n)
    f["steps/action/base_displacement_vertical_rotation"].float_list.value.extend([0.0] * n)
    f["steps/reward"].float_list.value.extend([0.0] * (n - 1) + [1.0])
    f["steps/is_first"].int64_list.value.extend([1] + [0] * (n - 1))
    f["steps/is_last"].int64_list.value.extend([0] * (n - 1) + [1])
    f["steps/is_terminal"].int64_list.value.extend([0] * (n - 1) + [1])
    f["success"].int64_list.value.append(1)
    return ex.SerializeToString()


def test_read_and_parse(tmp_path):
    path = tmp_path / "kuka-train.tfrecord-00000-of-01024"
    with open(path, "wb") as fh:
        for n in (4, 6):
            data = _example(n)
            fh.write(struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4)
    records = list(read_records(path))
    assert len(records) == 2
    ep, steps = parse_episode(records[1], "e1")
    assert ep["n_steps"] == 6 and ep["success"] == 1 and ep["instruction"] == "pick anything"
    assert ep["ws_bound_8"] == 8.0
    assert len(steps) == 6
    assert np.allclose(steps.ee_x, 0.5 + np.arange(6))
    assert steps.gripper_closed.tolist() == [0, 0, 0, 0, 0, 1]
    assert steps.act_terminate_1.tolist() == [1] * 6
