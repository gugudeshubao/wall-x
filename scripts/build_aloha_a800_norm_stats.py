#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def split_stats(min_vals, max_vals):
    min_vals = np.asarray(min_vals, dtype=np.float32)
    max_vals = np.asarray(max_vals, dtype=np.float32)
    delta_vals = (max_vals - min_vals).tolist()
    min_vals = min_vals.tolist()

    return {
        "follow_left_ee_cartesian_pos": {
            "min": min_vals[0:3],
            "delta": delta_vals[0:3],
        },
        "follow_left_ee_rotation": {
            "min": min_vals[3:6],
            "delta": delta_vals[3:6],
        },
        "follow_left_gripper": {"min": min_vals[6:7], "delta": delta_vals[6:7]},
        "follow_right_ee_cartesian_pos": {
            "min": min_vals[7:10],
            "delta": delta_vals[7:10],
        },
        "follow_right_ee_rotation": {
            "min": min_vals[10:13],
            "delta": delta_vals[10:13],
        },
        "follow_right_gripper": {"min": min_vals[13:14], "delta": delta_vals[13:14]},
        # ALOHA does not provide these channels; keep them neutral and rely on masks.
        "head_actions": {"min": [0.0, 0.0], "delta": [1.0, 1.0]},
        "height": {"min": [0.0], "delta": [1.0]},
        "car_pose": {"min": [0.0, 0.0, 0.0], "delta": [1.0, 1.0, 1.0]},
    }


def main():
    repo_id = "lerobot/aloha_mobile_cabinet"
    root = Path("/root/autodl-tmp/datasets")
    output_path = Path("/root/autodl-tmp/norm_stats/aloha_mobile_cabinet_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False)

    action_min = None
    action_max = None
    state_min = None
    state_max = None

    for idx in range(len(dataset.hf_dataset)):
        row = dataset.hf_dataset[idx]
        action = np.asarray(row["action"], dtype=np.float32)
        state = np.asarray(row["observation.state"], dtype=np.float32)

        if action_min is None:
            action_min = action.copy()
            action_max = action.copy()
            state_min = state.copy()
            state_max = state.copy()
        else:
            action_min = np.minimum(action_min, action)
            action_max = np.maximum(action_max, action)
            state_min = np.minimum(state_min, state)
            state_max = np.maximum(state_max, state)

    merged_min = np.minimum(action_min, state_min)
    merged_max = np.maximum(action_max, state_max)

    stats = {repo_id: split_stats(merged_min, merged_max)}

    output_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
