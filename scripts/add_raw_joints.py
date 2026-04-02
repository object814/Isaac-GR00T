#!/usr/bin/env python
"""
Add raw joint position fields to a GR00T LeRobot dataset.

Your dataset currently stores state as sin/cos encoded joint positions,
but actions are raw joint angles. To use ActionRepresentation.RELATIVE,
the stats script needs matching raw joint positions in the state.

This script:
1. Reads each parquet file
2. Recovers raw joint angles via atan2(sin, cos)
3. Expands observation.state to include raw joints
4. Updates meta/modality.json with the new state fields
5. Regenerates meta/stats.json (deletes stale one)

After running this, you can run gr00t/data/stats.py to generate
relative_stats.json.

Usage:
    python add_raw_joints.py --dataset-path /Isaac-GR00T/data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing files")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    modality_path = dataset_path / "meta" / "modality.json"
    stats_path = dataset_path / "meta" / "stats.json"
    rel_stats_path = dataset_path / "meta" / "relative_stats.json"

    # --- Load current modality.json ---
    with open(modality_path, "r") as f:
        modality = json.load(f)

    state = modality["state"]

    # Check if raw joints already exist
    if "left_arm" in state and "right_arm" in state:
        print("Raw joint keys (left_arm, right_arm) already exist in modality.json.")
        print("Nothing to do.")
        return

    # --- Verify expected sin/cos keys exist ---
    required = ["left_pos_cos", "left_pos_sin", "right_pos_cos", "right_pos_sin"]
    for k in required:
        assert k in state, f"Missing expected state key: {k}"

    # Current layout
    left_cos_start, left_cos_end = state["left_pos_cos"]["start"], state["left_pos_cos"]["end"]
    left_sin_start, left_sin_end = state["left_pos_sin"]["start"], state["left_pos_sin"]["end"]
    right_cos_start, right_cos_end = state["right_pos_cos"]["start"], state["right_pos_cos"]["end"]
    right_sin_start, right_sin_end = state["right_pos_sin"]["start"], state["right_pos_sin"]["end"]

    left_ndim = left_cos_end - left_cos_start   # 7
    right_ndim = right_cos_end - right_cos_start  # 7

    # Find the current end of the state array
    current_end = max(v["end"] for v in state.values())
    print(f"Current state dimension: {current_end}")
    print(f"Will append: left_arm ({left_ndim}D) + right_arm ({right_ndim}D)")
    new_end = current_end + left_ndim + right_ndim
    print(f"New state dimension: {new_end}")

    # New modality entries
    new_left_arm = {"start": current_end, "end": current_end + left_ndim}
    new_right_arm = {"start": current_end + left_ndim, "end": new_end}

    print(f"\nNew state keys:")
    print(f"  left_arm:  [{new_left_arm['start']}:{new_left_arm['end']}]")
    print(f"  right_arm: [{new_right_arm['start']}:{new_right_arm['end']}]")

    if args.dry_run:
        print("\n[DRY RUN] Would update the following files:")
        print(f"  - All parquet files in {dataset_path / 'data'}")
        print(f"  - {modality_path}")
        print(f"  - Delete {stats_path} (will be regenerated)")
        print(f"  - Delete {rel_stats_path} (will be regenerated)")
        return

    # --- Process each parquet file ---
    parquet_files = sorted(dataset_path.glob("data/**/*.parquet"))
    print(f"\nProcessing {len(parquet_files)} parquet files...")

    for pf in tqdm(parquet_files, desc="Updating parquet files"):
        df = pd.read_parquet(pf)

        new_states = []
        for idx in range(len(df)):
            state_arr = np.array(df.iloc[idx]["observation.state"], dtype=np.float32)

            # Extract sin/cos
            left_cos = state_arr[left_cos_start:left_cos_end]
            left_sin = state_arr[left_sin_start:left_sin_end]
            right_cos = state_arr[right_cos_start:right_cos_end]
            right_sin = state_arr[right_sin_start:right_sin_end]

            # Recover raw angles
            left_raw = np.arctan2(left_sin, left_cos).astype(np.float32)
            right_raw = np.arctan2(right_sin, right_cos).astype(np.float32)

            # Append to existing state
            new_state = np.concatenate([state_arr, left_raw, right_raw])
            new_states.append(new_state)

        df["observation.state"] = new_states
        df.to_parquet(pf, index=False)

    # --- Update modality.json ---
    modality["state"]["left_arm"] = new_left_arm
    modality["state"]["right_arm"] = new_right_arm

    with open(modality_path, "w") as f:
        json.dump(modality, f, indent=2)
    print(f"\nUpdated {modality_path}")

    # --- Delete stale stats so they get regenerated ---
    if stats_path.exists():
        stats_path.unlink()
        print(f"Deleted stale {stats_path}")

    if rel_stats_path.exists():
        rel_stats_path.unlink()
        print(f"Deleted stale {rel_stats_path}")

    # --- Update info.json state dimension ---
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        # Update the state shape if it's recorded
        if "features" in info and "observation.state" in info["features"]:
            old_shape = info["features"]["observation.state"].get("shape", [])
            if old_shape:
                info["features"]["observation.state"]["shape"] = [new_end]
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=4)
                print(f"Updated state shape in {info_path}: {old_shape} -> [{new_end}]")

    print("\n--- Done! ---")
    print("Next steps:")
    print("  1. Update your frank_config.py to add 'left_arm' and 'right_arm' to state modality_keys")
    print("  2. Run: python gr00t/data/stats.py <dataset_path> NEW_EMBODIMENT")
    print("     (make sure frank_config.py is imported first)")
    print("  3. Verify meta/relative_stats.json is populated")
    print("  4. Retrain with the updated config")


if __name__ == "__main__":
    main()