# SPDX-License-Identifier: Apache-2.0
"""
Scan training parquets and dump per-state-key statistics and a first-episode
trace, so we can compare against what the eval script logs at inference time.

Specifically: per-state-key (sliced from observation.state via meta/modality.json)
  - min/max/mean/std across the full dataset
  - first 50 steps of the first trajectory for `state.left_gripper` and
    `state.right_gripper`, so we can check the integral semantics

Reads parquet only (no video). Should finish in seconds.

Usage:
    python scripts/scan_state_stats.py \
        --dataset-path data/ \
        --n-first-steps 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def load_slices(dataset_path: str, modality: str) -> dict[str, tuple[int, int]]:
    with open(os.path.join(dataset_path, "meta", "modality.json")) as f:
        mod = json.load(f)
    return {k: (v["start"], v["end"]) for k, v in mod[modality].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--n-first-steps", type=int, default=50)
    parser.add_argument("--max-trajs", type=int, default=None)
    args = parser.parse_args()

    state_slices = load_slices(args.dataset_path, "state")
    print("state slices from meta/modality.json:")
    for k, (s, e) in state_slices.items():
        print(f"  state.{k}: [{s}:{e}]  (dim={e - s})")
    print()

    parquet_paths = sorted(
        glob.glob(os.path.join(args.dataset_path, "data", "*", "*.parquet"))
    )
    assert parquet_paths, f"No parquets under {args.dataset_path}/data/"
    if args.max_trajs is not None:
        parquet_paths = parquet_paths[: args.max_trajs]
    print(f"Scanning {len(parquet_paths)} trajectories...\n")

    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    first_traj_dump: dict[str, np.ndarray] | None = None

    for i, ppath in enumerate(parquet_paths):
        df = pd.read_parquet(ppath, columns=["observation.state"])
        states = np.stack(df["observation.state"].values).astype(np.float64)
        if i == 0:
            first_traj_dump = {
                short: states[: args.n_first_steps, s:e].copy()
                for short, (s, e) in state_slices.items()
            }
        for short, (s, e) in state_slices.items():
            collected[short].append(states[:, s:e])
        if (i + 1) % 20 == 0 or i == len(parquet_paths) - 1:
            print(f"  read {i + 1}/{len(parquet_paths)}")
    print()

    print("=" * 80)
    print("Per-state-key TRAINING distribution (over entire dataset)")
    print("=" * 80)
    for short in state_slices:
        vals = np.concatenate(collected[short], axis=0)  # (N, d)
        d = vals.shape[1]
        per_dim_min = vals.min(axis=0)
        per_dim_max = vals.max(axis=0)
        per_dim_mean = vals.mean(axis=0)
        per_dim_std = vals.std(axis=0)
        print(f"\nstate.{short}  (n={vals.shape[0]}, dim={d})")
        print(f"  global: min={vals.min():+.4f}  max={vals.max():+.4f}  "
              f"mean={vals.mean():+.4f}  std={vals.std():+.4f}")
        if d <= 8:
            for j in range(d):
                print(
                    f"  dim {j}: min={per_dim_min[j]:+.4f}  max={per_dim_max[j]:+.4f}  "
                    f"mean={per_dim_mean[j]:+.4f}  std={per_dim_std[j]:+.4f}"
                )

    if first_traj_dump is not None:
        print()
        print("=" * 80)
        print(f"First trajectory: first {args.n_first_steps} steps of grippers")
        print("=" * 80)
        for short in ("left_gripper", "right_gripper"):
            arr = first_traj_dump[short]
            print(f"\nstate.{short}  shape={arr.shape}")
            for t in range(arr.shape[0]):
                row = "  ".join(f"{v:+.4f}" for v in arr[t])
                print(f"  t={t:3d}: {row}")


if __name__ == "__main__":
    main()
