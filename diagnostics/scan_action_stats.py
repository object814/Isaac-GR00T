# SPDX-License-Identifier: Apache-2.0
"""
Scan a LeRobot dataset and report, per action key:
  - raw min/max/mean/std over all samples
  - fraction of samples outside the metadata's [q01, q99] and [min, max] ranges
  - for grippers: fraction of samples > 0.5 (i.e. "commanded close") and histogram

This reads the parquet files directly. It does NOT touch video, so no decoder
spam and it finishes in seconds even on large datasets.

Usage:
    python scripts/scan_action_stats.py \
        --dataset-path data/ \
        --data-config examples.frank.frank_config:FrankDataConfig
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from gr00t.experiment.data_config import load_data_config


def to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def histogram_line(values: np.ndarray, nbins: int = 10, lo: float | None = None, hi: float | None = None) -> str:
    if lo is None:
        lo = float(values.min())
    if hi is None:
        hi = float(values.max())
    if hi <= lo:
        return f"  (all values == {lo})"
    hist, edges = np.histogram(values, bins=nbins, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return "  (empty)"
    max_count = hist.max()
    out_lines = []
    for i, c in enumerate(hist):
        bar = "#" * int(40 * c / max_count) if max_count > 0 else ""
        pct = 100.0 * c / total
        out_lines.append(f"  [{edges[i]:+.4f}, {edges[i+1]:+.4f})  {c:>10}  ({pct:5.2f}%)  {bar}")
    return "\n".join(out_lines)


def load_action_slices(dataset_path: str) -> dict[str, tuple[int, int]]:
    """Read meta/modality.json and return {short_name: (start, end)} for action keys."""
    with open(os.path.join(dataset_path, "meta", "modality.json")) as f:
        mod = json.load(f)
    return {k: (v["start"], v["end"]) for k, v in mod["action"].items()}


def load_metadata_stats(
    dataset_path: str, slices: dict[str, tuple[int, int]]
) -> dict[str, dict[str, np.ndarray]]:
    """Read meta/stats.json. The schema stores flat per-column arrays
    (e.g. stats['action']['min'] is a 16-dim list), so we slice them per
    modality sub-key using the ranges from modality.json.

    Returns {short_name: {'min': np.ndarray(d,), 'q01': ..., ...}}.
    """
    with open(os.path.join(dataset_path, "meta", "stats.json")) as f:
        stats = json.load(f)
    action_stats = stats.get("action", {})
    # action_stats: {'min': [16], 'max': [16], 'mean': [16], 'std': [16], 'q01': [16], 'q99': [16]}
    flat = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in action_stats.items()}
    out: dict[str, dict[str, np.ndarray]] = {}
    for short, (s_idx, e_idx) in slices.items():
        out[short] = {fname: arr[s_idx:e_idx] for fname, arr in flat.items()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument("--data-config", required=True, type=str)
    parser.add_argument("--embodiment-tag", default="new_embodiment", type=str)
    parser.add_argument("--max-trajs", default=None, type=int, help="Only scan first N trajectories.")
    parser.add_argument("--gripper-close-threshold", default=0.5, type=float)
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    action_keys = list(data_config.action_keys)  # e.g. ["action.left_arm", ...]
    short_names = [k.split(".", 1)[1] for k in action_keys]

    slices = load_action_slices(args.dataset_path)
    meta_stats = load_metadata_stats(args.dataset_path, slices)

    # Verify slices exist for every action key in the data config.
    for short in short_names:
        assert short in slices, f"{short} not in meta/modality.json action keys: {list(slices)}"

    parquet_paths = sorted(glob.glob(os.path.join(args.dataset_path, "data", "*", "*.parquet")))
    assert parquet_paths, f"No parquet files found under {args.dataset_path}/data/*/*.parquet"
    if args.max_trajs is not None:
        parquet_paths = parquet_paths[: args.max_trajs]
    print(f"Scanning {len(parquet_paths)} parquet files from {args.dataset_path}/data/\n")

    # Metadata stats summary.
    print("=" * 80)
    print("Metadata action stats (from data/meta/stats.json)")
    print("=" * 80)
    for short in short_names:
        s = meta_stats.get(short, {})
        line = f"  action.{short}: "
        for fname in ("min", "max", "mean", "std", "q01", "q99"):
            if fname in s:
                arr = s[fname]
                if arr.size <= 4:
                    line += f"{fname}={np.array2string(arr, precision=4, suppress_small=True)}  "
                else:
                    line += f"{fname}=[{arr.min():+.4f}..{arr.max():+.4f}]  "
        print(line)
    print()

    # Stream all parquets, accumulate per-key values.
    collected: dict[str, list[np.ndarray]] = defaultdict(list)
    total_rows = 0
    for i, ppath in enumerate(parquet_paths):
        df = pd.read_parquet(ppath, columns=["action"])
        actions = np.stack(df["action"].values).astype(np.float64)  # (T, action_total_dim)
        total_rows += actions.shape[0]
        for short in short_names:
            s, e = slices[short]
            collected[short].append(actions[:, s:e])
        if (i + 1) % 20 == 0 or i == len(parquet_paths) - 1:
            print(f"  read {i + 1}/{len(parquet_paths)} parquets  (total rows so far: {total_rows})")
    print()

    print("=" * 80)
    print("Per-key raw action distribution and clipping diagnostics")
    print("=" * 80)

    summary_rows = []
    for short in short_names:
        full_key = f"action.{short}"
        vals2d = np.concatenate(collected[short], axis=0)  # (N, d)
        N, d = vals2d.shape
        values = vals2d.reshape(-1)

        print(f"\n{full_key}   (n_steps={N}, per-step dim={d})")
        print(
            f"  raw: min={values.min():+.4f}  max={values.max():+.4f}  "
            f"mean={values.mean():+.4f}  std={values.std():+.4f}"
        )

        s = meta_stats.get(short, {})
        for lo_name, hi_name in (("min", "max"), ("q01", "q99")):
            if lo_name not in s or hi_name not in s:
                continue
            lo_arr = s[lo_name]
            hi_arr = s[hi_name]
            if lo_arr.shape != (d,) or hi_arr.shape != (d,):
                print(f"  [{lo_name}, {hi_name}] shape mismatch: {lo_arr.shape} vs per-step dim {d}")
                continue
            below = (vals2d < lo_arr[None, :]).sum(axis=0)
            above = (vals2d > hi_arr[None, :]).sum(axis=0)
            tot_below = int(below.sum())
            tot_above = int(above.sum())
            denom = max(N * d, 1)
            pct_below = 100.0 * tot_below / denom
            pct_above = 100.0 * tot_above / denom
            print(
                f"  [{lo_name}, {hi_name}] clipping: below={tot_below} ({pct_below:.3f}%)  "
                f"above={tot_above} ({pct_above:.3f}%)"
            )
            if d > 1 and (tot_below > 0 or tot_above > 0):
                worst = int((below + above).argmax())
                print(
                    f"    worst dim {worst}: below={int(below[worst])}, "
                    f"above={int(above[worst])}, "
                    f"meta-range=[{lo_arr[worst]:+.4f}, {hi_arr[worst]:+.4f}], "
                    f"raw-range=[{vals2d[:, worst].min():+.4f}, {vals2d[:, worst].max():+.4f}]"
                )

        if "gripper" in short:
            thresh = args.gripper_close_threshold
            frac_closed = float((values > thresh).mean())
            frac_open = float((values < (1.0 - thresh)).mean())
            print(
                f"  gripper: frac(>{thresh})={frac_closed:.4f}  "
                f"frac(<{1 - thresh})={frac_open:.4f}"
            )
            print("  histogram (raw, 10 bins):")
            print(histogram_line(values, nbins=10))
            summary_rows.append((full_key, frac_closed))

    if summary_rows:
        print()
        print("=" * 80)
        print("Gripper summary")
        print("=" * 80)
        for full_key, frac in summary_rows:
            verdict = "OK" if frac > 0.05 else "WARN: gripper rarely commanded close in demos"
            print(f"  {full_key}: frac(>{args.gripper_close_threshold})={frac:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
