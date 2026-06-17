# SPDX-License-Identifier: Apache-2.0
"""
Sanity check: verify that the action normalization in our custom data config is invertible.

For one raw episode we run:
    raw action  ->  GR00T normalize (apply)  ->  GR00T un-normalize (unapply)
and compare against the raw input. Recovery should be near-perfect (down to fp32
roundoff plus any clipping the normalizer applies).

Usage:
    python scripts/check_action_roundtrip.py \
        --dataset-path data/ \
        --data-config examples.frank.frank_config:FrankDataConfig \
        --embodiment-tag new_embodiment \
        --traj-id 0 --num-steps 32
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.experiment.data_config import load_data_config


def build_action_only_transform(data_config) -> ComposedModalityTransform:
    """
    Reproduce only the action normalization pipeline from a data config:
    StateActionToTensor -> StateActionTransform -> ConcatTransform.

    GR00TTransform (the model-side padding/Eagle processor) is intentionally excluded:
    it does not normalize, and its apply() runs heavy video processing we don't need
    for a normalization round-trip check.
    """
    full = data_config.transform()
    action_keys = data_config.action_keys

    to_tensor = None
    state_action = None
    concat = None
    for t in full.transforms:
        if isinstance(t, StateActionToTensor) and set(t.apply_to) == set(action_keys):
            to_tensor = t
        elif isinstance(t, StateActionTransform) and set(t.apply_to) == set(action_keys):
            state_action = t
        elif isinstance(t, ConcatTransform):
            concat = t

    assert to_tensor is not None, "Could not find action StateActionToTensor in data config"
    assert state_action is not None, "Could not find action StateActionTransform in data config"
    assert concat is not None, "Could not find ConcatTransform in data config"

    # ConcatTransform also touches state/video; we strip those so only the action path runs.
    action_only_concat = ConcatTransform(
        video_concat_order=[],
        state_concat_order=None,
        action_concat_order=concat.action_concat_order,
    )

    return ComposedModalityTransform(transforms=[to_tensor, state_action, action_only_concat])


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def report_per_key(raw: dict, recovered: dict, action_keys: list[str]) -> bool:
    """Print per-key max-abs error and a short stats line. Returns True iff all close."""
    all_ok = True
    print("\nper-key round-trip error (max-abs and mean-abs):")
    print(f"{'key':<32} {'shape':<14} {'max_abs':>12} {'mean_abs':>12} {'raw_min':>10} {'raw_max':>10}")
    for key in action_keys:
        r = to_numpy(raw[key])
        u = to_numpy(recovered[key])
        assert r.shape == u.shape, f"shape mismatch for {key}: {r.shape=} vs {u.shape=}"
        diff = np.abs(r.astype(np.float64) - u.astype(np.float64))
        max_abs = float(diff.max()) if diff.size else 0.0
        mean_abs = float(diff.mean()) if diff.size else 0.0
        ok = np.allclose(r, u, atol=1e-5, rtol=1e-4)
        all_ok = all_ok and ok
        flag = "" if ok else "  <-- FAIL"
        print(
            f"{key:<32} {str(tuple(r.shape)):<14} {max_abs:>12.6g} {mean_abs:>12.6g} "
            f"{float(r.min()):>10.4f} {float(r.max()):>10.4f}{flag}"
        )
    return all_ok


def check_step(
    transform: ComposedModalityTransform,
    raw_step: dict,
    action_keys: list[str],
) -> bool:
    """Round-trip one step's action dict. raw_step contains raw per-key actions (np arrays)."""
    # Keep a pristine copy of raw inputs (transform.apply mutates).
    raw_for_compare = {k: np.array(raw_step[k], copy=True) for k in action_keys}

    forward_input = {k: np.array(raw_step[k], copy=True) for k in action_keys}
    normalized = transform.apply(forward_input)

    assert "action" in normalized, (
        f"Expected 'action' key after ConcatTransform, got {list(normalized.keys())}"
    )
    norm_action = normalized["action"]
    assert isinstance(norm_action, torch.Tensor)

    recovered = transform.unapply({"action": norm_action.clone()})
    return report_per_key(raw_for_compare, recovered, action_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True, type=str)
    parser.add_argument(
        "--data-config",
        required=True,
        type=str,
        help="e.g. 'examples.frank.frank_config:FrankDataConfig'",
    )
    parser.add_argument("--embodiment-tag", default="new_embodiment", type=str)
    parser.add_argument("--traj-id", default=0, type=int)
    parser.add_argument("--num-steps", default=32, type=int)
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    action_keys = list(data_config.action_keys)
    print(f"action_keys: {action_keys}")
    print(f"action_indices (horizon): len={len(data_config.action_indices)}")

    # Dataset with NO transforms — we want raw actions per-key.
    raw_dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=data_config.modality_config(),
        transforms=None,
        embodiment_tag=args.embodiment_tag,
        video_backend="torchcodec",
    )
    print(f"loaded dataset, n_steps={len(raw_dataset)}, n_trajs={len(raw_dataset.trajectory_lengths)}")

    transform = build_action_only_transform(data_config)
    transform.set_metadata(raw_dataset.metadata)
    transform.eval()

    print("\nnormalization modes (action keys):")
    for t in transform.transforms:
        if isinstance(t, StateActionTransform):
            for k, mode in t.normalization_modes.items():
                stats = t.normalization_statistics.get(k, {})
                stat_summary = {
                    sk: (float(to_numpy(sv).min()), float(to_numpy(sv).max()))
                    for sk, sv in stats.items()
                }
                print(f"  {k}: mode={mode}  stats(min/max per field)={stat_summary}")

    n_steps = min(args.num_steps, raw_dataset.trajectory_lengths[args.traj_id])
    print(f"\nrunning round-trip on traj {args.traj_id}, first {n_steps} steps")

    all_ok = True
    worst_per_key: dict[str, float] = {k: 0.0 for k in action_keys}
    for step in range(n_steps):
        raw_step = raw_dataset.get_step_data(args.traj_id, step)
        present = [k for k in action_keys if k in raw_step]
        if len(present) != len(action_keys):
            missing = set(action_keys) - set(present)
            raise RuntimeError(f"missing action keys at step {step}: {missing}")

        raw_for_compare = {k: np.array(raw_step[k], copy=True) for k in action_keys}
        forward_input = {k: np.array(raw_step[k], copy=True) for k in action_keys}
        normalized = transform.apply(forward_input)
        recovered = transform.unapply({"action": normalized["action"].clone()})

        for k in action_keys:
            r = to_numpy(raw_for_compare[k]).astype(np.float64)
            u = to_numpy(recovered[k]).astype(np.float64)
            diff = float(np.abs(r - u).max()) if r.size else 0.0
            if diff > worst_per_key[k]:
                worst_per_key[k] = diff
            if not np.allclose(r, u, atol=1e-5, rtol=1e-4):
                all_ok = False

        if step == 0:
            print("\n--- step 0 detail ---")
            report_per_key(raw_for_compare, recovered, action_keys)

    print("\n--- worst max-abs error per key across all steps ---")
    for k, v in worst_per_key.items():
        flag = "" if v <= 1e-4 else "  <-- LARGE"
        print(f"  {k:<32} {v:>12.6g}{flag}")

    if all_ok:
        print("\nRESULT: round-trip recovers raw actions to fp32 precision. Normalization OK.")
    else:
        print("\nRESULT: round-trip FAILED for at least one key/step. See per-key errors above.")
        print("Common causes: q99/min_max clipping (raw outside [q01, q99] or [min, max]),")
        print("  dtype downcasting, or stats mismatch between train and current metadata.")


if __name__ == "__main__":
    main()
