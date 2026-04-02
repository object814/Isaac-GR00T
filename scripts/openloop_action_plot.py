"""
Open-loop action prediction sanity check for the finetuned Frank GR00T policy.

For each requested episode:
  1. Load observations (video + state) step-by-step from the dataset.
  2. Feed them through the trained model to get predicted action chunks.
  3. Plot each action dimension (predicted vs ground truth) over time.

If training converged, predicted and ground-truth curves should track closely.

Usage:
    uv run python scripts/openloop_action_plot.py \
        --checkpoint-path /Isaac-GR00T/outputs/frank_ft_full_0327/checkpoint-10000 \
        --dataset-path /Isaac-GR00T/data \
        --episode 0

    # Multiple episodes, custom output dir:
    uv run python scripts/openloop_action_plot.py \
        --checkpoint-path /Isaac-GR00T/outputs/frank_ft_full_0327/checkpoint-10000 \
        --dataset-path /Isaac-GR00T/data \
        --episode 0 1 2 \
        --steps 200 \
        --action-horizon 16 \
        --save-dir outputs/openloop_plots
"""

import argparse
import pathlib
import sys
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Register the Frank modality config before loading the model.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(_REPO_ROOT / "examples" / "frank"))
import frank_config  # noqa: F401, E402  — registers NEW_EMBODIMENT config

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Axis labels for the Frank action space
# ---------------------------------------------------------------------------
# left_arm (7) + right_arm (7) + left_gripper (1) + right_gripper (1) = 16
_ACTION_LABELS = (
    [f"left_arm j{i}" for i in range(7)]
    + [f"right_arm j{i}" for i in range(7)]
    + ["left_gripper"]
    + ["right_gripper"]
)

# Separator indices (cumulative) so we can draw group dividers on the plots
_ACTION_KEY_DIMS = {"left_arm": 7, "right_arm": 7, "left_gripper": 1, "right_gripper": 1}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_obs(data_point, modality_configs: dict) -> dict:
    """Convert extract_step_data output to Gr00tPolicy input format."""
    obs = {}
    for k, v in data_point.states.items():
        obs[f"state.{k}"] = v          # (T, D)
    for k, v in data_point.images.items():
        obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
    for lang_key in modality_configs["language"].modality_keys:
        obs[lang_key] = data_point.text

    # Nest into {modality: {key: array}} with batch dim
    nested = {}
    for modality in ["video", "state", "language"]:
        nested[modality] = {}
        for key in modality_configs[modality].modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            arr = obs[parsed_key]
            if isinstance(arr, str):
                nested[modality][key] = [[arr]]
            else:
                nested[modality][key] = arr[None, :]   # add batch dim
    return nested


def _extract_actions_from_df(traj: pd.DataFrame, action_keys: list[str]) -> np.ndarray:
    """Stack all per-key action arrays from a trajectory DataFrame."""
    parts = []
    for key in action_keys:
        col = f"action.{key}"
        parts.append(np.vstack([np.atleast_1d(a) for a in traj[col]]))
    return np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def evaluate_episode(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    episode_id: int,
    steps: int,
    action_horizon: int,
    save_dir: pathlib.Path,
) -> dict:
    traj = loader[episode_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    print(f"  Episode {episode_id}: {traj_length} steps in dataset, evaluating {actual_steps}")

    action_keys = loader.modality_configs["action"].modality_keys

    # Drop action from modality_configs so extract_step_data doesn't need it
    mc_no_action = deepcopy(loader.modality_configs)
    mc_no_action.pop("action")

    pred_actions = []
    for step_start in range(0, actual_steps, action_horizon):
        data_point = extract_step_data(
            traj, step_start, mc_no_action, EmbodimentTag.NEW_EMBODIMENT
        )
        nested_obs = _build_obs(data_point, loader.modality_configs)
        action_chunk, _ = policy.get_action(nested_obs)

        steps_left = actual_steps - step_start
        for j in range(min(action_horizon, steps_left)):
            pred = np.concatenate(
                [np.atleast_1d(action_chunk[key][0, j]) for key in action_keys], axis=0
            )
            pred_actions.append(pred)

    pred_actions = np.array(pred_actions)                           # (T, D)
    gt_actions = _extract_actions_from_df(traj, action_keys)[:actual_steps]  # (T, D)
    assert pred_actions.shape == gt_actions.shape, (
        f"Shape mismatch: pred={pred_actions.shape}, gt={gt_actions.shape}"
    )

    mse = float(np.mean((gt_actions - pred_actions) ** 2))
    mae = float(np.mean(np.abs(gt_actions - pred_actions)))
    print(f"  MSE={mse:.6f}  MAE={mae:.6f}")

    # -----------------------------------------------------------------------
    # Plot: one row per action dimension, grouped by key
    # -----------------------------------------------------------------------
    action_dim = gt_actions.shape[1]
    labels = _ACTION_LABELS[:action_dim]

    fig, axes = plt.subplots(nrows=action_dim, ncols=1, figsize=(12, 3 * action_dim))
    if action_dim == 1:
        axes = [axes]

    # Compute group boundary x-positions for shading
    group_boundaries = []
    cursor = 0
    for key in action_keys:
        d = _ACTION_KEY_DIMS.get(key, 1)
        group_boundaries.append((key, cursor, cursor + d))
        cursor += d

    time_axis = np.arange(actual_steps)
    inference_points = np.arange(0, actual_steps, action_horizon)

    for dim_i, ax in enumerate(axes):
        ax.plot(time_axis, gt_actions[:, dim_i], color="tab:blue", lw=1.5, label="ground truth")
        ax.plot(time_axis, pred_actions[:, dim_i], color="tab:orange", lw=1.5,
                linestyle="--", label="predicted")
        # Mark inference trigger points
        ax.scatter(
            inference_points,
            gt_actions[inference_points, dim_i],
            color="red", s=20, zorder=5, label="inference step" if dim_i == 0 else None,
        )
        ax.set_ylabel(labels[dim_i], fontsize=9)
        ax.set_xlim(0, actual_steps - 1)
        ax.grid(True, alpha=0.3)
        if dim_i == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Group annotations on right y-axis
    for key, start, end in group_boundaries:
        mid = (start + end - 1) / 2.0
        if 0 <= int(mid) < len(axes):
            axes[int(mid)].set_title(f"[{key}]  dim {start}–{end - 1}", fontsize=9, loc="right")

    axes[-1].set_xlabel("Time step")
    fig.suptitle(
        f"Open-loop action prediction — Episode {episode_id}\n"
        f"MSE={mse:.6f}  MAE={mae:.6f}  (action_horizon={action_horizon})",
        fontsize=12,
    )
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir / f"episode_{episode_id:04d}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"  Plot saved → {plot_path}")

    return {"episode_id": episode_id, "mse": mse, "mae": mae, "plot": str(plot_path)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Open-loop action sanity check for Frank GR00T")
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to finetuned checkpoint (e.g. outputs/frank_ft_full_0327/checkpoint-10000)",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="/data/engs-a2i/catz0908/Isaac-GR00T/data",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--episode", type=int, nargs="+", default=[0],
        help="Episode index(es) to evaluate (default: 0)",
    )
    parser.add_argument(
        "--steps", type=int, default=300,
        help="Max timesteps to evaluate per episode (default: 300)",
    )
    parser.add_argument(
        "--action-horizon", type=int, default=16,
        help="Number of predicted steps to unroll before re-querying the model (default: 16)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Inference device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save plots (default: <checkpoint_dir>/openloop_plots)",
    )
    args = parser.parse_args()

    save_dir = (
        pathlib.Path(args.save_dir)
        if args.save_dir
        else pathlib.Path(args.checkpoint_path).parent / "openloop_plots"
    )

    # Load model
    print(f"Loading model from {args.checkpoint_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint_path,
        device=args.device,
    )
    modality_config = policy.get_modality_config()
    model_horizon = len(modality_config["action"].delta_indices)
    if args.action_horizon > model_horizon:
        print(
            f"Warning: --action-horizon {args.action_horizon} > model horizon {model_horizon}; "
            f"clamping to {model_horizon}."
        )
        args.action_horizon = model_horizon
    print(f"Model loaded. Action horizon: {model_horizon}\n")

    # Load dataset
    loader = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality_config,
        video_backend="torchcodec",
        video_backend_kwargs=None,
    )
    print(f"Dataset: {len(loader)} episodes at {args.dataset_path}\n")

    # Evaluate
    results = []
    for ep_id in args.episode:
        if ep_id >= len(loader):
            print(f"Episode {ep_id} out of range (dataset has {len(loader)}), skipping.")
            continue
        print(f"=== Episode {ep_id} ===")
        r = evaluate_episode(
            policy, loader, ep_id,
            steps=args.steps,
            action_horizon=args.action_horizon,
            save_dir=save_dir,
        )
        results.append(r)
        print()

    # Summary
    if results:
        avg_mse = np.mean([r["mse"] for r in results])
        avg_mae = np.mean([r["mae"] for r in results])
        print("=" * 50)
        print(f"Evaluated {len(results)} episode(s)")
        print(f"  Avg MSE: {avg_mse:.6f}")
        print(f"  Avg MAE: {avg_mae:.6f}")
        print(f"  Plots:   {save_dir}/")
        print("=" * 50)


if __name__ == "__main__":
    main()
