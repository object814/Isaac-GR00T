"""
Closed-loop evaluation of a finetuned GR00T policy on the CubeAssemble MuJoCo environment.

This script:
  1. Loads a finetuned GR00T checkpoint
  2. Instantiates the CubeAssembleEnvironment from bimanual_suite
  3. Runs a closed-loop rollout: observe → predict → act → repeat
  4. Reports success/failure and saves a video of the rollout

Usage:
python \
    scripts/frank_eval.py \
    --checkpoint-path outputs/frank_ft_N1.5_0406/checkpoint-40000/ \
    --data-config examples.frank.frank_config:FrankDataConfig \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --use-low-pass-filter \
    --seed 1 \
    --execute-steps 4
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import itertools
import pathlib
import time
from typing import cast

import cv2
import imageio
import numpy as np

from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.experiment.data_config import load_data_config

from bimanual_suite.mjc import CubeAssembleEnvironment  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching the Frank dataset (from data/meta/info.json)
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 144, 256
RAW_RENDER_SCALE = 2
CTRL_HZ = 20
TASK_DESCRIPTION = (
    "Stack the blue block onto the black block, then stack the orange block on top of the blue block."
)

# Arm-specific prompts, matching data_relabeled/meta/tasks.jsonl (task_index 1-4).
# Use --arm to condition the closed-loop rollout on one of these, or --task-description
# to pass an arbitrary string.
ARM_PROMPTS = {
    "generic": TASK_DESCRIPTION,
    "left": "Using the left arm to stack the blue block onto the black block, then stack the orange block on top of the blue block.",
    "right": "Using the right arm to stack the blue block onto the black block, then stack the orange block on top of the blue block.",
    "left_right": "Using the left arm to stack the blue block onto the black block, then using the right arm to stack the orange block on top of the blue block.",
    "right_left": "Using the right arm to stack the blue block onto the black block, then using the left arm to stack the orange block on top of the blue block.",
}


def resize_like_training(image: np.ndarray) -> np.ndarray:
    """Match bimanual_suite data-generation resize exactly (RGB -> BGR -> resize -> RGB)."""
    return cv2.cvtColor(
        cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (IMG_W, IMG_H)),
        cv2.COLOR_BGR2RGB,
    )


# ---------------------------------------------------------------------------
# Observation conversion: MuJoCo obs → GR00T policy input
# ---------------------------------------------------------------------------
def mjc_obs_to_gr00t(
    obs: dict,
    left_gripper_integral: float,
    right_gripper_integral: float,
    task_description: str = TASK_DESCRIPTION,
) -> dict:
    """Convert a raw MuJoCo observation dict to the GR00T policy format.

    The GR00T Frank state vector (46-dim) is:
        [0:7]   left_pos_cos   — cos(left 7 joint positions)
        [7:14]  left_pos_sin   — sin(left 7 joint positions)
        [14:21] right_pos_cos  — cos(right 7 joint positions)
        [21:28] right_pos_sin  — sin(right 7 joint positions)
        [28:30] left_gripper   — [left_gripper_pos, right_gripper_pos]
        [30:32] right_gripper  — [left_gripper_integral, right_gripper_integral]
        [32:39] left_arm       — raw left 7 joint positions
        [39:46] right_arm      — raw right 7 joint positions
    """
    left_joints = np.arctan2(np.sin(obs["left_pos"][:7]), np.cos(obs["left_pos"][:7]))
    right_joints = np.arctan2(np.sin(obs["right_pos"][:7]), np.cos(obs["right_pos"][:7]))

    left_grip_pos = obs["left_pos"][7]   # scalar
    right_grip_pos = obs["right_pos"][7]  # scalar
    overhead = resize_like_training(obs["overhead_camera"])
    left = resize_like_training(obs["left_camera"])
    right = resize_like_training(obs["right_camera"])

    observation = {
        "video.overhead": overhead[None].astype(np.uint8),  # (T=1,H,W,3)
        "video.left": left[None].astype(np.uint8),
        "video.right": right[None].astype(np.uint8),
        "state.left_pos_cos": np.cos(left_joints).reshape(1, 7).astype(np.float32),
        "state.left_pos_sin": np.sin(left_joints).reshape(1, 7).astype(np.float32),
        "state.right_pos_cos": np.cos(right_joints).reshape(1, 7).astype(np.float32),
        "state.right_pos_sin": np.sin(right_joints).reshape(1, 7).astype(np.float32),
        "state.left_gripper": np.array(
            [[left_grip_pos, right_grip_pos]], dtype=np.float32
        ),
        "state.right_gripper": np.array(
            [[left_gripper_integral, right_gripper_integral]], dtype=np.float32
        ),
        "state.left_arm": left_joints.reshape(1, 7).astype(np.float32),
        "state.right_arm": right_joints.reshape(1, 7).astype(np.float32),
        "annotation.human.task_description": [task_description],
    }
    return observation


# ---------------------------------------------------------------------------
# Action conversion: GR00T policy output → MuJoCo env action
# ---------------------------------------------------------------------------
def gr00t_action_to_mjc(action: dict, step_in_chunk: int = 0) -> np.ndarray:
    """Convert GR00T action dict to the 16-dim env action.

    GR00T outputs:
        left_arm      (1, 16, 7) — 7 joint targets
        right_arm     (1, 16, 7) — 7 joint targets
        left_gripper  (1, 16, 1) — left gripper command
        right_gripper (1, 16, 1) — right gripper command

    Env expects:
        action[0:7]   left joint positions
        action[7:14]  right joint positions
        action[14]    left gripper command
        action[15]    right gripper command
    """
    # =========================================================================
    # TEMPORARY REMAP for old checkpoint (trained with wrong modality split):
    #   old left_arm  learned [left_j1..7, right_j1]  (8-dim)
    #   old right_arm learned [right_j2..7, left_grip, right_grip] (8-dim)
    # Delete this block and uncomment the block below after retraining.
    # -------------------------------------------------------------------------
    # old_left = action["left_arm"][0, step_in_chunk]    # (8,) from old checkpoint
    # old_right = action["right_arm"][0, step_in_chunk]  # (8,) from old checkpoint
    # left_joints = old_left[0:7]                        # left_j1..7  — correct
    # right_joints = np.concatenate([
    #     old_left[7:8],                                 # right_j1    — was in left_arm slot 7
    #     old_right[0:6],                                # right_j2..7 — shifted by one
    # ])
    # left_grip = np.clip(old_right[6:7], 0.0, 1.0)     # left_grip   — was in right_arm slot 6
    # right_grip = np.clip(old_right[7:8], 0.0, 1.0)    # right_grip  — was in right_arm slot 7
    # =========================================================================

    # =========================================================================
    # CORRECT logic for retrained checkpoint (4 action keys). Uncomment after
    # retraining with the fixed modality config, and delete the block above.
    # -------------------------------------------------------------------------
    left_key = "action.left_arm" if "action.left_arm" in action else "left_arm"
    right_key = "action.right_arm" if "action.right_arm" in action else "right_arm"
    left_grip_key = "action.left_gripper" if "action.left_gripper" in action else "left_gripper"
    right_grip_key = (
        "action.right_gripper" if "action.right_gripper" in action else "right_gripper"
    )

    def chunk_step(x: np.ndarray) -> np.ndarray:
        if x.ndim == 2:
            return x[step_in_chunk]
        if x.ndim == 3:
            return x[0, step_in_chunk]
        raise ValueError(f"Unexpected action tensor shape: {x.shape}")

    left_joints = chunk_step(action[left_key])          # (7,)
    right_joints = chunk_step(action[right_key])        # (7,)
    left_grip = np.clip(np.atleast_1d(chunk_step(action[left_grip_key])), 0.0, 1.0)   # (1,)
    right_grip = np.clip(np.atleast_1d(chunk_step(action[right_grip_key])), 0.0, 1.0) # (1,)
    # =========================================================================

    return np.concatenate([
        left_joints,    # left arm joints
        right_joints,   # right arm joints
        left_grip,      # left gripper
        right_grip,     # right gripper
    ]).astype(np.float32)


def low_pass_filter_action(
    current_action: np.ndarray,
    previous_filtered_action: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Apply first-order low-pass filter (EMA) to one env action vector."""
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"low_pass_alpha must be in (0, 1], got {alpha}")
    if previous_filtered_action is None:
        return current_action
    return (alpha * current_action + (1.0 - alpha) * previous_filtered_action).astype(np.float32)


# ---------------------------------------------------------------------------
# Single rollout
# ---------------------------------------------------------------------------
def run_rollout(
    policy: Gr00tPolicy,
    seed: int,
    max_steps: int = 800,
    execute_steps: int = 8,
    use_low_pass_filter: bool = False,
    low_pass_alpha: float = 0.2,
    save_dir: pathlib.Path | None = None,
    run_name: str | None = None,
    task_description: str = TASK_DESCRIPTION,
) -> dict:
    """Run one closed-loop episode and return results."""
    print(f"  task prompt: {task_description!r}")
    modality_config = policy.get_modality_config()
    model_horizon = len(modality_config["action"].delta_indices)
    execute_steps = max(1, min(execute_steps, model_horizon))

    env = CubeAssembleEnvironment(
        seed=seed,
        render_height=IMG_H * RAW_RENDER_SCALE,
        render_width=IMG_W * RAW_RENDER_SCALE,
    )
    obs = env.reset()

    # Running gripper integrals (cumulative sum of gripper position, matching training data)
    left_gripper_integral = 0.0
    right_gripper_integral = 0.0

    frames = []  # for video recording
    frame_annotations = []  # per-frame debug info for overlay
    step = 0
    t_start = time.time()
    inference_times = []
    previous_filtered_action = None

    # --- inference-time state logging (compare against training distribution) ---
    state_keys_to_log = [
        "state.left_pos_cos", "state.left_pos_sin",
        "state.right_pos_cos", "state.right_pos_sin",
        "state.left_gripper", "state.right_gripper",
        "state.left_arm", "state.right_arm",
    ]
    eval_state_log: dict[str, list[np.ndarray]] = {k: [] for k in state_keys_to_log}
    first_frame_dumped = False

    while step < max_steps:
        # Accumulate gripper integrals
        left_gripper_integral += obs["left_pos"][7]
        right_gripper_integral += obs["right_pos"][7]

        # Convert MuJoCo observation to GR00T format
        gr00t_obs = mjc_obs_to_gr00t(obs, left_gripper_integral, right_gripper_integral, task_description)

        # Log raw (pre-normalization) state we feed to the policy.
        for sk in state_keys_to_log:
            if sk in gr00t_obs:
                eval_state_log[sk].append(np.asarray(gr00t_obs[sk]).reshape(-1).copy())

        # Dump first-step camera frames (exactly as fed to the policy) for side-by-side
        # comparison against the training video. PNG keeps it lossless.
        if not first_frame_dumped and save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            for cam_key in ("video.overhead", "video.left", "video.right"):
                frame_rgb = gr00t_obs[cam_key][0]  # (H, W, 3) uint8 RGB
                out_path = save_dir / f"eval_first_frame_seed{seed}_{cam_key.split('.', 1)[1]}.png"
                # cv2.imwrite expects BGR.
                cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                print(f"  dumped eval frame: {out_path}  shape={frame_rgb.shape}  dtype={frame_rgb.dtype}")
            first_frame_dumped = True

        # Predict action chunk
        t0 = time.time()
        action_chunk = policy.get_action(gr00t_obs)
        inference_times.append(time.time() - t0)

        # Log chunk diagnostics to console
        if step % (execute_steps * 10) == 0:  # every 10 replans
            print(
                f"  [step={step:4d}] integrals=({left_gripper_integral:.2f}, {right_gripper_integral:.2f})"
            )

        # Extract full-chunk gripper profile for diagnostics
        chunk_lg_key = "action.left_gripper" if "action.left_gripper" in action_chunk else "left_gripper"
        chunk_rg_key = "action.right_gripper" if "action.right_gripper" in action_chunk else "right_gripper"
        chunk_lg = action_chunk[chunk_lg_key].squeeze()  # (H,) or (H,1) → (H,)
        chunk_rg = action_chunk[chunk_rg_key].squeeze()
        if chunk_lg.ndim > 1:
            chunk_lg = chunk_lg[:, 0]
            chunk_rg = chunk_rg[:, 0]
        chunk_lg_list = chunk_lg.tolist() if hasattr(chunk_lg, 'tolist') else [float(chunk_lg)]
        chunk_rg_list = chunk_rg.tolist() if hasattr(chunk_rg, 'tolist') else [float(chunk_rg)]
        chunk_lg_max = max(chunk_lg_list)
        chunk_rg_max = max(chunk_rg_list)
        # Find first step in chunk where gripper action > 0.5
        chunk_lg_first_close = next((i for i, v in enumerate(chunk_lg_list) if v > 0.5), -1)
        chunk_rg_first_close = next((i for i, v in enumerate(chunk_rg_list) if v > 0.5), -1)

        if step % (execute_steps * 10) == 0:
            chunk_lg_min = min(chunk_lg_list)
            chunk_rg_min = min(chunk_rg_list)
            chunk_lg_mean = float(np.mean(chunk_lg_list))
            chunk_rg_mean = float(np.mean(chunk_rg_list))
            chunk_lg_neg = sum(1 for v in chunk_lg_list if v < 0.0)
            chunk_rg_neg = sum(1 for v in chunk_rg_list if v < 0.0)
            print(
                f"           RAW (pre-clip) L grip min/mean/max=({chunk_lg_min:+.4f}/{chunk_lg_mean:+.4f}/{chunk_lg_max:+.4f}) "
                f"neg={chunk_lg_neg}/{len(chunk_lg_list)}"
            )
            print(
                f"           RAW (pre-clip) R grip min/mean/max=({chunk_rg_min:+.4f}/{chunk_rg_mean:+.4f}/{chunk_rg_max:+.4f}) "
                f"neg={chunk_rg_neg}/{len(chunk_rg_list)}"
            )
            print(
                f"           1st close(>0.5) step=({chunk_lg_first_close}, {chunk_rg_first_close})"
            )

        # Execute the full action chunk (or until max_steps)
        for t in range(execute_steps):
            if step >= max_steps:
                break

            raw_env_action = gr00t_action_to_mjc(action_chunk, step_in_chunk=t)
            if use_low_pass_filter:
                smoothed = low_pass_filter_action(raw_env_action[:14], previous_filtered_action, low_pass_alpha)
                previous_filtered_action = smoothed
                env_action = np.concatenate([smoothed, raw_env_action[14:16]]).astype(np.float32)
            else:
                env_action = raw_env_action
            obs = env.step(env_action)

            # Record frame for video
            if save_dir is not None:
                frames.append(obs["user_camera"].copy())
                # Store per-frame debug annotations
                left_grip_key = "action.left_gripper" if "action.left_gripper" in action_chunk else "left_gripper"
                right_grip_key = "action.right_gripper" if "action.right_gripper" in action_chunk else "right_gripper"
                lg_action = action_chunk[left_grip_key]
                rg_action = action_chunk[right_grip_key]
                # Extract scalar for current step
                if lg_action.ndim == 3:
                    lg_val = float(lg_action[0, t, 0]) if lg_action.shape[-1] >= 1 else float(lg_action[0, t])
                    rg_val = float(rg_action[0, t, 0]) if rg_action.shape[-1] >= 1 else float(rg_action[0, t])
                elif lg_action.ndim == 2:
                    lg_val = float(lg_action[t, 0]) if lg_action.shape[-1] >= 1 else float(lg_action[t])
                    rg_val = float(rg_action[t, 0]) if rg_action.shape[-1] >= 1 else float(rg_action[t])
                else:
                    lg_val = float(lg_action)
                    rg_val = float(rg_action)
                frame_annotations.append({
                    "left_integral": left_gripper_integral,
                    "right_integral": right_gripper_integral,
                    "left_grip_pos": float(obs["left_pos"][7]),
                    "right_grip_pos": float(obs["right_pos"][7]),
                    "left_grip_action": lg_val,
                    "right_grip_action": rg_val,
                    "step": step,
                    "chunk_step": t,
                    "chunk_lg_max": chunk_lg_max,
                    "chunk_rg_max": chunk_rg_max,
                    "chunk_lg_first_close": chunk_lg_first_close,
                    "chunk_rg_first_close": chunk_rg_first_close,
                    "chunk_len": len(chunk_lg_list),
                })

            step += 1

            # Update integrals for intermediate steps within the chunk
            if t < execute_steps - 1:
                left_gripper_integral += obs["left_pos"][7]
                right_gripper_integral += obs["right_pos"][7]

    wall_time = time.time() - t_start
    success = env.success()
    env.close()

    # Save video
    if save_dir is not None and frames:
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = "success" if success else "failure"
        prefix = f"{run_name}_es{execute_steps}" if run_name else f"es{execute_steps}"
        video_path = save_dir / f"{prefix}_seed_{seed}_{tag}.mp4"
        writer = imageio.get_writer(str(video_path), fps=CTRL_HZ, codec="mpeg4", quality=8)
        for i, frame in enumerate(frames):
            annotated = frame.astype(np.uint8).copy()
            if i < len(frame_annotations):
                ann = frame_annotations[i]
                lg_close_str = str(ann['chunk_lg_first_close']) if ann['chunk_lg_first_close'] >= 0 else "NEVER"
                rg_close_str = str(ann['chunk_rg_first_close']) if ann['chunk_rg_first_close'] >= 0 else "NEVER"
                lines = [
                    f"step: {ann['step']}  chunk_t: {ann['chunk_step']}",
                    f"L grip integral: {ann['left_integral']:.3f}",
                    f"R grip integral: {ann['right_integral']:.3f}",
                    f"L grip pos: {ann['left_grip_pos']:.4f}  action: {ann['left_grip_action']:.4f}",
                    f"R grip pos: {ann['right_grip_pos']:.4f}  action: {ann['right_grip_action']:.4f}",
                    f"--- full chunk (all {ann['chunk_len']} steps) ---",
                    f"L chunk max: {ann['chunk_lg_max']:.4f}  1st close(>0.5): {lg_close_str}",
                    f"R chunk max: {ann['chunk_rg_max']:.4f}  1st close(>0.5): {rg_close_str}",
                ]
                y0 = 20
                for j, line in enumerate(lines):
                    y = y0 + j * 18
                    # Black outline for readability
                    cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(annotated, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            writer.append_data(annotated)
        writer.close()
        print(f"  Video saved to {video_path}")

    # ---- dump inference-time state distribution for OOD comparison ----
    print("\n" + "=" * 80)
    print(f"INFERENCE state distribution (seed={seed}, steps={step})")
    print("=" * 80)
    n_dump = min(50, step)
    for sk, frames_list in eval_state_log.items():
        if not frames_list:
            continue
        arr = np.stack(frames_list, axis=0)  # (T, d)
        d = arr.shape[1]
        print(
            f"\n{sk}  (n={arr.shape[0]}, dim={d})  "
            f"global min={arr.min():+.4f}  max={arr.max():+.4f}  "
            f"mean={arr.mean():+.4f}  std={arr.std():+.4f}"
        )
        if d <= 8:
            for j in range(d):
                col = arr[:, j]
                print(
                    f"  dim {j}: min={col.min():+.4f}  max={col.max():+.4f}  "
                    f"mean={col.mean():+.4f}  std={col.std():+.4f}"
                )

    # First-N steps trace for both gripper state channels (the suspicious integrals).
    for sk in ("state.left_gripper", "state.right_gripper"):
        if not eval_state_log[sk]:
            continue
        arr = np.stack(eval_state_log[sk][:n_dump], axis=0)
        print(f"\nFirst {arr.shape[0]} steps of {sk} (inference)")
        for t in range(arr.shape[0]):
            row = "  ".join(f"{v:+.4f}" for v in arr[t])
            print(f"  t={t:3d}: {row}")
    print("=" * 80 + "\n")

    avg_inference = np.mean(inference_times) if inference_times else 0
    return {
        "seed": seed,
        "success": success,
        "steps": step,
        "sim_seconds": step / CTRL_HZ,
        "wall_seconds": wall_time,
        "avg_inference_ms": avg_inference * 1000,
        "num_inference_calls": len(inference_times),
        "model_action_horizon": model_horizon,
        "execute_steps": execute_steps,
        "use_low_pass_filter": use_low_pass_filter,
        "low_pass_alpha": low_pass_alpha,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Frank GR00T closed-loop evaluation")
    parser.add_argument(
        "--checkpoint-path", type=str, required=True,
        help="Path to the finetuned checkpoint (e.g., outputs/frank_ft_full/checkpoint-10000)",
    )
    parser.add_argument(
        "--data-config", type=str, default="examples.frank.frank_config:FrankDataConfig",
        help=(
            "Data config name or external module path in module:ClassName format "
            "(default: examples.frank.frank_config:FrankDataConfig)"
        ),
    )
    parser.add_argument(
        "--embodiment-tag", type=str, default="new_embodiment",
        choices=sorted(EMBODIMENT_TAG_MAPPING.keys()),
        help="Embodiment tag for loading metadata/transforms (default: new_embodiment)",
    )
    parser.add_argument(
        "--denoising-steps", type=int, default=4,
        help="Number of denoising steps for action head inference (default: 4)",
    )
    parser.add_argument(
        "--seed", type=int, nargs="+", default=[42],
        help="Environment seed(s) to evaluate (e.g., --seed 0 1 2 3)",
    )
    parser.add_argument(
        "--sweep-seed", type=int, default=42,
        help="Fixed seed used for each parameter setup in sweep mode (default: 42)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1500,
        help="Maximum env steps per episode (default: 1500 = 75s at 20Hz)",
    )
    parser.add_argument(
        "--execute-steps", type=int, default=16,
        help="Number of predicted steps to execute before replanning (default: 16)",
    )
    parser.add_argument(
        "--use-low-pass-filter", action="store_true",
        help="Enable first-order low-pass filtering on deployed env actions",
    )
    parser.add_argument(
        "--low-pass-alpha", type=float, default=0.3,
        help="Low-pass alpha in (0, 1]; smaller means stronger smoothing (default: 0.2)",
    )
    parser.add_argument(
        "--sweep-execute-steps", type=int, nargs="+", default=None,
        help="Sweep values for execute-steps (e.g., --sweep-execute-steps 4 8 12 16)",
    )
    parser.add_argument(
        "--sweep-max-steps", type=int, nargs="+", default=None,
        help="Sweep values for max-steps (e.g., --sweep-max-steps 600 900 1200)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save rollout videos (default: outputs/frank_eval/)",
    )
    parser.add_argument(
        "--arm", type=str, default="generic", choices=sorted(ARM_PROMPTS.keys()),
        help="Which arm-specific prompt to condition on (matches data_relabeled tasks.jsonl)",
    )
    parser.add_argument(
        "--task-description", type=str, default=None,
        help="Override the language prompt with an arbitrary string (takes precedence over --arm)",
    )
    args = parser.parse_args()

    task_description = args.task_description if args.task_description else ARM_PROMPTS[args.arm]

    save_dir = pathlib.Path(args.save_dir) if args.save_dir else (
        pathlib.Path(args.checkpoint_path).parent / "eval_videos"
    )

    # Build modality config/transforms to match N1.5 policy loading path.
    data_config = load_data_config(args.data_config)
    modality_config = data_config.modality_config()
    modality_transform = cast(ComposedModalityTransform, data_config.transform())

    # Load model
    print(f"Loading model from {args.checkpoint_path} ...")
    policy = Gr00tPolicy(
        model_path=args.checkpoint_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    print("Model loaded.\n")

    # ---- DIAGNOSTIC: dump the gripper stats baked into THIS checkpoint's metadata ----
    # These are the stats used to un-normalize the gripper at inference. If they don't
    # match the dataset the policy was trained against, the gripper command will be
    # offset/scaled wrong even if the model itself is perfect.
    try:
        action_meta = policy.metadata.statistics.action
        print("=" * 60)
        print(f"Checkpoint metadata source: {args.checkpoint_path}/experiment_cfg/metadata.json")
        print("Action stats baked into this checkpoint (used at inference):")
        for key in ("left_gripper", "right_gripper", "left_arm", "right_arm"):
            if key in action_meta:
                s = action_meta[key]
                fields = {}
                for fname in ("min", "max", "q01", "q99", "mean", "std"):
                    fv = getattr(s, fname, None)
                    if fv is not None:
                        arr = np.asarray(fv)
                        fields[fname] = (float(arr.min()), float(arr.max()))
                print(f"  action.{key}: {fields}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"(could not dump action stats: {e})\n")

    sweep_active = args.sweep_execute_steps is not None or args.sweep_max_steps is not None

    if sweep_active:
        if args.seed != [args.sweep_seed]:
            print(
                f"Sweep mode uses fixed seed={args.sweep_seed}; ignoring --seed values {args.seed}."
            )

        execute_steps_values = (
            sorted(set(args.sweep_execute_steps))
            if args.sweep_execute_steps is not None
            else [args.execute_steps]
        )
        max_steps_values = (
            sorted(set(args.sweep_max_steps))
            if args.sweep_max_steps is not None
            else [args.max_steps]
        )

        sweep_configs = list(itertools.product(execute_steps_values, max_steps_values))
        print("=" * 60)
        print(
            f"Sweep mode: {len(sweep_configs)} setup(s), fixed seed={args.sweep_seed}, "
            f"execute_steps={execute_steps_values}, max_steps={max_steps_values}"
        )
        print("=" * 60)

        sweep_results = []
        for idx, (execute_steps, max_steps) in enumerate(sweep_configs, start=1):
            run_name = f"exec{execute_steps}_max{max_steps}"
            print(f"--- Sweep {idx}/{len(sweep_configs)} | {run_name} ---")
            result = run_rollout(
                policy,
                seed=args.sweep_seed,
                max_steps=max_steps,
                execute_steps=execute_steps,
                use_low_pass_filter=args.use_low_pass_filter,
                low_pass_alpha=args.low_pass_alpha,
                save_dir=save_dir,
                run_name=run_name,
                task_description=task_description,
            )
            status = "SUCCESS" if result["success"] else "FAILURE"
            print(
                f"  {status}  |  {result['steps']} steps  |  "
                f"{result['sim_seconds']:.1f}s sim  |  "
                f"{result['wall_seconds']:.1f}s wall  |  "
                f"{result['avg_inference_ms']:.1f}ms/inference  |  "
                f"exec/model horizon={result['execute_steps']}/{result['model_action_horizon']}\n"
            )
            sweep_results.append(
                {
                    **result,
                    "config_execute_steps": execute_steps,
                    "config_max_steps": max_steps,
                    "run_name": run_name,
                }
            )

        print("=" * 90)
        print("Sweep Summary (seed fixed)")
        print("run_name               success  execute  max_steps  steps  sim_s   wall_s   inf_ms")
        for r in sweep_results:
            print(
                f"{r['run_name']:<22} {str(r['success']):<7}  {r['config_execute_steps']:<7} "
                f"{r['config_max_steps']:<9} {r['steps']:<5} {r['sim_seconds']:<7.1f} "
                f"{r['wall_seconds']:<8.1f} {r['avg_inference_ms']:<7.1f}"
            )

        # Prefer successful setups, then lower wall time, then lower inference time.
        best = sorted(
            sweep_results,
            key=lambda r: (
                not r["success"],
                r["wall_seconds"],
                r["avg_inference_ms"],
            ),
        )[0]
        print("-" * 90)
        print(
            f"Best setup: {best['run_name']} | success={best['success']} | "
            f"steps={best['steps']} | wall={best['wall_seconds']:.1f}s | "
            f"inf={best['avg_inference_ms']:.1f}ms"
        )
        print("=" * 90)
        return

    # Run rollouts
    results = []
    for seed in args.seed:
        print(f"--- Seed {seed} ---")
        result = run_rollout(
            policy,
            seed=seed,
            max_steps=args.max_steps,
            execute_steps=args.execute_steps,
            use_low_pass_filter=args.use_low_pass_filter,
            low_pass_alpha=args.low_pass_alpha,
            save_dir=save_dir,
            run_name=args.arm,
            task_description=task_description,
        )
        status = "SUCCESS" if result["success"] else "FAILURE"
        print(f"  {status}  |  {result['steps']} steps  |  "
              f"{result['sim_seconds']:.1f}s sim  |  "
              f"{result['wall_seconds']:.1f}s wall  |  "
              f"{result['avg_inference_ms']:.1f}ms/inference  |  "
              f"exec/model horizon={result['execute_steps']}/{result['model_action_horizon']}\n")
        results.append(result)

    # Summary
    n_success = sum(r["success"] for r in results)
    n_total = len(results)
    print("=" * 60)
    print(f"Success rate: {n_success}/{n_total} ({100 * n_success / n_total:.1f}%)")
    if results:
        avg_inf = np.mean([r["avg_inference_ms"] for r in results])
        print(f"Avg inference time: {avg_inf:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()