"""
Closed-loop evaluation of a finetuned GR00T policy on the CubeAssemble MuJoCo environment.

This script:
  1. Loads a finetuned GR00T checkpoint
  2. Instantiates the CubeAssembleEnvironment from bimanual_suite
  3. Runs a closed-loop rollout: observe → predict → act → repeat
  4. Reports success/failure and saves a video of the rollout

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/frank_eval.py \
        --checkpoint-path /Isaac-GR00T/outputs/frank_ft_full/checkpoint-10000 \
        --seed 42

    # Evaluate across multiple seeds:
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/frank_eval.py \
        --checkpoint-path /Isaac-GR00T/outputs/frank_ft_full/checkpoint-10000 \
        --seed 0 1 2 3 4 5 6 7 8 9
"""

import os

os.environ["MUJOCO_GL"] = "egl"

import argparse
import itertools
import pathlib
import sys
import time

import cv2
import imageio
import numpy as np
import torch

# Register the Frank modality config before loading the model.
sys.path.append(
    str(pathlib.Path(__file__).resolve().parent.parent / "examples" / "frank")
)
import frank_config  # noqa: F401, E402

from bimanual_suite.mjc import CubeAssembleEnvironment  # noqa: E402
from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy.gr00t_policy import Gr00tPolicy  # noqa: E402

# ---------------------------------------------------------------------------
# Constants matching the Frank dataset (from data/meta/info.json)
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 144, 256
RAW_RENDER_SCALE = 2
CTRL_HZ = 20
TASK_DESCRIPTION = (
    "Stack the blue block onto the black block, then stack the orange block on top of the blue block."
)


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
    left_joints = obs["left_pos"][:7]   # (7,)
    right_joints = obs["right_pos"][:7]  # (7,)
    left_grip_pos = obs["left_pos"][7]   # scalar
    right_grip_pos = obs["right_pos"][7]  # scalar
    overhead = resize_like_training(obs["overhead_camera"])
    left = resize_like_training(obs["left_camera"])
    right = resize_like_training(obs["right_camera"])

    observation = {
        "video": {
            "overhead": overhead[None, None].astype(np.uint8),  # (1,1,H,W,3)
            "left": left[None, None].astype(np.uint8),
            "right": right[None, None].astype(np.uint8),
        },
        "state": {
            "left_pos_cos": np.cos(left_joints).reshape(1, 1, 7).astype(np.float32),
            "left_pos_sin": np.sin(left_joints).reshape(1, 1, 7).astype(np.float32),
            "right_pos_cos": np.cos(right_joints).reshape(1, 1, 7).astype(np.float32),
            "right_pos_sin": np.sin(right_joints).reshape(1, 1, 7).astype(np.float32),
            "left_gripper": np.array(
                [[left_grip_pos, right_grip_pos]], dtype=np.float32
            ).reshape(1, 1, 2),
            "right_gripper": np.array(
                [[left_gripper_integral, right_gripper_integral]], dtype=np.float32
            ).reshape(1, 1, 2),
            "left_arm": left_joints.reshape(1, 1, 7).astype(np.float32),
            "right_arm": right_joints.reshape(1, 1, 7).astype(np.float32),
        },
        "language": {
            "annotation.human.task_description": [[TASK_DESCRIPTION]],
        },
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
    left_joints = action["left_arm"][0, step_in_chunk]          # (7,)
    right_joints = action["right_arm"][0, step_in_chunk]        # (7,)
    left_grip = np.clip(action["left_gripper"][0, step_in_chunk], 0.0, 1.0)   # (1,)
    right_grip = np.clip(action["right_gripper"][0, step_in_chunk], 0.0, 1.0) # (1,)
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
    max_steps: int = 1500,
    execute_steps: int = 8,
    use_low_pass_filter: bool = False,
    low_pass_alpha: float = 0.2,
    save_dir: pathlib.Path | None = None,
    run_name: str | None = None,
) -> dict:
    """Run one closed-loop episode and return results."""
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
    step = 0
    t_start = time.time()
    inference_times = []
    previous_filtered_action = None

    while step < max_steps:
        # Accumulate gripper integrals
        left_gripper_integral += obs["left_pos"][7]
        right_gripper_integral += obs["right_pos"][7]

        # Convert MuJoCo observation to GR00T format
        gr00t_obs = mjc_obs_to_gr00t(obs, left_gripper_integral, right_gripper_integral)

        # Predict action chunk
        t0 = time.time()
        action_chunk, _ = policy.get_action(gr00t_obs)
        inference_times.append(time.time() - t0)

        # Execute the full action chunk (or until max_steps)
        for t in range(execute_steps):
            if step >= max_steps:
                break

            raw_env_action = gr00t_action_to_mjc(action_chunk, step_in_chunk=t)
            if use_low_pass_filter:
                env_action = low_pass_filter_action(
                    raw_env_action,
                    previous_filtered_action,
                    low_pass_alpha,
                )
                previous_filtered_action = env_action
            else:
                env_action = raw_env_action
            obs = env.step(env_action)

            # Record frame for video
            if save_dir is not None:
                frames.append(obs["user_camera"].copy())

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
        prefix = f"{run_name}_" if run_name else ""
        video_path = save_dir / f"{prefix}seed_{seed}_{tag}.mp4"
        writer = imageio.get_writer(str(video_path), fps=CTRL_HZ, codec="mpeg4", quality=8)
        for frame in frames:
            writer.append_data(frame.astype(np.uint8))
        writer.close()
        print(f"  Video saved to {video_path}")

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
        "--low-pass-alpha", type=float, default=0.2,
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
        "--action-horizon", type=int, default=None,
        help="Deprecated alias for --execute-steps",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save rollout videos (default: outputs/frank_eval/)",
    )
    args = parser.parse_args()

    # Backward-compatible CLI alias.
    if args.action_horizon is not None:
        args.execute_steps = args.action_horizon

    save_dir = pathlib.Path(args.save_dir) if args.save_dir else (
        pathlib.Path(args.checkpoint_path).parent / "eval_videos"
    )

    # Load model
    print(f"Loading model from {args.checkpoint_path} ...")
    policy = Gr00tPolicy(
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        model_path=args.checkpoint_path,
        device=args.device,
    )
    print("Model loaded.\n")

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
