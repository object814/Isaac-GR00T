"""
Replay recorded actions from a training episode in the MuJoCo environment.

Sanity check: loads an episode's actions from the parquet dataset and replays
them open-loop in the CubeAssembleEnvironment, saving a video of the result.

If the replay looks correct (robot assembles the cubes), the data pipeline is
fine and any evaluation issues are in the policy, not the action format.

Usage:
    MUJOCO_GL=egl uv run python scripts/replay_action.py \
        --data-dir /Isaac-GR00T/data \
        --episode 0 \
        --seed 0

    # If you don't know which seed produced episode N, try episode index as seed:
    MUJOCO_GL=egl uv run python scripts/replay_action.py \
        --data-dir /Isaac-GR00T/data \
        --episode 0
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import pathlib

import imageio
import numpy as np
import pandas as pd

from bimanual_suite.mjc import CubeAssembleEnvironment

# ---------------------------------------------------------------------------
# Constants (must match data generation)
# ---------------------------------------------------------------------------
IMG_H, IMG_W = 144, 256
RAW_RENDER_SCALE = 2
CTRL_HZ = 20


def load_episode_actions(data_dir: pathlib.Path, episode_index: int) -> np.ndarray:
    """Load the action array for a given episode from its parquet file."""
    chunk = episode_index // 1000
    parquet_path = data_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    actions = np.stack(df["action"].values).astype(np.float32)
    return actions


def parquet_action_to_env(action: np.ndarray) -> np.ndarray:
    """Parquet action is already in env format — pass through.

    Both formats: [left_joints(7), right_joints(7), left_grip(1), right_grip(1)]
    """
    return action


def main():
    parser = argparse.ArgumentParser(description="Replay training episode actions in MuJoCo")
    parser.add_argument(
        "--data-dir", type=str, default="/data/engs-a2i/catz0908/Isaac-GR00T/data",
        help="Path to the dataset root (contains data/ and meta/)",
    )
    parser.add_argument(
        "--episode", type=int, default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Environment seed (default: same as episode index)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="outputs/replay_videos",
        help="Directory to save the replay video",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    episode_index = args.episode
    seed = args.seed if args.seed is not None else episode_index
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load episode actions
    print(f"Loading episode {episode_index} from {data_dir} ...")
    actions = load_episode_actions(data_dir, episode_index)
    print(f"  {actions.shape[0]} steps, action dim={actions.shape[1]}")

    # Create environment
    print(f"Creating CubeAssembleEnvironment with seed={seed} ...")
    env = CubeAssembleEnvironment(
        seed=seed,
        render_height=IMG_H * RAW_RENDER_SCALE,
        render_width=IMG_W * RAW_RENDER_SCALE,
    )
    obs = env.reset()

    # Replay actions
    frames = []
    frames.append(obs["user_camera"].copy())

    for i in range(len(actions)):
        env_action = parquet_action_to_env(actions[i])
        obs = env.step(env_action)
        frames.append(obs["user_camera"].copy())

        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{len(actions)}")

    success = env.success()
    env.close()

    # Save video
    tag = "success" if success else "failure"
    video_path = save_dir / f"replay_ep{episode_index}_seed{seed}_{tag}.mp4"
    writer = imageio.get_writer(str(video_path), fps=CTRL_HZ, codec="mpeg4", quality=8)
    for frame in frames:
        writer.append_data(frame.astype(np.uint8))
    writer.close()

    print(f"\nDone: {len(actions)} steps replayed")
    print(f"  Success: {success}")
    print(f"  Video:   {video_path}")


if __name__ == "__main__":
    main()
