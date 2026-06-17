# Frank Bimanual GR00T Finetuning — How To

End-to-end guide to finetune and evaluate GR00T-N1.5 on the **Frank** bimanual
block-stacking dataset (two Kinova arms + overhead/left/right cameras). Covers
entering the Apptainer container, training, and both open-loop and closed-loop
evaluation.

---

## 0. Prerequisites

- The Apptainer image `gr00t.sif` (kept in `docker/`, not committed — build or
  copy it separately).
- 4 GPUs for training (the script auto-launches `torchrun` for multi-GPU).
- The datasets (not committed — pull from Hugging Face, see step 2):
  - `data/` — original dataset (one generic prompt for every episode).
  - `data_relabeled/` — same data with **arm-specific** language prompts
    (this is what we train on). See [Dataset & relabeling](#3-dataset--relabeling).

---

## 1. Enter the container

From the host, bind the repo root to `/Isaac-GR00T` inside the container:

```bash
apptainer exec --nv --bind PATH_TO_ISAAC_GR00T_ROOT:/Isaac-GR00T gr00t.sif bash
```

Then, **inside the container**:

```bash
cd /Isaac-GR00T
source .venv/bin/activate
```

All commands below assume you are inside the container, in `/Isaac-GR00T`, with
the venv activated.

---

## 2. Get the datasets from Hugging Face

```bash
# pip install -U "huggingface_hub[cli]"   # if not already present
huggingface-cli download <HF_USER>/frank-cubes           --repo-type dataset --local-dir data
huggingface-cli download <HF_USER>/frank-cubes-relabeled --repo-type dataset --local-dir data_relabeled

# data_relabeled ships WITHOUT videos (they are identical to data/videos).
# Recreate the relative symlink so the loader finds them:
ln -sfn ../data/videos data_relabeled/videos
```

> Replace `<HF_USER>/...` with the actual dataset repo names.

`data_relabeled/videos` is a **relative** symlink to `data/videos` to avoid
duplicating ~2.3 GB of video, so keep `data/` and `data_relabeled/` as siblings.

**Simplest alternative:** download only `data/`, then regenerate the relabeled
set locally — this recreates the symlink for you:

```bash
python scripts/relabel_arm_prompts.py --src data --dst data_relabeled
```

---

## 3. Dataset & relabeling

Each arm has very different semantics: one arm is the **active** grasper, the
other only repositions its wrist camera. We make the language prompt name the
arm(s) used so the policy can learn the asymmetry.

`scripts/relabel_arm_prompts.py` reads the recorded gripper-close commands to
recover which arm grasped each cube (the blue cube is always grasped first) and
writes a new dataset with 4 prompt classes:

| task_index | class       | meaning                                   |
|-----------:|-------------|-------------------------------------------|
| 1 | `left`        | left arm grasps both cubes                |
| 2 | `right`       | right arm grasps both cubes               |
| 3 | `left_right`  | left grasps blue, right grasps orange     |
| 4 | `right_left`  | right grasps blue, left grasps orange     |

To regenerate `data_relabeled/` from `data/`:

```bash
python scripts/relabel_arm_prompts.py --src data --dst data_relabeled
# add --copy-videos to duplicate videos instead of symlinking
# add --dry-run to only print the label distribution
```

A per-episode label map is written to `data_relabeled/meta/arm_labels.json`.

---

## 4. Finetuning

Run from `/Isaac-GR00T`. The trailing pipe filters the noisy `libdav1d` AV1
decoder banners out of the log and writes everything else to `output.txt`:

```bash
PYTHONUNBUFFERED=1 python scripts/gr00t_finetune.py \
  --dataset-path data_relabeled/ \
  --num-gpus 4 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --output-dir outputs/frank_ft_N1.5_0609_relabeled \
  --max-steps 400000 \
  --data-config examples.frank.frank_config:FrankDataConfig \
  --save-steps 50000 \
  --lora-rank 64 --lora-alpha 128 --lora-full-model \
  --embodiment-tag new_embodiment \
  --dataloader-num-workers 4 --dataloader-prefetch-factor 2 \
  2>&1 | grep -vF --line-buffered 'libdav1d' > output.txt
```

- Effective batch size = `batch-size × grad-accum × num-gpus` = `4 × 8 × 4 = 128`.
- Watch progress with `tail -f output.txt`.
- Checkpoints land in `--output-dir/checkpoint-<step>/`.

### Resume from a checkpoint

`--resume` continues from the **latest** checkpoint in `--output-dir`. Re-run the
exact same command (same `--output-dir`, `--num-gpus`, `--max-steps`) with
`--resume` added, and append to the log with `>>`:

```bash
PYTHONUNBUFFERED=1 python scripts/gr00t_finetune.py \
  ... (identical args) ... \
  --resume \
  2>&1 | grep -vF --line-buffered 'libdav1d' >> output.txt
```

---

## 5. Open-loop evaluation (MSE vs. ground truth)

Predicts actions on real dataset episodes and compares to the recorded actions.
The prompt is read from each episode automatically, so just pick one episode per
class. Representative episodes: `left=ep0, right=ep3, left_right=ep4, right_left=ep2`.

```bash
CKPT=outputs/frank_ft_N1.5_0609_relabeled/checkpoint-100000
OUT=$CKPT/eval_openloop; mkdir -p $OUT
for spec in "0:left" "3:right" "4:left_right" "2:right_left"; do
  ep=${spec%%:*}; name=${spec##*:}
  python scripts/eval_policy.py \
    --model-path $CKPT \
    --data-config examples.frank.frank_config:FrankDataConfig \
    --embodiment-tag new_embodiment \
    --dataset-path data_relabeled/ \
    --modality-keys left_arm right_arm left_gripper right_gripper \
    --denoising-steps 4 --steps 300 \
    --start-traj $ep --trajs 1 \
    --plot --save-plot-path $OUT/ep${ep}_${name}.png
done
```

Prints MSE per episode and saves a prediction-vs-ground-truth plot for each.

---

## 6. Closed-loop evaluation (MuJoCo rollout)

Runs the policy in the `CubeAssembleEnvironment` simulator (requires
`bimanual_suite`, vendored in `external_dependencies/`). One run per language
command, same seed so only the prompt changes — a direct test of whether the
policy obeys the arm instruction:

```bash
CKPT=outputs/frank_ft_N1.5_0609_relabeled/checkpoint-100000
for arm in left right left_right right_left; do
  python scripts/frank_eval.py \
    --checkpoint-path $CKPT/ \
    --data-config examples.frank.frank_config:FrankDataConfig \
    --embodiment-tag new_embodiment \
    --denoising-steps 8 \
    --use-low-pass-filter \
    --seed 1 --execute-steps 4 \
    --arm $arm
done
```

- `--arm {generic,left,right,left_right,right_left}` selects the prompt
  (matches the `tasks.jsonl` classes); `--task-description "..."` overrides with
  free text.
- Videos save to `$CKPT/eval_videos/<arm>_es4_seed_1_<success|failure>.mp4`.
- `--use-low-pass-filter` applies an EMA smoothing filter to the deployed
  actions (`--low-pass-alpha`, smaller = smoother).

> If `bimanual_suite` is not importable, install it once:
> `pip install -e external_dependencies/bimanual_suite`.

---

## 7. Important files

| File | What it is |
|------|------------|
| `examples/frank/frank_config.py` | **Data config.** Defines state/action keys, the video transforms (resize to 224, mild color jitter), state normalization (sin/cos left raw, others `min_max`) and action normalization (`q99` to clip outliers to ±1). Edit here to change inputs/normalization. |
| `scripts/gr00t_finetune.py` | Training entry point (LoRA + multi-GPU via torchrun). |
| `scripts/relabel_arm_prompts.py` | Builds `data_relabeled/` with arm-specific prompts from gripper-close detection. |
| `scripts/eval_policy.py` | Open-loop MSE evaluation on dataset episodes. |
| `scripts/frank_eval.py` | Closed-loop MuJoCo evaluation. Holds `ARM_PROMPTS` + the `--arm` flag, and the **low-pass action filter** (`low_pass_filter_action`, enabled with `--use-low-pass-filter`). |
| `gr00t/model/policy.py`, `gr00t/utils/eval.py` | Local edits to the GR00T library for this setup. |
| `external_dependencies/bimanual_suite/` | Vendored MuJoCo simulator used by closed-loop eval and data generation. |
| `examples/frank/data_generation.py` | Reference fixed-policy data generator (not run here). |
| `diagnostics/` | Standalone analysis tools (`scan_action_stats.py`, `scan_state_stats.py`, `check_action_roundtrip.py`) — **not** part of the train/eval pipeline; handy for inspecting dataset/action distributions. |

### Two "filters" to know about
1. **Log filter** — the `grep -vF 'libdav1d'` in the training/eval command drops
   the AV1 decoder's per-clip banner spam from `output.txt`.
2. **Action filter** — the optional first-order low-pass (EMA) smoothing on
   deployed actions in closed-loop eval (`--use-low-pass-filter` in
   `scripts/frank_eval.py`).

---

## Notes

- `data/`, `data_relabeled/`, `outputs/`, `wandb/`, `docker/`, `output.txt`, and
  `*.sif` are git-ignored — they are pulled from HF or generated locally, never
  committed.
