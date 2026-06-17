from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class FrankDataConfig(BaseDataConfig):
    video_keys = ["video.overhead", "video.left", "video.right"]
    state_keys = [
        "state.left_pos_cos",
        "state.left_pos_sin",
        "state.right_pos_cos",
        "state.right_pos_sin",
        "state.left_gripper",
        "state.right_gripper",
        "state.left_arm",
        "state.right_arm",
    ]
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_gripper",
        "action.right_gripper",
    ]
    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    action_indices = list(range(0, 20))
    # action_indices = list(range(0, 16))

    def transform(self):
        transforms = [
            # video transforms — kept symmetric train/eval to avoid distribution shift.
            # VideoCrop(scale=1.0) is a no-op (center crop == full frame), and jitter is mild
            # so the model sees something close to the raw MuJoCo render at eval time.
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=1.0),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.02,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            # sin/cos features are bounded in [-1, 1] by construction — normalizing them
            # against empirical dataset min/max distorts the trig identity, so leave them raw.
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    key: "min_max"
                    for key in self.state_keys
                    if not (key.endswith("_pos_cos") or key.endswith("_pos_sin"))
                },
            ),
            # action transforms — q99 clips outliers to ±1, avoiding the jitter that min_max
            # produces when a single extreme sample compresses the action distribution.
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "q99" for key in self.action_keys},
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)
