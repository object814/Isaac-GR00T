import enum
import mujoco
from mujoco import viewer
import mink
import numpy as np
import imageio
import pathlib
from scipy import spatial
import time
import cv2
import matplotlib.pyplot as plt

from bimanual_suite.mjc import control, CubeAssembleEnvironment
import utils

SOLVER = "osqp"

LIFT_HEIGHT: float = 0.4
PTU_TILT: float = -1.0
PTU_PAN: float = 0.0


CALIBRATION_POSE = {
    "ewellix_lift_top_joint": 0.4000,
    "ptu_pan": 0.0000,
    "ptu_tilt": -1.0000,
    "left_kinova_arm_joint_1": -np.pi + 2.9376,
    "left_kinova_arm_joint_2": 0.9546,
    "left_kinova_arm_joint_3": -2.5865,
    "left_kinova_arm_joint_4": -1.9366,
    "left_kinova_arm_joint_5": -1.5037,
    "left_kinova_arm_joint_6": -2.0908,
    "left_kinova_arm_joint_7": 1.2453,
    "right_kinova_arm_joint_1": -2.9376,
    "right_kinova_arm_joint_2": -0.9546,
    "right_kinova_arm_joint_3": 2.5865,
    "right_kinova_arm_joint_4": 1.9366,
    "right_kinova_arm_joint_5": 1.5037,
    "right_kinova_arm_joint_6": 2.0908,
    "right_kinova_arm_joint_7": -1.2453,
}


class State(enum.Enum):
    START = 0
    BLUE_MOVE_BLACK = 1
    BLUE_MOVE_BLUE = 2
    BLUE_PREGRASP = 3
    BLUE_GRASP = 4
    BLUE_PRECARRY = 5
    BLUE_CARRY = 6
    BLUE_PREPLACE = 7
    BLUE_PLACE = 8
    BLUE_POSTPLACE = 9
    ORANGE_MOVE = 10
    ORANGE_PREGRASP = 11
    ORANGE_GRASP = 12
    ORANGE_PRECARRY = 13
    ORANGE_CARRY = 14
    ORANGE_PREPLACE = 15
    ORANGE_PLACE = 16
    ORANGE_POSTPLACE = 17
    HOME = 18
    DONE = 19


STARTING_STATE = State.BLUE_MOVE_BLACK
PREGRASP = [State.BLUE_PREGRASP, State.ORANGE_PREGRASP]


class StateMachine:

    def __init__(
        self,
        env,
        blue_pos: np.ndarray,
        blue_ori: np.ndarray,
        orange_pos: np.ndarray,
        orange_ori: np.ndarray,
        black_pos: np.ndarray,
        black_ori: np.ndarray,
        left_ee_pose: mink.SE3,
        right_ee_pose: mink.SE3,
    ):
        ee_pose = {control.Arm.LEFT: left_ee_pose, control.Arm.RIGHT: right_ee_pose}
        self.cube_width = env.cube_width
        self.x_max = max(blue_pos[0], orange_pos[0], black_pos[0])
        self.grasp_action = {
            control.Arm.LEFT: np.array([1.0, 0.0]),
            control.Arm.RIGHT: np.array([0.0, 1.0]),
        }
        self.blue_pos = blue_pos
        self.orange_pos = orange_pos
        self.black_pos = black_pos
        # assign a single active arm: whichever EE is closest to the average cube position
        avg_cube_pos = (blue_pos + orange_pos + black_pos) / 3.0
        left_dist = np.linalg.norm(left_ee_pose.translation() - avg_cube_pos)
        right_dist = np.linalg.norm(right_ee_pose.translation() - avg_cube_pos)
        self.active_arm = (
            control.Arm.LEFT if left_dist < right_dist else control.Arm.RIGHT
        )
        self.passive_arm = (
            control.Arm.RIGHT
            if self.active_arm == control.Arm.LEFT
            else control.Arm.LEFT
        )
        self.blue_pick_arm = self.active_arm
        self.orange_pick_arm = self.active_arm
        print(f"Active arm: {self.active_arm}")
        # simplify target orientations based on rotational symmetry
        blue_rot = spatial.transform.Rotation.from_quat(blue_ori, scalar_first=True)
        black_rot = spatial.transform.Rotation.from_quat(black_ori, scalar_first=True)
        orange_rot = spatial.transform.Rotation.from_quat(orange_ori, scalar_first=True)
        # simplify blue
        black_to_blue_rot = blue_rot * black_rot.inv()
        euler = black_to_blue_rot.as_euler("xyz", degrees=True)
        euler[2] = np.mod(euler[2], 90.0)
        blue_ori = (
            spatial.transform.Rotation.from_euler("xyz", euler, degrees=True)
            * black_rot
        ).as_quat(scalar_first=True)
        # simplify orange
        black_to_orange_rot = orange_rot * black_rot.inv()
        euler = black_to_orange_rot.as_euler("xyz", degrees=True)
        euler[2] = np.mod(euler[2], 90.0)
        orange_ori = (
            spatial.transform.Rotation.from_euler("xyz", euler, degrees=True)
            * black_rot
        ).as_quat(scalar_first=True)
        self.blue_pick_pose = mink.SO3(blue_ori)
        self.orange_pick_pose = mink.SO3(orange_ori)
        self.black_pick_pose = mink.SO3(black_ori)

        # rotate picking orientation 90 degrees if it seems like it might collide with another cube
        def correct_pose(grasp_pos, grasp_pose, object_pos):
            diff_pos = object_pos - grasp_pos
            grasp_rot = grasp_pose.as_matrix()
            relative_pos_local = grasp_rot.T @ diff_pos
            print("relative pos local:", relative_pos_local)
            gripper_length = 0.15
            collision = (
                abs(relative_pos_local[0]) < gripper_length
                and abs(relative_pos_local[1]) < self.cube_width
            )
            if collision:
                print("collision!")
                rot_90_z = spatial.transform.Rotation.from_euler("z", 90, degrees=True)
                grasp_rot = spatial.transform.Rotation.from_quat(
                    grasp_pose.wxyz, scalar_first=True
                )
                new_quat = (grasp_rot * rot_90_z).as_quat(scalar_first=True)
                grasp_pose = mink.SO3(new_quat)
            return grasp_pos, grasp_pose

        # self.blue_pick_pos = self.blue_pos
        nearest_pos = (
            self.black_pos
            if np.linalg.norm(self.black_pos - self.blue_pos)
            < np.linalg.norm(self.orange_pos - self.blue_pos)
            else self.orange_pos
        )
        self.blue_pick_pos, self.blue_pick_pose = correct_pose(
            self.blue_pos, self.blue_pick_pose, nearest_pos
        )
        self.orange_pick_pos, self.orange_pick_pose = correct_pose(
            self.orange_pos, self.orange_pick_pose, self.black_pos
        )

        self.ee_pose_init = {
            control.Arm.LEFT: left_ee_pose,
            control.Arm.RIGHT: right_ee_pose,
        }
        self.timing = {s: 0 for s in State}
        self.min_time = 20

        # previous IK error is saved to check IK convergence
        self.previous_active_error = 1e6

        # These poses are calculated at runtime to improve accuracy and IK success
        self.blue_place_pose = None
        self.orange_place_pose = None

        self.object_targets = {State.START: ee_pose[self.active_arm]}

    def step(
        self,
        state: State,
        left_ee_pose: mink.SE3,
        right_ee_pose: mink.SE3,
        blue_pose: mink.SE3,
        orange_pose: mink.SE3,
    ):
        done = False
        ee_pose = {control.Arm.LEFT: left_ee_pose, control.Arm.RIGHT: right_ee_pose}

        # calculate the blue place pose by incorporate the grasped cube position w.r.t the end effector after grasping
        if state == State.BLUE_PREPLACE and self.blue_place_pose is None:
            x, y, z = self.black_pos
            x_blue, y_blue, z_blue = blue_pose.translation()
            x_active, y_active, z_active = ee_pose[self.blue_pick_arm].translation()
            x_gap = x_blue - x
            y_gap = y_blue - y
            z_gap = z_blue - z - self.cube_width * 1.33
            self.blue_place_pose = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose,
                np.array([x_active - x_gap, y_active - y_gap, z_active - z_gap]),
            )
        # calculate the blue place pose by incorporate the grasped cube position w.r.t the end effector after grasping
        if state == State.ORANGE_PREPLACE and self.orange_place_pose is None:
            x, y, z = self.black_pos
            x_orange, y_orange, z_orange = orange_pose.translation()
            x_active, y_active, z_active = ee_pose[self.orange_pick_arm].translation()
            z_gap = z_orange - z - self.cube_width * 2.33
            x_gap = x_orange - x
            y_gap = y_orange - y
            self.orange_place_pose = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose,
                np.array([x_active - x_gap, y_active - y_gap, z_active - z_gap]),
            )
        gripper_actions = np.zeros((2,))
        self.timing[state] += 1

        active_arm = self.active_arm
        passive_arm = self.passive_arm

        # state machine look up table
        if state is State.BLUE_MOVE_BLACK:
            x, y, z = self.blue_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.blue_pick_pose, np.array([x, y, z + 0.2])
            )
            next_state = State.BLUE_MOVE_BLUE
        elif state is State.BLUE_MOVE_BLUE:
            x, y, z = self.blue_pick_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.blue_pick_pose, np.array([x, y, z + 0.2])
            )
            next_state = State.BLUE_PREGRASP
        elif state is State.BLUE_PREGRASP:
            x, y, z = self.blue_pick_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.blue_pick_pose, np.array([x, y, z + control.GRASP_HEIGHT])
            )
            next_state = State.BLUE_GRASP
        elif state is State.BLUE_GRASP:
            x, y, z = self.blue_pick_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.blue_pick_pose, np.array([x, y, z + control.GRASP_HEIGHT])
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.BLUE_PRECARRY
        elif state is State.BLUE_PRECARRY:
            x, y, z = self.blue_pick_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.blue_pick_pose,
                np.array([x, y, z + control.GRASP_HEIGHT + self.cube_width * 3]),
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.BLUE_CARRY
        elif state is State.BLUE_CARRY:
            x, y, z = self.black_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose,
                np.array([x, y, z + control.GRASP_HEIGHT + self.cube_width * 3]),
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.BLUE_PREPLACE
        elif state is State.BLUE_PREPLACE:
            object_target = self.blue_place_pose
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.BLUE_PLACE
        elif state is State.BLUE_PLACE:
            object_target = self.blue_place_pose
            next_state = State.BLUE_POSTPLACE
        elif state is State.BLUE_POSTPLACE:
            x, y, z = self.black_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose, np.array([x, y, z + self.cube_width * 3])
            )
            next_state = State.ORANGE_MOVE
        elif state is State.ORANGE_MOVE:
            x, y, z = self.orange_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.orange_pick_pose, np.array([x, y, z + self.cube_width * 2])
            )
            next_state = State.ORANGE_PREGRASP
        elif state is State.ORANGE_PREGRASP:
            x, y, z = self.orange_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.orange_pick_pose, np.array([x, y, z])
            )
            next_state = State.ORANGE_GRASP
        elif state is State.ORANGE_GRASP:
            x, y, z = self.orange_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.orange_pick_pose, np.array([x, y, z])
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.ORANGE_PRECARRY
        elif state is State.ORANGE_PRECARRY:
            x, y, z = self.orange_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.orange_pick_pose, np.array([x, y, z + self.cube_width * 3.5])
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.ORANGE_CARRY
        elif state is State.ORANGE_CARRY:
            x, y, z = self.black_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose, np.array([x, y, z + self.cube_width * 3.5])
            )
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.ORANGE_PREPLACE
        elif state is State.ORANGE_PREPLACE:
            object_target = self.orange_place_pose
            gripper_actions = self.grasp_action[active_arm]
            next_state = State.ORANGE_PLACE
        elif state is State.ORANGE_PLACE:
            object_target = self.orange_place_pose
            next_state = State.ORANGE_POSTPLACE
        elif state is State.ORANGE_POSTPLACE:
            x, y, z = self.black_pos
            object_target = mink.SE3.from_rotation_and_translation(
                self.black_pick_pose, np.array([x, y, z + self.cube_width * 5])
            )
            next_state = State.HOME
        elif state is State.HOME:
            object_target = self.ee_pose_init[active_arm]
            next_state = State.HOME
        elif state is State.DONE:
            object_target = self.ee_pose_init[active_arm]
            next_state = None
        else:
            raise ValueError("Invalid state")

        self.object_targets[state] = object_target

        previous_target = self.object_targets[State(state.value - 1)]

        # linearly interpolate actual target between current and target
        interpolated_object_target = control.project_pose(
            object_target,
            previous_target,
            ee_pose[active_arm],
            (
                control.DELTA_POS_NORM_PREGRASP
                if state in PREGRASP
                else control.DELTA_POS_NORM
            ),
        )

        active_complete, active_error, active_error_info = control.compute_complete(
            object_target,
            ee_pose[active_arm],
            self.previous_active_error,
            x_threshold=0.01,
            y_threshold=self.cube_width / 2,
            z_threshold=self.cube_width * 0.75,
            quat_threshold=0.1,
            rate_threshold=1e-4,
        )
        self.previous_active_error = active_error
        return {
            "next_state": next_state if active_complete else state,
            active_arm.value: interpolated_object_target,
            passive_arm.value: self.ee_pose_init[passive_arm],
            "active_arm": active_arm,
            "passive_arm": passive_arm,
            "current_left": left_ee_pose,
            "current_right": right_ee_pose,
            "gripper_actions": gripper_actions,
            "done": done,
            "active_error": active_error,
            "active_error_info": active_error_info,
        }


def main(seed: int, verbose: int, use_viewer: bool):
    # make it easy
    env = CubeAssembleEnvironment(seed, render_height=144 * 2, render_width=256 * 2, initial_pose=CALIBRATION_POSE)

    static_lift_cost = np.zeros((env.model.nv,))
    lift_idx = env.model.jnt_dofadr[
        mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_JOINT, "ewellix_lift_top_joint"
        )
    ]
    static_lift_cost[lift_idx] = 100.0
    tasks = [
        left_ee_task := mink.FrameTask(
            frame_name="left_ee",
            frame_type="site",
            position_cost=np.array([2.0, 2.0, 1.0]),
            orientation_cost=1.0,
        ),
        left_elbow_task := mink.FrameTask(
            frame_name="left_elbow",
            frame_type="site",
            position_cost=np.array([0.1, 0.1, 0.1]),
            orientation_cost=0.0,
        ),
        right_ee_task := mink.FrameTask(
            frame_name="right_ee",
            frame_type="site",
            position_cost=np.array([2.0, 2.0, 1.0]),
            orientation_cost=1.0,
        ),
        right_elbow_task := mink.FrameTask(
            frame_name="right_elbow",
            frame_type="site",
            position_cost=np.array([0.1, 0.1, 0.1]),
            orientation_cost=0.0,
        ),
        damping_task := mink.DampingTask(env.model, static_lift_cost),
    ]
    obs = env.reset()

    left_ee_pose, right_ee_pose = control.obs_to_poses(obs)

    _, left_elbow_ori = env.get_site_pose(env.model.site("left_elbow").id)
    _, right_elbow_ori = env.get_site_pose(env.model.site("right_elbow").id)
    left_elbow_task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3(left_elbow_ori), np.array([0.3, 0.4, 1.0])
        )
    )
    right_elbow_task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3(right_elbow_ori), np.array([0.3, -0.4, 1.0])
        )
    )

    initial_cube_poses = env.get_cube_poses()
    action = np.zeros((16,))
    action[:14] = np.concatenate((obs["left_pos"][:7], obs["right_pos"][:7]))

    cube_poses = env.get_cube_poses()

    state_machine = StateMachine(
        env,
        cube_poses["blue_pos"],
        cube_poses["blue_quat"],
        cube_poses["orange_pos"],
        cube_poses["orange_quat"],
        cube_poses["black_pos"],
        cube_poses["black_quat"],
        left_ee_pose,
        right_ee_pose,
    )
    left_arm_geoms = mink.get_subtree_geom_ids(
        env.model, env.model.body("left_kinova_arm_shoulder_link").id
    )
    right_arm_geoms = mink.get_subtree_geom_ids(
        env.model, env.model.body("right_kinova_arm_shoulder_link").id
    )
    left_gripper_geoms = mink.get_subtree_geom_ids(
        env.model, env.model.body("left_kinova_arm_bracelet_link").id
    )
    right_gripper_geoms = mink.get_subtree_geom_ids(
        env.model, env.model.body("right_kinova_arm_bracelet_link").id
    )
    cube_geoms = ["black_cube_geom", "blue_cube_geom", "orange_cube_geom"]
    env_geoms = cube_geoms + ["table"]
    collision_pairs = [
        (left_arm_geoms, left_gripper_geoms),
        (right_arm_geoms, right_gripper_geoms),
        (
            env_geoms,
            left_arm_geoms + right_arm_geoms + left_gripper_geoms + right_gripper_geoms,
        ),
        (["black_cube_geom"], ["blue_cube_geom", "orange_cube_geom"]),
        (["blue_cube_geom"], ["orange_cube_geom"]),
    ]
    limits = [
        mink.ConfigurationLimit(model=env.model),
        mink.CollisionAvoidanceLimit(
            model=env.model,
            geom_pairs=collision_pairs,
            minimum_distance_from_collisions=0.001,
            collision_detection_distance=0.01,
        ),
    ]
    configuration = mink.Configuration(env.model, q=env.data.qpos.copy())
    state = STARTING_STATE
    steps = 0
    homing_steps = 0
    time_start = time.time()
    if use_viewer:
        context = mujoco.viewer.launch_passive(
            model=env.model, data=env.data, show_left_ui=False, show_right_ui=False
        )
    else:
        context = utils.NullContext()

    with context as viewer:
        if use_viewer:
            viewer.opt.frame = (
                mujoco.mjtFrame.mjFRAME_SITE
            )  # view coordinate frames for debugging
        user, overhead, left, right, actions, robot_states, left_crop, right_crop = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        trace = {
            "left_ee_pos": [],
            "left_ee_quat": [],
            "left_ee_rel_pos": [],
            "left_ee_rel_quat": [],
            "right_ee_pos": [],
            "right_ee_quat": [],
            "right_ee_rel_pos": [],
            "right_ee_rel_quat": [],
            "overhead_cam_pos": [],
            "overhead_cam_quat": [],
            "left_cam_pos": [],
            "left_cam_quat": [],
            "right_cam_pos": [],
            "right_cam_quat": [],
            "blue_pos": [],
            "blue_quat": [],
            "black_pos": [],
            "black_quat": [],
            "orange_pos": [],
            "orange_quat": [],
            "left_ee_blue_rel_pos": [],
            "left_ee_blue_rel_quat": [],
            "left_ee_orange_rel_pos": [],
            "left_ee_orange_rel_quat": [],
            "left_ee_black_rel_pos": [],
            "left_ee_black_rel_quat": [],
            "right_ee_blue_rel_pos": [],
            "right_ee_blue_rel_quat": [],
            "right_ee_orange_rel_pos": [],
            "right_ee_orange_rel_quat": [],
            "right_ee_black_rel_pos": [],
            "right_ee_black_rel_quat": [],
        }
        while viewer.is_running():
            left_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(obs["left_ee_quat"]), obs["left_ee_pos"]
            )
            right_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(obs["right_ee_quat"]), obs["right_ee_pos"]
            )
            ee_pose = {
                "left": left_pose,
                "right": right_pose,
            }
            cube_poses = env.get_cube_poses()
            black_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(cube_poses["black_quat"]), cube_poses["black_pos"]
            )
            blue_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(cube_poses["blue_quat"]), cube_poses["blue_pos"]
            )
            orange_pose = mink.SE3.from_rotation_and_translation(
                mink.SO3(cube_poses["orange_quat"]), cube_poses["orange_pos"]
            )
            cube_pose = {
                "black": black_pose,
                "blue": blue_pose,
                "orange": orange_pose,
            }
            if state != state.HOME and state_machine.timing[state] > 200:
                print("state machine is stuck!")
                for k, v in sm_out["active_error_info"].items():
                    print(f"Active {k}: {v}")
                break

            sm_out = state_machine.step(
                state, left_pose, right_pose, blue_pose, orange_pose
            )

            if state == State.HOME:
                break
                # # move back to the home position using a heuristic 3-stage joint space trajectory
                # current = np.concatenate(
                #     (obs["left_pos"][:7], obs["right_pos"][:7], np.zeros((2,)))
                # )
                # target1 = current.copy()
                # target2 = env.default_actions.copy()
                # target3 = env.default_actions.copy()
                # target1[0] = 0.0
                # target1[1] = 0.0
                # target1[5] = 0.0
                # target1[7] = 0.0
                # target1[8] = 0.0
                # target1[12] = 0.0
                # target2[1] = 0.0
                # target2[8] = 0.0
                # mask1 = 1.0 * (np.abs((target1 - target3)) < 1e-2)
                # mask2 = 1.0 * (np.abs((target2 - target3)) < 1e-2)
                # stage1 = np.linalg.norm(mask1 * (current - target1)) > 0.2
                # stage2 = np.linalg.norm(mask2 * (current - target2)) > 0.2
                # homed = np.linalg.norm(current - target3) < 0.1
                # if stage1:
                #     target = target1
                # elif stage2:
                #     target = target2
                # else:
                #     target = target3
                # delta = target - current
                # delta = 0.1 * delta / max(0.1, np.linalg.norm(delta))
                # action = current + delta
                # homing_steps += 1
                # if homed or homing_steps > 500:
                #     break
            else:
                if use_viewer and verbose:
                    print(
                        state,
                        sm_out["next_state"],
                        sm_out["active_error"],
                        sm_out["active_error_info"],
                    )
                left_ee_task.set_target(sm_out["left"])
                right_ee_task.set_target(sm_out["right"])
                ik_succeeded = False
                for damping in [1e-2, 1e-1, 1.0]:
                    try:
                        configuration.update(q=env.data.qpos.copy())
                        vel = mink.solve_ik(
                            configuration,
                            tasks,
                            control.DT,
                            SOLVER,
                            damping=damping,
                            safety_break=False,
                            limits=limits,
                        )
                        ik_succeeded = True
                        break
                    except mink.exceptions.NoSolutionFound:
                        print(f"{damping} failed")
                if not ik_succeeded:
                    print("IK failed across all damping")
                    break
                vel = np.concatenate(
                    (
                        vel[env.left_joint_state_to_vel],
                        vel[env.right_joint_state_to_vel],
                    )
                )
                vel[:6] = (
                    control.MAX_VEL_NORM
                    * vel[:6]
                    / max(control.MAX_VEL_NORM, np.linalg.norm(vel[:6]))
                )
                vel[6] = np.clip(vel[6], -2.0, 2.0)
                vel[7:13] = (
                    control.MAX_VEL_NORM
                    * vel[7:13]
                    / max(control.MAX_VEL_NORM, np.linalg.norm(vel[7:13]))
                )
                vel[13] = np.clip(vel[13], -2.0, 2.0)
                # integrate velocity setpoint to get position setpoint
                action_joint = (
                    np.concatenate((obs["left_pos"][:7], obs["right_pos"][:7]))
                    + control.DT * vel
                )
                action_gripper = sm_out["gripper_actions"]
                action = np.concatenate((action_joint, action_gripper))
            if np.linalg.norm(vel) < 1.0:
                state = sm_out["next_state"]

            robot_state = np.concatenate(
                [obs["left_pos"], obs["right_pos"], obs["left_vel"], obs["right_vel"]],
                axis=-1,
            )[None, :]
            robot_states.append(robot_state)
            actions.append(action[None, :])
            user.append(obs["user_camera"])
            overhead.append(utils.resize(obs["overhead_camera"]))
            left.append(utils.resize(obs["left_camera"]))
            left_crop.append(utils.resize(obs["left_camera_crop"]))
            right.append(utils.resize(obs["right_camera"]))
            right_crop.append(utils.resize(obs["right_camera_crop"]))
            for k in trace:
                if k in obs:
                    trace[k].append(obs[k][None, ...].copy())
            left_ee_pose_, right_ee_pose_ = control.obs_to_poses(obs)
            left_ee_rel = control.relative_pose(left_ee_pose, left_ee_pose_)
            right_ee_rel = control.relative_pose(right_ee_pose, right_ee_pose_)
            trace["left_ee_rel_pos"].append(left_ee_rel.translation()[None, ...])
            trace["left_ee_rel_quat"].append(left_ee_rel.rotation().wxyz[None, ...])
            trace["right_ee_rel_pos"].append(right_ee_rel.translation()[None, ...])
            trace["right_ee_rel_quat"].append(right_ee_rel.rotation().wxyz[None, ...])
            for arm in ["left", "right"]:
                for cube in ["blue", "orange", "black"]:
                    rel = control.relative_pose(ee_pose[arm], cube_pose[cube])
                    trace[f"{arm}_ee_{cube}_rel_pos"].append(
                        rel.translation()[None, ...]
                    )
                    trace[f"{arm}_ee_{cube}_rel_quat"].append(
                        rel.rotation().wxyz[None, ...]
                    )
            left_ee_pose, right_ee_pose = left_ee_pose_, right_ee_pose_

            # gets the next observation
            obs = env.step(action)
            # updates GUI (if running)
            viewer.sync()
            steps += 1

        task_success = state == state.HOME and env.success()
    folder_name = "success" if task_success else "failure"
    emoji = "\U0001f680" if task_success else "\U0001f4a9"
    print(
        f"{folder_name} {emoji}, {steps} steps, {steps / 20} seconds, {time.time() - time_start:.2f} real seconds"
    )
    folder = pathlib.Path(pathlib.Path(__file__).parent / "data" / folder_name)
    folder.mkdir(parents=True, exist_ok=True)

    robot_states = np.concatenate(robot_states, axis=0)
    gripper_integrals = {
        "left_gripper_integral": np.cumsum(robot_states[:, 7]),
        "right_gripper_integral": np.cumsum(robot_states[:, 15]),
    }
    # actions are [left_pos, right_pos, left_gripper, right_gripper]
    actions = np.concatenate(actions, axis=0)
    for k in trace:
        trace[k] = np.concatenate(trace[k], axis=0)

    # cannot use MuJoCo GUI and matplotlib at the same time for some reason
    if not use_viewer:
        fig, axs = plt.subplots(8, 2, figsize=(8, 8))
        for i in range(7):
            axs[i, 0].plot(robot_states[:, i], "b")
            axs[i, 1].plot(robot_states[:, 8 + i], "b")
            axs[i, 0].plot(actions[:, i], "r--")
            axs[i, 1].plot(actions[:, 7 + i], "r--")
        axs[7, 0].plot(robot_states[:, 7], "b")
        axs[7, 0].plot(actions[:, 14], "r--")
        axs[7, 1].plot(robot_states[:, 15], "b")
        axs[7, 1].plot(actions[:, 15], "r--")
        plot_folder = folder / "debug"
        plot_folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_folder / f"rollout_{seed}.png", bbox_inches="tight")

    # write dataset like our teleoperation data
    action_data = {f"action_{i}": actions[:, i] for i in range(actions.shape[1])}
    # robot is [left_pos, left_gripper, right_pos, right_gripper, left_vel, right_vel]
    robot_data = {
        f"robot_state_{i}": robot_states[:, i] for i in range(robot_states.shape[1])
    }
    timestep_data = {
        "timesteps": control.DT * np.arange(steps),
    }
    data = {
        **timestep_data,
        **robot_data,
        **action_data,
        **gripper_integrals,
        **trace,
    }
    data_folder = folder / "data"
    data_folder.mkdir(parents=True, exist_ok=True)
    filename = data_folder / f"stacking_{seed}"
    np.savez_compressed(filename, **data)
    writer = imageio.get_writer(
        folder / f"{seed}_demo.mp4", fps=20, codec="mpeg4", quality=10
    )
    for img in user:
        writer.append_data(img.astype(np.uint8))
    writer.close()
    videos = {
        "overhead": overhead,
        "left": left,
        "right": right,
        "left_crop": left_crop,
        "right_crop": right_crop,
    }
    for name, frames in videos.items():
        writer = imageio.get_writer(
            data_folder / f"stacking_{seed}_{name}.mp4",
            fps=20,
            codec="mpeg4",
            quality=10,
        )
        for img in frames:
            writer.append_data(img.astype(np.uint8))
        writer.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Environment random seed")
    parser.add_argument(
        "--viewer", action="store_true", help="Whether to launch the viewer"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print debug information"
    )
    parser.add_argument(
        "--skip", action="store_true", help="Whether to launch the viewer"
    )
    args = parser.parse_args()
    success_file = (
        pathlib.Path(__file__).parent
        / "data"
        / "success"
        / "data"
        / f"stacking_{args.seed}.npz"
    )
    fail_file = (
        pathlib.Path(__file__).parent
        / "data"
        / "failure"
        / "data"
        / f"stacking_{args.seed}.npz"
    )
    if args.skip and (success_file.exists() or fail_file.exists()):
        print(f"Success results exists, skipping generation.")
    else:
        main(args.seed, args.verbose, args.viewer)