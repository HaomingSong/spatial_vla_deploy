"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""
reset_joint_positions = [
    0.0760389047913384,
    -1.0362613022620384,
    -0.054254247684777324,
    -2.383951857286591,
    -0.004505598470154735,
    1.3820559157131187,
    0.784935455988679,
]

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

from transformers import AutoModel, AutoProcessor
import torch
import numpy as np
from collections import deque
from PIL import Image
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from bridgev2_utils import (
    get_action,
    get_image_resize_size,
)

import cv2
from tensorflow.python.framework.ops import re
import torch
import os.path as osp
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.experimental.motion_utils import reset_joints_to
import deoxys.utils.transform_utils as dft
from realsense_camera import MultiCamera
from PIL import Image
from pathlib import Path
import imageio

import json
import time

ego_camera = "213522070137"
third_camera = "243222074139"
# third_camera = "134322071435" # eye-to-hand-2

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Path = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "10.6.8.66"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.1, -0.20, -0.01, -1.57, 0],
            [0.45, 0.25, 0.30, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/D435/color/image_raw"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 200                                         # Max number of timesteps per episode
    control_frequency: float = 5                                # WidowX control frequency

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    unnorm_key: str = "banana"
    prompt_path: str = "/new_home/haoming/projs/vla_deploy/spatial_vla/franka/prompt.json"
    
    ensemble_actions: bool = False
    action_horizon: int = 4                                    # Number of actions
    sticky_gripper_num_steps: int =1

    # fmt: on

def convert_gripper_action(action):
    action[-1] = 1 - action[-1]
    if action[-1] < 0.5:
        action[-1] = -1

    return action

def get_robot_interface():

    robot_interface = FrankaInterface(osp.join(config_root, "charmander.yml"))
    controller_cfg = YamlConfig(osp.join(config_root, "osc-pose-controller.yml")).as_easydict()
    controller_type = "OSC_POSE"

    return robot_interface, controller_cfg, controller_type

def prepare_vla(cfg):
    model = (
        AutoModel.from_pretrained(
            cfg.pretrained_checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)

    if hasattr(processor, "action_chunk_size"):
        action_ensembler = get_action_ensembler(processor=processor, action_ensemble_temp=-0.8)
    else:
        action_ensembler = None
    if hasattr(processor, "num_obs_steps"):
        obs_recoder = get_obs_recoder(processor=processor)
    else:
        obs_recoder = None

    return model, processor, action_ensembler, obs_recoder


def get_action_ensembler(processor, action_ensemble_temp):
    class ActionEnsembler:
        def __init__(self, processor, action_ensemble_temp=-0.8):
            self.pred_action_horizon = processor.action_chunk_size
            self.action_ensemble_temp = action_ensemble_temp
            self.action_history = deque(maxlen=self.pred_action_horizon)

        def reset(self):
            self.action_history.clear()

        def ensemble_action(self, cur_action):
            self.action_history.append(cur_action)
            num_actions = len(self.action_history)
            if cur_action.ndim == 1:
                curr_act_preds = np.stack(self.action_history)
            else:
                curr_act_preds = np.stack(
                    [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
                )
            # if temp > 0, more recent predictions get exponentially *less* weight than older predictions
            weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
            weights = weights / weights.sum()
            # compute the weighted average across all predictions for this timestep
            cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

            return cur_action

    return ActionEnsembler(processor, action_ensemble_temp)


def get_obs_recoder(processor):
    class ObsRecoder:
        def __init__(self, processor):
            self.obs_horizon = (processor.num_obs_steps - 1) * processor.obs_delta + 1
            self.obs_interval = processor.obs_delta
            self.image_history = deque(maxlen=self.obs_horizon)

        def reset(self):
            self.image_history.clear()

        def add_image_to_history(self, image: Image) -> None:
            if len(self.image_history) == 0:
                self.image_history.extend([image] * self.obs_horizon)
            else:
                self.image_history.append(image)

        def obtain_image_history(self) -> List[Image.Image]:
            image_history = list(self.image_history)
            images = image_history[:: self.obs_interval]
            return images

    return ObsRecoder(processor)


@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    model, processor, action_ensembler, obs_recoder = prepare_vla(cfg)

    # Initialize the WidowX environment
    multi_camera = MultiCamera()
    robot_interface, controller_cfg, controller_type = get_robot_interface()

    with open(cfg.prompt_path, "r") as f:
        prompt = json.load(f)
    task_label = prompt[cfg.unnorm_key]

    # Start evaluation
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        reset_joints_to(robot_interface, reset_joint_positions)

        # Setup
        t = 0
        replay_images = []
        if cfg.save_data:
            rollout_images = []
            rollout_states = []
            rollout_actions = []
        # Start episode
        input(f"Press Enter to start episode {episode_idx+1}...")
        print("Starting episode... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        
        openloop_traj_step = cfg.action_horizon - 1
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        
        while t < cfg.max_steps:
            try:
                # Refresh the camera image and proprioceptive state
                obs = {}
                images = multi_camera.get_frame()
                frame, depth = images[third_camera]
                import copy
                obs["full_image"] = copy.deepcopy(frame)
                replay_images.append(obs["full_image"])

                # Get preprocessed image
                obs["full_image"] = get_preprocessed_image(obs, (224, 224))

                # import cv2
                # cv2.imshow("preprocessed_image", obs["full_image"])
                # cv2.waitKey(1)

                # Query model to get action
                t1 = time.time()
                if cfg.ensemble_actions:
                    action = get_action(
                        cfg,
                        model,
                        obs,
                        task_label,
                        processor=processor,
                        action_ensembler=action_ensembler,
                        obs_recoder=obs_recoder,
                    )
                else:
                    if openloop_traj_step != cfg.action_horizon - 1:
                        openloop_traj_step += 1
                        time.sleep(0.08)
                    else:
                        traj_action = get_action(
                            cfg,
                            model,
                            obs,
                            task_label,
                            processor=processor,
                            action_ensembler=action_ensembler,
                            obs_recoder=obs_recoder,
                        )
                        openloop_traj_step = 0

                        action = traj_action[openloop_traj_step]
                t2 = time.time()
                print(f"Model inference time: {t2 - t1:.2f}s")
                # __import__('ipdb').set_trace()
                # action[0] = 0.8

                # [If saving rollout data] Save preprocessed image, robot state, and action
                if cfg.save_data:
                    rollout_images.append(obs["full_image"])
                    rollout_states.append(obs["proprio"])
                    rollout_actions.append(action)
                
                # sticky gripper
                if (action[-1] < 0.5) != is_gripper_closed:
                    num_consecutive_gripper_change_actions += 1
                else:
                    num_consecutive_gripper_change_actions = 0
# 
                if num_consecutive_gripper_change_actions >= cfg.sticky_gripper_num_steps:
                    is_gripper_closed = not is_gripper_closed
                    num_consecutive_gripper_change_actions = 0
                action[-1] = 0.0 if is_gripper_closed else 1.0

                # Execute action
                rotation_matrix = dft.euler2mat(action[3:6])
                quat = dft.mat2quat(rotation_matrix)
                axis_angle = dft.quat2axisangle(quat)
                action[3:6] = axis_angle
                action = convert_gripper_action(action)
                print("action:", action)
                robot_interface.control(controller_type=controller_type, action=action, controller_cfg=controller_cfg)
                t += 1

            except KeyboardInterrupt as e:
                print("\nCaught KeyboardInterrupt: Terminating episode early.")
                break

        # Save a replay video of the episode
        save_rollout_video(cfg, replay_images, cfg.unnorm_key, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
