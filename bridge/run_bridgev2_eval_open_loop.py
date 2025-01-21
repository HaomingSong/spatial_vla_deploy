"""
run_bridgev2_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval.py --model_family openvla --pretrained_checkpoint openvla/openvla-7b
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)
from bridgev2_utils import (
    get_action,
    get_image_resize_size,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
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

    ##################################Image###############################################################################
    # Utils
    #################################################################################################################
    save_data: bool = False                                     # Whether to save rollout data (images, actions, etc.)
    ensemble_actions: bool = False                              # Whether to ensemble
    action_horizon: int = 25                                    # Number of actions

    # fmt: on


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
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "bridge_orig/1.0.0"

    model, processor, action_ensembler, obs_recoder = prepare_vla(cfg)

    # Initialize the WidowX environment
    env = get_widowx_env(cfg, model)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < cfg.max_episodes:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        obs, _ = env.reset()

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
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

        while t < cfg.max_steps:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])

                    # Get preprocessed image
                    obs["full_image"] = get_preprocessed_image(obs, resize_size)
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

                    # [If saving rollout data] Save preprocessed image, robot state, and action
                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_states.append(obs["proprio"])
                        rollout_actions.append(action)

                    # Execute action
                    print("action:", action)
                    obs, _, _, _, _ = env.step(action)
                    t += 1

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, task_label, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Redo episode or continue
        if input("Enter 'r' if you want to redo the episode, or press Enter to continue: ") != "r":
            episode_idx += 1


if __name__ == "__main__":
    eval_model_in_bridge_env()
