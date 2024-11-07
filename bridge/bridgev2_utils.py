"""Utils for evaluating policies in real-world BridgeData V2 environments."""

import os
import sys
import time
import imageio
import numpy as np
import tensorflow as tf
import torch
from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs
from PIL import Image
from typing import List
import cv2

sys.path.append(".")
from experiments.robot.bridge.widowx_env import WidowXGym

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
BRIDGE_PROPRIO_DIM = 7
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})


def get_widowx_env_params(cfg):
    """Gets (mostly default) environment parameters for the WidowX environment."""
    env_params = WidowXConfigs.DefaultEnvParams.copy()
    env_params["override_workspace_boundaries"] = cfg.bounds
    env_params["camera_topics"] = cfg.camera_topics
    env_params["return_full_image"] = True
    return env_params


def get_widowx_env(cfg, model=None):
    """Get WidowX control environment."""
    # Set up the WidowX environment parameters
    env_params = get_widowx_env_params(cfg)
    start_state = np.concatenate([cfg.init_ee_pos, cfg.init_ee_quat])
    env_params["start_state"] = list(start_state)
    # Set up the WidowX client
    widowx_client = WidowXClient(host=cfg.host_ip, port=cfg.port)
    widowx_client.init(env_params)
    env = WidowXGym(
        widowx_client,
        cfg=cfg,
        blocking=cfg.blocking,
    )
    return env


def get_next_task_label(task_label):
    """Prompt the user to input the next task."""
    if task_label == "":
        user_input = ""
        while user_input == "":
            user_input = input("Enter the task name: ")
        task_label = user_input
    else:
        user_input = input("Enter the task name (or leave blank to repeat the previous task): ")
        if user_input == "":
            pass  # Do nothing -> Let task_label be the same
        else:
            task_label = user_input
    print(f"Task: {task_label}")
    return task_label


def save_rollout_video(rollout_images, idx):
    """Saves an MP4 replay of an episode."""
    os.makedirs("./rollouts", exist_ok=True)
    mp4_path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=5)
    for img in rollout_images:
        img = tf.image.flip_left_right(img)
        img = tf.image.flip_up_down(img)
        img = img.numpy()
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")


def save_rollout_data(rollout_orig_images, rollout_images, rollout_states, rollout_actions, idx):
    """
    Saves rollout data from an episode.

    Args:
        rollout_orig_images (list): Original rollout images (before preprocessing).
        rollout_images (list): Preprocessed images.
        rollout_states (list): Proprioceptive states.
        rollout_actions (list): Predicted actions.
        idx (int): Episode index.
    """
    os.makedirs("./rollouts", exist_ok=True)
    path = f"./rollouts/rollout-{DATE_TIME}-{idx+1}.npz"
    # Convert lists to numpy arrays
    orig_images_array = np.array(rollout_orig_images)
    images_array = np.array(rollout_images)
    states_array = np.array(rollout_states)
    actions_array = np.array(rollout_actions)
    # Save to a single .npz file
    np.savez(path, orig_images=orig_images_array, images=images_array, states=states_array, actions=actions_array)
    print(f"Saved rollout data at path {path}")


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    # img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    # img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = tf.image.flip_left_right(img)
    img = tf.image.flip_up_down(img)
    img = img.numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs["full_image"] = resize_image(obs["full_image"], resize_size)
    return obs["full_image"]


def refresh_obs(obs, env):
    """Fetches new observations from the environment and updates the current observations."""
    new_obs = env.get_observation()
    obs["full_image"] = new_obs["full_image"]
    obs["image_primary"] = new_obs["image_primary"]
    obs["proprio"] = new_obs["proprio"]
    return obs


def get_action(cfg, model, obs, task_label, processor=None, action_ensembler=None, obs_recoder=None):
    """Queries the model to get an action."""
    if cfg.model_family == "openvla":
        action = get_vla_action(
            model,
            processor,
            obs,
            task_label,
            cfg.unnorm_key,
            center_crop=cfg.center_crop,
            action_ensembler=action_ensembler,
            obs_recoder=obs_recoder,
        )
        assert action.shape == (ACTION_DIM,)
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return action


def get_image_resize_size(cfg):
    """
    Gets image resize size for a model class.
    If `resize_size` is an int, then the resized image will be a square.
    Else, the image will be a rectangle.
    """
    if cfg.model_family == "openvla":
        resize_size = 224
    else:
        raise ValueError("Unexpected `model_family` found in config.")
    return resize_size


def get_vla_action(
    vla, processor, obs, task_label, unnorm_key, center_crop=False, action_ensembler=None, obs_recoder=None
):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    if obs_recoder is not None:
        obs_recoder.add_image_to_history(image)
        images: List[Image.Image] = obs_recoder.obtain_image_history()
    else:
        images = image

    # Build VLA prompt
    prompt = task_label.lower()

    # predict action (7-dof; un-normalize for bridgev2)
    inputs = processor(prompt, images, unnorm_key)
    if hasattr(processor, "action_tokenizer"):
        generation_outputs = vla.predict_action(**inputs)

        raw_actions = processor.decode_actions(
            generation_outputs=generation_outputs,
            unnorm_key=unnorm_key,
        )["actions"]

    else:
        # __import__('ipdb').set_trace()
        raw_actions = vla.predict_action(**inputs)["actions"]
        raw_actions = raw_actions.cpu().numpy()

    if action_ensembler:
        raw_actions = action_ensembler.ensemble_action(raw_actions)
        # raw_actions = raw_actions[None]

    return raw_actions


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image
