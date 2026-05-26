import argparse
import cv2
import json
import os
import pickle
import re
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from policy_variant_b import ACTPolicy, CNNMLPPolicy
from utils_variant_b import (
    compute_dict_mean,
    load_data,
    sample_box_pose,
    sample_insertion_pose,
    set_seed,
)
from visualize_episodes import save_videos

"""
Multi-stage training notes (Variant B):

- Purpose: `two_stage` and `three_stage` are multi-phase training workflows
    for the ACT policy that control which parts of the model are trained and
    in what order. They help stabilize training when adding the future-
    prediction branch by pretraining parts of the network separately.

- `two_stage`: stage order `act` -> `roi`.
    * Stage 1 (`act`): train action prediction components.
    * Stage 2 (`roi`): train ROI / communication components while keeping
        action parameters fixed.

- `three_stage`: stage order `act` -> `future` -> `joint`.
    * Stage 1 (`act`): train action prediction.
    * Stage 2 (`future`): train the action-conditioned future visual simulator
        (multi-horizon future RGB and latent prediction present in Variant B).
    * Stage 3 (`joint`): fine-tune all components together.

- Implementation details:
    * `policy.set_train_stage(stage)` in `policy_variant_b.py` toggles
        `requires_grad` for parameters: it recognizes ROI vs future parameters
        and enables/disables them per stage.
    * `--stage1_epochs`, `--stage2_epochs`, and `--stage3_epochs` must be
        supplied for `--three_stage`; `--two_stage` requires `--stage1_epochs`
        and `--stage2_epochs`.
"""


def main(args):
    set_seed(1)
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
    device = args["device"]

    is_sim = task_name[:4] == "sim_"
    if is_sim:
        from constants import SIM_TASK_CONFIGS

        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS

        task_config = TASK_CONFIGS[task_name]
    dataset_dir = args.get("dataset_dir") or task_config["dataset_dir"]
    if args.get("num_episodes") is not None:
        num_episodes = args["num_episodes"]
    elif args.get("dataset_dir"):
        num_episodes = len(
            [
                filename
                for filename in os.listdir(dataset_dir)
                if filename.startswith("episode_") and filename.endswith(".hdf5")
            ]
        )
    else:
        num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    target_camera = args.get("target_camera") or ("top" if "top" in camera_names else camera_names[0])
    if target_camera not in camera_names:
        raise ValueError(f"target_camera {target_camera} must be one of {camera_names}")

    state_dim = 14
    lr_backbone = 1e-5
    backbone = "resnet18"
    if policy_class == "ACT":
        policy_config = {
            "lr": args["lr"],
            "num_queries": args["chunk_size"],
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": camera_names,
            "target_camera": target_camera,
            "lambda_roi": args["lambda_roi"],
            "lambda_sem": args["lambda_sem"],
            "lambda_sig": args["lambda_sig"],
            "lambda_recon_grad": args["lambda_recon_grad"],
            "focus_masked_region": args["focus_masked_region"],
            "comm_num_queries": args["comm_num_queries"],
            "comm_layers": args["comm_layers"],
            "comm_detach_warmup": args["comm_detach_warmup"],
            "ema_momentum": args["ema_momentum"],
            "future_horizons": args["future_horizons"],
            "future_image_height": args["future_image_height"],
            "future_image_width": args["future_image_width"],
            "future_layers": args["future_layers"],
            "lambda_future_rgb": args["lambda_future_rgb"],
            "lambda_future_grad": args["lambda_future_grad"],
            "lambda_future_latent": args["lambda_future_latent"],
            "future_rgb_decay_alpha": args["future_rgb_decay_alpha"],
            "future_latent_decay_alpha": args["future_latent_decay_alpha"],
            "future_teacher_mix_steps": args["future_teacher_mix_steps"],
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "device": device,
        "resume_ckpt": args.get("resume_ckpt"),
        "grad_accum_steps": args.get("grad_accum_steps", 1),
        "two_stage": args.get("two_stage", False),
        "three_stage": args.get("three_stage", False),
        "stage1_epochs": args.get("stage1_epochs"),
        "stage2_epochs": args.get("stage2_epochs"),
        "stage3_epochs": args.get("stage3_epochs"),
    }

    if args.get("export_netron_onnx"):
        export_act_netron_onnx(config, args["export_netron_onnx"], args.get("export_ckpt"))
        return

    if is_eval:
        results = []
        for ckpt_name in ["policy_best.ckpt"]:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        return

    if config["three_stage"]:
        if policy_class != "ACT":
            raise NotImplementedError("--three_stage is implemented for ACT policy only.")
        if args.get("stage1_epochs") is None or args.get("stage2_epochs") is None or args.get("stage3_epochs") is None:
            raise ValueError("--three_stage requires --stage1_epochs, --stage2_epochs, and --stage3_epochs.")
    elif config["two_stage"]:
        if policy_class != "ACT":
            raise NotImplementedError("--two_stage is implemented for ACT policy only.")
        if args.get("stage1_epochs") is None or args.get("stage2_epochs") is None:
            raise ValueError("--two_stage requires both --stage1_epochs and --stage2_epochs.")
    elif num_epochs is None:
        raise ValueError("--num_epochs is required unless --two_stage is enabled.")

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_training_config_snapshot(
        ckpt_dir,
        args,
        config,
        {
            "dataset_dir": dataset_dir,
            "num_episodes": num_episodes,
            "episode_len": episode_len,
            "camera_names": camera_names,
            "target_camera": target_camera,
            "is_sim": is_sim,
            "task_config": task_config,
        },
    )

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        target_camera,
        roi_background_weight=args["roi_background_weight"],
        roi_detail_weight=args["roi_detail_weight"],
        focus_masked_region=args["focus_masked_region"],
        future_horizons=args["future_horizons"],
        future_image_height=args["future_image_height"],
        future_image_width=args["future_image_width"],
    )

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as file_obj:
        pickle.dump(stats, file_obj)

    if config["three_stage"]:
        best_ckpt_info = train_bc_three_stage(train_dataloader, val_dataloader, config)
    elif config["two_stage"]:
        best_ckpt_info = train_bc_two_stage(train_dataloader, val_dataloader, config)
    else:
        best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        torch.save(best_state_dict, os.path.join(ckpt_dir, "policy_best.ckpt"))
        print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError


def make_optimizer(policy_class, policy):
    if policy_class in {"ACT", "CNNMLP"}:
        return policy.configure_optimizers()
    raise NotImplementedError


def checkpoint_model_state(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def save_training_config_snapshot(ckpt_dir, args, config, resolved_context):
    snapshot = {
        "command": " ".join(sys.argv),
        "args": json_safe(args),
        "resolved_context": json_safe(resolved_context),
        "training_config": json_safe(config),
    }
    config_path = os.path.join(ckpt_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as file_obj:
        json.dump(snapshot, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    command_path = os.path.join(ckpt_dir, "training_command.txt")
    with open(command_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(snapshot["command"])
        file_obj.write("\n")

    print(f"Saved training config snapshot to {config_path}")


def load_policy_checkpoint(policy, ckpt_path, device, description):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{description} checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    loading_status = policy.load_state_dict(checkpoint_model_state(checkpoint))
    print(f"Loaded {description} checkpoint from {ckpt_path}: {loading_status}")
    return checkpoint


def checkpoint_epoch(checkpoint, ckpt_path):
    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        return int(checkpoint["epoch"])

    match = re.search(r"epoch_(\d+)", os.path.basename(ckpt_path))
    if match:
        return int(match.group(1))

    return None


def checkpoint_stage_index(ckpt_path, stage_prefixes):
    basename = os.path.basename(ckpt_path).lower()
    for idx, prefix in enumerate(stage_prefixes):
        if prefix.lower() in basename:
            return idx
    return None


def staged_resume_plan(resume_ckpt, checkpoint, stage_prefixes, stage_epoch_counts):
    stage_idx = checkpoint_stage_index(resume_ckpt, stage_prefixes)
    if stage_idx is None:
        raise ValueError(
            f"Could not infer resume stage from checkpoint name {resume_ckpt!r}. "
            f"Expected one of these stage markers in the filename: {stage_prefixes}."
        )

    saved_epoch = checkpoint_epoch(checkpoint, resume_ckpt)
    if saved_epoch is None:
        basename = os.path.basename(resume_ckpt)
        if basename.endswith("_last.ckpt"):
            saved_epoch = stage_epoch_counts[stage_idx] - 1
            print(f"Inferred completed stage from last checkpoint {basename}: epoch {saved_epoch}")
        else:
            raise ValueError(
                f"Could not infer saved epoch from checkpoint {resume_ckpt!r}. "
                "Use a *_training.ckpt checkpoint or an epoch-named weights checkpoint."
            )

    next_epoch = saved_epoch + 1
    if next_epoch < stage_epoch_counts[stage_idx]:
        return stage_idx, next_epoch
    if next_epoch == stage_epoch_counts[stage_idx]:
        return stage_idx + 1, 0
    raise ValueError(
        f"Checkpoint epoch {saved_epoch} exceeds configured length for {stage_prefixes[stage_idx]} "
        f"({stage_epoch_counts[stage_idx]} epochs)."
    )


class ACTNetronExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

    def forward(self, qpos, image):
        image = (image - self.image_mean) / self.image_std
        a_hat, is_pad_hat, _, _, comm_outputs = self.model(qpos, image, None)
        return a_hat, is_pad_hat, comm_outputs["roi_hat"], comm_outputs["z_comm_pool"]


def export_act_netron_onnx(config, output_path, ckpt_path=None):
    if config["policy_class"] != "ACT":
        raise NotImplementedError("Netron ONNX export is implemented for ACT policy only.")

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    policy = make_policy(config["policy_class"], config["policy_config"])
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        loading_status = policy.load_state_dict(checkpoint_model_state(checkpoint))
        print(f"Loaded export checkpoint from {ckpt_path}: {loading_status}")

    policy.to(device)
    policy.eval()
    wrapper = ACTNetronExportWrapper(policy.model).to(device).eval()

    batch_size = 1
    num_cameras = len(config["camera_names"])
    state_dim = config["state_dim"]
    dummy_qpos = torch.zeros(batch_size, state_dim, dtype=torch.float32, device=device)
    dummy_image = torch.zeros(batch_size, num_cameras, 3, 480, 640, dtype=torch.float32, device=device)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_qpos, dummy_image),
            output_path,
            input_names=["qpos", "image"],
            output_names=["actions", "is_pad", "roi_hat", "z_comm_pool"],
            dynamic_axes={
                "qpos": {0: "batch"},
                "image": {0: "batch"},
                "actions": {0: "batch"},
                "is_pad": {0: "batch"},
                "roi_hat": {0: "batch"},
                "z_comm_pool": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
    print(f"Saved Netron ONNX graph to {output_path}")


def scalarize_dict(metrics):
    result = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().item()
        else:
            result[key] = value
    return result


def save_training_checkpoint(
    ckpt_path,
    epoch,
    policy,
    optimizer,
    train_history,
    validation_history,
    min_val_loss,
    best_ckpt_info,
):
    best_epoch = None
    best_state_dict = None
    if best_ckpt_info is not None:
        best_epoch, _, best_state_dict = best_ckpt_info

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_history": train_history,
            "validation_history": validation_history,
            "min_val_loss": min_val_loss,
            "best_epoch": best_epoch,
            "best_state_dict": best_state_dict,
        },
        ckpt_path,
    )


def load_training_checkpoint(resume_ckpt, policy, optimizer, device):
    checkpoint = torch.load(resume_ckpt, map_location=device)
    loading_status = policy.load_state_dict(checkpoint_model_state(checkpoint))
    print(loading_status)

    start_epoch = 0
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        start_epoch = checkpoint.get("epoch", -1) + 1
        train_history = checkpoint.get("train_history", [])
        validation_history = checkpoint.get("validation_history", [])
        min_val_loss = checkpoint.get("min_val_loss", np.inf)

        best_epoch = checkpoint.get("best_epoch")
        best_state_dict = checkpoint.get("best_state_dict")
        if best_epoch is not None and best_state_dict is not None:
            best_ckpt_info = (best_epoch, min_val_loss, best_state_dict)

        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            try:
                optimizer.load_state_dict(optimizer_state_dict)
                print("Loaded optimizer state successfully.")
            except ValueError as exc:
                print(f"Warning: could not load optimizer state: {exc}")
                print("Continuing with a fresh optimizer state.")
    else:
        match = re.search(r"epoch_(\d+)", os.path.basename(resume_ckpt))
        if match:
            start_epoch = int(match.group(1)) + 1

    return start_epoch, train_history, validation_history, min_val_loss, best_ckpt_info


def get_image(ts, camera_names, device="cuda"):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation["images"][cam_name], "h w c -> c h w")
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)


def tensor_to_uint8_image(image_tensor):
    image = image_tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def tensor_to_uint8_frames(frame_tensor):
    frames = frame_tensor.detach().cpu().squeeze(0).permute(0, 2, 3, 1).numpy()
    frames = np.clip(frames, 0.0, 1.0)
    return (frames * 255.0).astype(np.uint8)


def add_panel_label(image, text):
    labeled = image.copy()
    cv2.putText(
        labeled,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_future_strip(frames, labels, panel_size=(160, 120)):
    panels = []
    for frame, label in zip(frames, labels):
        panel = cv2.resize(frame, panel_size, interpolation=cv2.INTER_AREA)
        panels.append(add_panel_label(panel, label))
    return np.concatenate(panels, axis=1)


def build_eval_frame(robot_view, pred_future, gt_future, horizons):
    robot_panel = cv2.resize(robot_view, (320, 240), interpolation=cv2.INTER_AREA)
    robot_panel = add_panel_label(robot_panel, "Robot rollout")
    labels = [f"h{h}" for h in horizons]
    pred_strip = make_future_strip(pred_future, [f"Pred {label}" for label in labels])
    gt_strip = make_future_strip(gt_future, [f"GT {label}" for label in labels])
    right = np.concatenate([pred_strip, gt_strip], axis=0)
    if right.shape[0] != robot_panel.shape[0]:
        robot_panel = cv2.resize(robot_panel, (robot_panel.shape[1], right.shape[0]), interpolation=cv2.INTER_AREA)
    return np.concatenate([robot_panel, right], axis=1)


def save_eval_video(frames, dt, video_path):
    if not frames:
        return
    h, w, _ = frames[0].shape
    fps = int(1 / dt)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame[:, :, [2, 1, 0]])
    out.release()
    print(f"Saved eval video to: {video_path}")


def sample_non_colliding_pose(existing_poses, min_dist=0.06, max_tries=100):
    for _ in range(max_tries):
        pose = sample_box_pose()
        collision = False
        for existing_pose in existing_poses:
            if np.linalg.norm(pose[:2] - existing_pose[:2]) < min_dist:
                collision = True
                break
        if not collision:
            return pose
    return sample_box_pose()


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config["ckpt_dir"]
    state_dim = config["state_dim"]
    real_robot = config["real_robot"]
    policy_class = config["policy_class"]
    onscreen_render = config["onscreen_render"]
    policy_config = config["policy_config"]
    camera_names = config["camera_names"]
    max_timesteps = config["episode_len"]
    task_name = config["task_name"]
    temporal_agg = config["temporal_agg"]
    device = config["device"]
    onscreen_cam = "angle"
    target_camera = policy_config.get("target_camera", camera_names[0])
    future_horizons = policy_config.get("future_horizons", [0, 5, 15, 30, 60, 99])

    if policy_class != "ACT":
        raise NotImplementedError("Variant B future visualization is implemented for ACT policy only.")
    if real_robot:
        raise NotImplementedError("Variant B future visualization currently supports simulation only.")

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device_obj)
    loading_status = policy.load_state_dict(checkpoint_model_state(checkpoint))
    print(loading_status)
    policy.to(device_obj)
    policy.eval()

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as file_obj:
        stats = pickle.load(file_obj)

    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda action: action * stats["action_std"] + stats["action_mean"]

    from sim_env import BOX_POSE, make_sim_env

    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward
    if "sim_transfer_cube" not in task_name:
        raise NotImplementedError("Variant B future visualization currently supports sim_transfer_cube only.")

    query_frequency = policy_config["num_queries"]
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config["num_queries"]

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 1
        if "sim_transfer_cube" in task_name:
            p_red = sample_box_pose()
            p_dist1 = sample_non_colliding_pose([p_red])
            p_dist2 = sample_non_colliding_pose([p_red, p_dist1])
            p_dist3 = sample_non_colliding_pose([p_red, p_dist1, p_dist2])
            BOX_POSE[0] = np.concatenate([p_red, p_dist1, p_dist2, p_dist3])
        elif "sim_insertion" in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).to(device_obj)

        robot_views = []
        target_views = []
        pred_future_by_t = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
        pred_future = None
        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                qpos_numpy = np.array(obs["qpos"])
                qpos = torch.from_numpy(pre_process(qpos_numpy)).float().to(device_obj).unsqueeze(0)
                curr_image = get_image(ts, camera_names, device_obj)
                obs_target_rgb = obs["images"][target_camera]

                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions, _, comm_outputs = policy(qpos, curr_image, return_comm=True)
                        future_rgb_hat = comm_outputs.get("future_rgb_hat")
                        if future_rgb_hat is not None:
                            pred_future = tensor_to_uint8_frames(future_rgb_hat)
                        else:
                            pred_future = None
                    if temporal_agg:
                        all_time_actions[[t], t : t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).to(device_obj).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config["policy_class"] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                raw_action = raw_action.squeeze(0).cpu().numpy()
                target_qpos = post_process(raw_action)
                ts = env.step(target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                robot_views.append(obs["images"][onscreen_cam])
                target_views.append(obs_target_rgb)
                if pred_future is None:
                    pred_future_by_t.append(
                        np.zeros(
                            (
                                len(future_horizons),
                                policy_config.get("future_image_height", 240),
                                policy_config.get("future_image_width", 320),
                                3,
                            ),
                            dtype=np.uint8,
                        )
                    )
                else:
                    pred_future_by_t.append(pred_future)

            plt.close()
        eval_frames = []
        if save_episode:
            for t, pred_future_t in enumerate(pred_future_by_t):
                gt_future = []
                for horizon in future_horizons:
                    gt_idx = min(t + int(horizon), len(target_views) - 1)
                    gt_future.append(target_views[gt_idx])
                eval_frames.append(build_eval_frame(robot_views[t], pred_future_t, gt_future, future_horizons))
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, "
            f"{env_max_reward=}, Success: {episode_highest_reward == env_max_reward}"
        )

        if save_episode:
            save_eval_video(eval_frames, DT, video_path=os.path.join(ckpt_dir, f"video{rollout_id}.mp4"))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for reward in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= reward).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {reward}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)
    result_file_name = "result_" + ckpt_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_dir, result_file_name), "w", encoding="utf-8") as file_obj:
        file_obj.write(summary_str)
        file_obj.write(repr(episode_returns))
        file_obj.write("\n\n")
        file_obj.write(repr(highest_rewards))
    return success_rate, avg_return


def forward_pass(data, policy, device="cuda"):
    (
        image_data,
        qpos_data,
        action_data,
        is_pad,
        _,
        future_rgb,
        future_valid,
        future_roi_weight_mask,
        future_focus_weight_mask,
    ) = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    future_rgb = future_rgb.to(device)
    future_valid = future_valid.to(device)
    future_roi_weight_mask = future_roi_weight_mask.to(device)
    future_focus_weight_mask = future_focus_weight_mask.to(device)
    return policy(
        qpos_data,
        image_data,
        action_data,
        is_pad,
        future_rgb,
        future_valid,
        future_roi_weight_mask,
        future_focus_weight_mask,
    )


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]
    resume_ckpt = config.get("resume_ckpt")
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)
    if grad_accum_steps > 1:
        print(f"Using gradient accumulation: {grad_accum_steps} steps")

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    start_epoch = 0

    if resume_ckpt is not None:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")
        print(f"\nResuming training from checkpoint: {resume_ckpt}")
        start_epoch, train_history, validation_history, min_val_loss, best_ckpt_info = load_training_checkpoint(
            resume_ckpt, policy, optimizer, device
        )
        print(f"Resumed at epoch {start_epoch}.")

    if start_epoch >= num_epochs:
        print(f"Checkpoint is already at or beyond num_epochs={num_epochs}. Nothing to train.")
        if best_ckpt_info is None:
            best_ckpt_info = (start_epoch - 1, min_val_loss, deepcopy(policy.state_dict()))
        return best_ckpt_info

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\nEpoch {epoch}")
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                epoch_dicts.append(scalarize_dict(forward_pass(data, policy, device)))
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        print("".join([f"{key}: {value:.3f} " for key, value in epoch_summary.items()]))

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start_idx = len(train_history)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            (loss / grad_accum_steps).backward()
            should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader)
            if should_step:
                optimizer.step()
                if hasattr(policy, "update_ema"):
                    policy.update_ema()
                optimizer.zero_grad(set_to_none=True)
            train_history.append(scalarize_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        print("".join([f"{key}: {value:.3f} " for key, value in epoch_summary.items()]))

        if epoch % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"))
            save_training_checkpoint(
                os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}_training.ckpt"),
                epoch,
                policy,
                optimizer,
                train_history,
                validation_history,
                min_val_loss,
                best_ckpt_info,
            )
            plot_history(train_history, validation_history, epoch + 1, ckpt_dir, seed)

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, "policy_last.ckpt"))
    save_training_checkpoint(
        os.path.join(ckpt_dir, "policy_last_training.ckpt"),
        num_epochs - 1,
        policy,
        optimizer,
        train_history,
        validation_history,
        min_val_loss,
        best_ckpt_info,
    )
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt"))
    print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def train_stage(
    policy,
    train_dataloader,
    val_dataloader,
    config,
    stage,
    num_epochs,
    ckpt_prefix,
    start_epoch=0,
    resume_checkpoint=None,
):
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    device = config["device"]
    grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))

    policy.set_train_stage(stage)
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    if resume_checkpoint is not None:
        if isinstance(resume_checkpoint, dict) and "model_state_dict" in resume_checkpoint:
            train_history = resume_checkpoint.get("train_history", [])
            validation_history = resume_checkpoint.get("validation_history", [])
            min_val_loss = resume_checkpoint.get("min_val_loss", np.inf)
            best_epoch = resume_checkpoint.get("best_epoch")
            best_state_dict = resume_checkpoint.get("best_state_dict")
            if best_epoch is not None and best_state_dict is not None:
                best_ckpt_info = (best_epoch, min_val_loss, best_state_dict)

            optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
            if optimizer_state_dict is not None:
                try:
                    optimizer.load_state_dict(optimizer_state_dict)
                    print(f"Loaded optimizer state for resumed {ckpt_prefix}.")
                except ValueError as exc:
                    print(f"Warning: could not load optimizer state for resumed {ckpt_prefix}: {exc}")
                    print("Continuing with a fresh optimizer state.")
        else:
            print(f"Resuming {ckpt_prefix} from model weights only; optimizer/history start fresh.")

    if start_epoch >= num_epochs:
        raise ValueError(f"{ckpt_prefix} resume start_epoch={start_epoch} is not less than num_epochs={num_epochs}.")

    if grad_accum_steps > 1:
        print(f"Using gradient accumulation for {ckpt_prefix}: {grad_accum_steps} steps")
    if start_epoch > 0:
        print(f"Resuming {ckpt_prefix} at epoch {start_epoch} / {num_epochs}")

    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f"\n{ckpt_prefix} Epoch {epoch}")
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for data in val_dataloader:
                epoch_dicts.append(scalarize_dict(forward_pass(data, policy, device)))
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f"Val loss:   {epoch_val_loss:.5f}")
        print("".join([f"{key}: {value:.3f} " for key, value in epoch_summary.items()]))

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start_idx = len(train_history)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            (loss / grad_accum_steps).backward()
            should_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_dataloader)
            if should_step:
                optimizer.step()
                if hasattr(policy, "update_ema"):
                    policy.update_ema()
                optimizer.zero_grad(set_to_none=True)
            train_history.append(scalarize_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[epoch_start_idx:])
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        print("".join([f"{key}: {value:.3f} " for key, value in epoch_summary.items()]))

        if epoch % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, f"policy_{ckpt_prefix}_epoch_{epoch}_seed_{seed}.ckpt"))
            save_training_checkpoint(
                os.path.join(ckpt_dir, f"policy_{ckpt_prefix}_epoch_{epoch}_seed_{seed}_training.ckpt"),
                epoch,
                policy,
                optimizer,
                train_history,
                validation_history,
                min_val_loss,
                best_ckpt_info,
            )
            plot_history(train_history, validation_history, epoch + 1, ckpt_dir, seed, prefix=f"{ckpt_prefix}_")

    torch.save(policy.state_dict(), os.path.join(ckpt_dir, f"policy_{ckpt_prefix}_last.ckpt"))
    save_training_checkpoint(
        os.path.join(ckpt_dir, f"policy_{ckpt_prefix}_last_training.ckpt"),
        num_epochs - 1,
        policy,
        optimizer,
        train_history,
        validation_history,
        min_val_loss,
        best_ckpt_info,
    )
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, f"policy_{ckpt_prefix}_best.ckpt"))
    print(f"{ckpt_prefix} finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, prefix=f"{ckpt_prefix}_")
    return best_ckpt_info


def train_bc_two_stage(train_dataloader, val_dataloader, config):
    seed = config["seed"]
    device = config["device"]
    ckpt_dir = config["ckpt_dir"]
    stage1_epochs = int(config["stage1_epochs"])
    stage2_epochs = int(config["stage2_epochs"])

    if stage1_epochs <= 0 or stage2_epochs <= 0:
        raise ValueError("--stage1_epochs and --stage2_epochs must both be positive.")

    set_seed(seed)
    policy = make_policy(config["policy_class"], config["policy_config"])
    policy.to(device)
    resume_ckpt = config.get("resume_ckpt")
    resume_checkpoint = None
    resume_stage_idx = 0
    resume_start_epoch = 0
    stage_prefixes = ["stage1", "stage2"]
    stage_epoch_counts = [stage1_epochs, stage2_epochs]

    if resume_ckpt is not None:
        resume_checkpoint = load_policy_checkpoint(policy, resume_ckpt, device, "two-stage resume")
        resume_stage_idx, resume_start_epoch = staged_resume_plan(
            resume_ckpt,
            resume_checkpoint,
            stage_prefixes,
            stage_epoch_counts,
        )
        if resume_stage_idx >= len(stage_prefixes):
            raise ValueError("The supplied two-stage checkpoint is already at or beyond the end of training.")

    print(f"\nStarting two-stage training: stage1_epochs={stage1_epochs}, stage2_epochs={stage2_epochs}")
    if resume_stage_idx <= 0:
        stage1_best_info = train_stage(
            policy,
            train_dataloader,
            val_dataloader,
            config,
            stage="act",
            num_epochs=stage1_epochs,
            ckpt_prefix="stage1",
            start_epoch=resume_start_epoch if resume_stage_idx == 0 else 0,
            resume_checkpoint=resume_checkpoint if resume_stage_idx == 0 else None,
        )

        _, _, stage1_best_state_dict = stage1_best_info
        loading_status = policy.load_state_dict(stage1_best_state_dict)
        print(f"Loaded stage 1 best checkpoint for stage 2: {loading_status}")
    else:
        print("Resume checkpoint is already in stage 2; skipping stage 1.")

    stage2_best_info = train_stage(
        policy,
        train_dataloader,
        val_dataloader,
        config,
        stage="roi",
        num_epochs=stage2_epochs,
        ckpt_prefix="stage2",
        start_epoch=resume_start_epoch if resume_stage_idx == 1 else 0,
        resume_checkpoint=resume_checkpoint if resume_stage_idx == 1 else None,
    )

    best_epoch, min_val_loss, best_state_dict = stage2_best_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, "policy_best.ckpt"))
    print(f"Best two-stage ckpt, stage 2 val loss {min_val_loss:.6f} @ epoch{best_epoch}")
    return stage2_best_info


def train_bc_three_stage(train_dataloader, val_dataloader, config):
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    device = config["device"]
    stage1_epochs = int(config["stage1_epochs"])
    stage2_epochs = int(config["stage2_epochs"])
    stage3_epochs = int(config["stage3_epochs"])

    if stage1_epochs <= 0 or stage2_epochs <= 0 or stage3_epochs <= 0:
        raise ValueError("--stage1_epochs, --stage2_epochs, and --stage3_epochs must all be positive.")

    set_seed(seed)
    policy = make_policy(config["policy_class"], config["policy_config"])
    policy.to(device)
    resume_ckpt = config.get("resume_ckpt")
    resume_checkpoint = None
    resume_stage_idx = 0
    resume_start_epoch = 0
    stage_prefixes = ["stage1_act", "stage2_future", "stage3_joint"]
    stage_epoch_counts = [stage1_epochs, stage2_epochs, stage3_epochs]

    if resume_ckpt is not None:
        resume_checkpoint = load_policy_checkpoint(policy, resume_ckpt, device, "three-stage resume")
        resume_stage_idx, resume_start_epoch = staged_resume_plan(
            resume_ckpt,
            resume_checkpoint,
            stage_prefixes,
            stage_epoch_counts,
        )
        if resume_stage_idx >= len(stage_prefixes):
            raise ValueError("The supplied three-stage checkpoint is already at or beyond the end of training.")

    print(
        "\nStarting three-stage Variant B training: "
        f"stage1_act={stage1_epochs}, stage2_future={stage2_epochs}, stage3_joint={stage3_epochs}"
    )
    if resume_stage_idx <= 0:
        stage1_best_info = train_stage(
            policy,
            train_dataloader,
            val_dataloader,
            config,
            stage="act",
            num_epochs=stage1_epochs,
            ckpt_prefix="stage1_act",
            start_epoch=resume_start_epoch if resume_stage_idx == 0 else 0,
            resume_checkpoint=resume_checkpoint if resume_stage_idx == 0 else None,
        )

        _, _, stage1_best_state_dict = stage1_best_info
        loading_status = policy.load_state_dict(stage1_best_state_dict)
        print(f"Loaded stage 1 best checkpoint for future simulator stage: {loading_status}")
    else:
        print("Resume checkpoint is already past stage 1; skipping stage 1.")

    if resume_stage_idx <= 1:
        stage2_best_info = train_stage(
            policy,
            train_dataloader,
            val_dataloader,
            config,
            stage="future",
            num_epochs=stage2_epochs,
            ckpt_prefix="stage2_future",
            start_epoch=resume_start_epoch if resume_stage_idx == 1 else 0,
            resume_checkpoint=resume_checkpoint if resume_stage_idx == 1 else None,
        )

        _, _, stage2_best_state_dict = stage2_best_info
        loading_status = policy.load_state_dict(stage2_best_state_dict)
        print(f"Loaded stage 2 best checkpoint for joint fine-tuning: {loading_status}")
    else:
        print("Resume checkpoint is already in stage 3; skipping stage 2.")

    stage3_best_info = train_stage(
        policy,
        train_dataloader,
        val_dataloader,
        config,
        stage="joint",
        num_epochs=stage3_epochs,
        ckpt_prefix="stage3_joint",
        start_epoch=resume_start_epoch if resume_stage_idx == 2 else 0,
        resume_checkpoint=resume_checkpoint if resume_stage_idx == 2 else None,
    )

    best_epoch, min_val_loss, best_state_dict = stage3_best_info
    torch.save(best_state_dict, os.path.join(ckpt_dir, "policy_best.ckpt"))
    print(f"Best three-stage ckpt, stage 3 val loss {min_val_loss:.6f} @ epoch{best_epoch}")
    return stage3_best_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed, prefix=""):
    if not train_history or not validation_history:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"{prefix}train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key] for summary in train_history]
        val_values = [summary[key] for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label="train")
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label="validation")
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, required=True)
    parser.add_argument("--policy_class", action="store", type=str, required=True)
    parser.add_argument("--task_name", action="store", type=str, required=True)
    parser.add_argument("--dataset_dir", action="store", type=str, default=None)
    parser.add_argument("--num_episodes", action="store", type=int, default=None)
    parser.add_argument("--batch_size", action="store", type=int, required=True)
    parser.add_argument("--seed", action="store", type=int, required=True)
    parser.add_argument("--num_epochs", action="store", type=int, default=None)
    parser.add_argument("--two_stage", action="store_true")
    parser.add_argument("--three_stage", action="store_true")
    parser.add_argument("--stage1_epochs", action="store", type=int, default=None)
    parser.add_argument("--stage2_epochs", action="store", type=int, default=None)
    parser.add_argument("--stage3_epochs", action="store", type=int, default=None)
    parser.add_argument("--lr", action="store", type=float, required=True)
    parser.add_argument("--device", action="store", type=str, default="cuda")
    parser.add_argument("--kl_weight", action="store", type=float, required=False, default=10.0)
    parser.add_argument("--chunk_size", action="store", type=int, required=False, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, required=False, default=256)
    parser.add_argument("--dim_feedforward", action="store", type=int, required=False, default=3200)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--target_camera", action="store", type=str, default=None)
    parser.add_argument("--lambda_roi", action="store", type=float, default=1.0)
    parser.add_argument("--lambda_recon_grad", action="store", type=float, default=0.25)
    parser.add_argument("--lambda_sem", action="store", type=float, default=0.1)
    parser.add_argument("--lambda_sig", action="store", type=float, default=0.0)
    parser.add_argument("--roi_background_weight", action="store", type=float, default=1.0)
    parser.add_argument("--roi_detail_weight", action="store", type=float, default=10.0)
    parser.add_argument("--focus_masked_region", action="store_true")
    parser.add_argument("--comm_num_queries", action="store", type=int, default=8)
    parser.add_argument("--comm_layers", action="store", type=int, default=2)
    parser.add_argument("--comm_detach_warmup", action="store", type=int, default=0)
    parser.add_argument("--ema_momentum", action="store", type=float, default=0.99)
    parser.add_argument("--future_horizons", action="store", type=int, nargs="+", default=[0, 5, 15, 30, 60, 99])
    parser.add_argument("--future_image_height", action="store", type=int, default=240)
    parser.add_argument("--future_image_width", action="store", type=int, default=320)
    parser.add_argument("--future_layers", action="store", type=int, default=2)
    parser.add_argument("--future_teacher_mix_steps", action="store", type=int, default=10000)
    parser.add_argument("--future_rgb_decay_alpha", action="store", type=float, default=0.03)
    parser.add_argument("--future_latent_decay_alpha", action="store", type=float, default=0.01)
    parser.add_argument("--lambda_future_rgb", action="store", type=float, default=1.0)
    parser.add_argument("--lambda_future_grad", action="store", type=float, default=0.25)
    parser.add_argument("--lambda_future_latent", action="store", type=float, default=0.1)
    parser.add_argument("--resume_ckpt", action="store", type=str, default=None)
    parser.add_argument("--grad_accum_steps", action="store", type=int, default=1)
    parser.add_argument("--export_netron_onnx", action="store", type=str, default=None)
    parser.add_argument("--export_ckpt", action="store", type=str, default=None)
    main(vars(parser.parse_args()))
