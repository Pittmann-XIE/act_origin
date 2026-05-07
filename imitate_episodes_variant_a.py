import argparse
import cv2
import os
import pickle
import re
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from policy_variant_a import ACTPolicy, CNNMLPPolicy
from record_sim_episodes import get_target_geom_classes, render_target_mask
from utils_variant_a import (
    compute_dict_mean,
    load_data,
    sample_box_pose,
    sample_insertion_pose,
    set_seed,
)
from visualize_episodes import save_videos


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
            "comm_num_queries": args["comm_num_queries"],
            "comm_layers": args["comm_layers"],
            "comm_detach_warmup": args["comm_detach_warmup"],
            "ema_momentum": args["ema_momentum"],
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
    }

    if is_eval:
        results = []
        for ckpt_name in ["policy_best.ckpt"]:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        for ckpt_name, success_rate, avg_return in results:
            print(f"{ckpt_name}: {success_rate=} {avg_return=}")
        return

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        target_camera,
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as file_obj:
        pickle.dump(stats, file_obj)

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


def make_masked_roi(rgb_image, roi_mask):
    masked = rgb_image.astype(np.float32) * roi_mask[..., None].astype(np.float32)
    return np.clip(masked, 0.0, 255.0).astype(np.uint8)


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


def build_eval_frame(robot_view, pred_roi, gt_roi):
    pred_roi = cv2.resize(pred_roi, (320, 480), interpolation=cv2.INTER_AREA)
    gt_roi = cv2.resize(gt_roi, (320, 480), interpolation=cv2.INTER_AREA)
    robot_panel = add_panel_label(robot_view, "Robot rollout")
    roi_pred_panel = add_panel_label(pred_roi, "Pred ROI")
    roi_gt_panel = add_panel_label(gt_roi, "GT ROI")
    roi_panel = np.concatenate([roi_pred_panel, roi_gt_panel], axis=1)
    return np.concatenate([robot_panel, roi_panel], axis=1)


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

    if policy_class != "ACT":
        raise NotImplementedError("Variant A eval ROI visualization is implemented for ACT policy only.")
    if real_robot:
        raise NotImplementedError("Variant A eval ROI visualization currently supports simulation only.")

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
        raise NotImplementedError("Variant A eval ROI visualization currently supports sim_transfer_cube only.")
    target_geom_classes = get_target_geom_classes(env.physics)

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

        eval_frames = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
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
                roi_mask = render_target_mask(env.physics, target_camera, target_geom_classes) > 0
                gt_roi = make_masked_roi(obs_target_rgb, roi_mask)

                if config["policy_class"] == "ACT":
                    if t % query_frequency == 0:
                        all_actions, _, comm_outputs = policy(qpos, curr_image, return_comm=True)
                        roi_hat = comm_outputs.get("roi_hat")
                        pred_roi_full = tensor_to_uint8_image(roi_hat) if roi_hat is not None else np.zeros_like(gt_roi)
                        pred_roi = make_masked_roi(pred_roi_full, roi_mask)
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
                eval_frames.append(build_eval_frame(obs["images"][onscreen_cam], pred_roi, gt_roi))

            plt.close()
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
    image_data, qpos_data, action_data, is_pad, _, target_rgb, roi_weight_mask = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    target_rgb = target_rgb.to(device)
    roi_weight_mask = roi_weight_mask.to(device)
    return policy(qpos_data, image_data, action_data, is_pad, target_rgb, roi_weight_mask)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    device = config["device"]
    resume_ckpt = config.get("resume_ckpt")

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_class, policy)

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
        optimizer.zero_grad()
        epoch_start_idx = len(train_history)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            if hasattr(policy, "update_ema"):
                policy.update_ema()
            optimizer.zero_grad()
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


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    if not train_history or not validation_history:
        return
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
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
    parser.add_argument("--num_epochs", action="store", type=int, required=True)
    parser.add_argument("--lr", action="store", type=float, required=True)
    parser.add_argument("--device", action="store", type=str, default="cuda")
    parser.add_argument("--kl_weight", action="store", type=float, required=False, default=10.0)
    parser.add_argument("--chunk_size", action="store", type=int, required=False, default=100)
    parser.add_argument("--hidden_dim", action="store", type=int, required=False, default=256)
    parser.add_argument("--dim_feedforward", action="store", type=int, required=False, default=3200)
    parser.add_argument("--temporal_agg", action="store_true")
    parser.add_argument("--target_camera", action="store", type=str, default=None)
    parser.add_argument("--lambda_roi", action="store", type=float, default=1.0)
    parser.add_argument("--lambda_sem", action="store", type=float, default=0.1)
    parser.add_argument("--lambda_sig", action="store", type=float, default=0.0)
    parser.add_argument("--comm_num_queries", action="store", type=int, default=8)
    parser.add_argument("--comm_layers", action="store", type=int, default=2)
    parser.add_argument("--comm_detach_warmup", action="store", type=int, default=0)
    parser.add_argument("--ema_momentum", action="store", type=float, default=0.99)
    parser.add_argument("--resume_ckpt", action="store", type=str, default=None)
    main(vars(parser.parse_args()))
