import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

# Import the new ENABLE_DISTRACTOR flag
from sim_env import BOX_POSE, ENABLE_DISTRACTOR

import IPython
e = IPython.embed

import matplotlib.pyplot as plt
import cv2

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print("Error importing SAM 2.1 predictor. Please ensure the SAM repository is correctly set up and the path is correct.")

def get_user_click(image_rgb):
    """Displays the first frame and waits for the user to click the object."""
    print("Please click on the object to track, then close the window.")
    plt.imshow(image_rgb)
    # ginput(1) gets one click. timeout=0 means it waits forever.
    clicked_points = plt.ginput(1, timeout=0) 
    plt.close()
    
    if not clicked_points:
        raise ValueError("No click detected! Please click the object.")
    
    # Return as [x, y]
    return [int(clicked_points[0][0]), int(clicked_points[0][1])]

def get_square_crop(raw_frame, mask, padding=10, target_size=(640, 480)):
    """Extracts a square crop based on the mask and resizes to target_size."""
    img_h, img_w, _ = raw_frame.shape
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        return cv2.resize(raw_frame, target_size) # Fallback if tracking fails
        
    x_min_t = max(0, x_indices.min() - padding)
    x_max_t = min(img_w - 1, x_indices.max() + padding)
    y_min_t = max(0, y_indices.min() - padding)
    y_max_t = min(img_h - 1, y_indices.max() + padding)
    
    # Create a SQUARE crop (Aspect Ratio Preservation)
    side = max(x_max_t - x_min_t, y_max_t - y_min_t)
    cx, cy = (x_min_t + x_max_t) / 2, (y_min_t + y_max_t) / 2
    
    x_min_s = int(max(0, cx - side / 2))
    x_max_s = int(min(img_w - 1, cx + side / 2))
    y_min_s = int(max(0, cy - side / 2))
    y_max_s = int(min(img_h - 1, cy + side / 2))
    
    crop = raw_frame[y_min_s:y_max_s, x_min_s:x_max_s]
    
    # Handle edge cases where the crop is empty (e.g., object leaves frame)
    if crop.size == 0:
        return cv2.resize(raw_frame, target_size)
        
    return cv2.resize(crop, target_size)

def visualize_attention(curr_image, attn_weights, camera_names):
    """
    Overlay attention map on the current image.
    curr_image: Tensor (1, num_cameras, C, H, W)
    attn_weights: Tensor (1, num_queries, seq_len)
    """
    # 1. Process Images
    curr_images_np = curr_image.detach().cpu().numpy().squeeze(0).transpose(0, 2, 3, 1)

    # 2. Process Attention Weights
    attn_weights = attn_weights.detach().cpu().numpy().squeeze(0)
    
    # Average attention across all queries
    attn_map = np.mean(attn_weights, axis=0) 
    
    # 3. Handle ACT's specific formatting
    # ACT adds 2 extra tokens (latent_input and proprio_input) at the front
    attn_map = attn_map[2:] 
    
    num_cams = len(camera_names)
    h_feat, w_feat = 15, 20
    
    # Because detr_vae.py concatenates along width (axis=3), the flattened sequence
    # represents a combined wide image of size (h_feat, w_feat * num_cams).
    # We must reshape it to this wide format first!
    combined_attn_2d = attn_map.reshape((h_feat, w_feat * num_cams))
    
    vis_images = []
    
    for cam_idx in range(num_cams):
        # Crop the specific camera's attention map out of the combined wide map
        start_col = cam_idx * w_feat
        end_col = (cam_idx + 1) * w_feat
        cam_attn_2d = combined_attn_2d[:, start_col:end_col]
        
        # Normalize attention map 0-1
        cam_attn_2d = cam_attn_2d - cam_attn_2d.min()
        cam_attn_2d = cam_attn_2d / (cam_attn_2d.max() + 1e-8)
        
        # Resize to original image size
        img_h, img_w = curr_images_np[cam_idx].shape[:2]
        cam_attn_resized = cv2.resize(cam_attn_2d, (img_w, img_h))
        
        # Apply Colormap (JET)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_attn_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
        
        # Overlay: 60% Image, 40% Heatmap
        original_img = curr_images_np[cam_idx]
        overlayed_img = 0.6 * original_img + 0.4 * heatmap
        
        vis_images.append(overlayed_img)
        
    # Concatenate cameras horizontally
    return np.concatenate(vis_images, axis=1)

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'box_weight': 0.5
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

#     return success_rate, avg_return
def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(config['seed'])
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # Check if SAM 2 is needed based on camera names containing 'cropped'
    use_sam2 = any('cropped' in cam for cam in camera_names)

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    
    # 1. Initialize SAM 2 Small conditionally
    if use_sam2:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_s.yaml", # SMALL MODEL
            "/home/pengtao/ws_ros2humble-main_lab/sam2/checkpoints/sam2.1_hiera_small.pt", 
            device=device
        )
    else:
        sam2_predictor = None

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    
    for rollout_id in range(num_rollouts):
        print(f"\n--- Starting Rollout {rollout_id} ---")
        
        # 1. Sample the static box position ONCE for this rollout so it's identical for both runs
        if 'sim_transfer_cube' in task_name:
            base_box_pose = sample_box_pose() 
        elif 'sim_insertion' in task_name:
            base_box_pose = np.concatenate(sample_insertion_pose()) 
        else:
            base_box_pose = None

        # 2. Run the full episode twice: once without distractor, once with
        for has_distractor in [False, True]:
            # Set the flag and reset the target box pose
            ENABLE_DISTRACTOR[0] = has_distractor
            if base_box_pose is not None:
                BOX_POSE[0] = base_box_pose.copy()
            
            ts = env.reset()

            ### --- SAM 2 INITIALIZATION FOR THIS EPISODE --- ###
            if use_sam2:
                # Assuming the primary uncropped camera is the first one in the config to track from
                track_cam_name = [cam for cam in camera_names if 'cropped' not in cam][0] if any('cropped' not in cam for cam in camera_names) else camera_names[0]
                first_frame = ts.observation['images'][track_cam_name]
                
                # Prompt user to click the object
                print(f"Waiting for user to click object in {track_cam_name}...")
                click_pt = get_user_click(first_frame)
                
                # Use the Process ID (PID) to create a unique directory path
                video_dir = f"/tmp/sam2_temp_rollout_{os.getpid()}"
                os.makedirs(video_dir, exist_ok=True)
                
                # Clear old frames from this specific instance to avoid confusion
                for f in os.listdir(video_dir):
                    os.remove(os.path.join(video_dir, f))
                
                # Save the first frame as 00000.jpg
                cv2.imwrite(os.path.join(video_dir, "00000.jpg"), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
                
                # Offload video and state to CPU to save massive amounts of GPU VRAM
                inference_state = sam2_predictor.init_state(
                    video_path=video_dir,
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=True
                )
                
                sam2_predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=np.array([click_pt], dtype=np.float32),
                    labels=np.array([1], np.int32)
                )
            ### --------------------------------------------- ###

            ### onscreen render setup
            if onscreen_render:
                ax = plt.subplot()
                plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
                plt.ion()

            if temporal_agg:
                all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

            qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
            image_list = [] 
            qpos_list = []
            target_qpos_list = []
            rewards = []
            
            attn_vis_list = []
            curr_attn_weights = None
            curr_box_hat = None

            print(f"Running evaluation (Distractor: {has_distractor})...")
            with torch.inference_mode():
                for t in range(max_timesteps):
                    if onscreen_render:
                        image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                        plt_img.set_data(image)
                        plt.pause(DT)

                    obs = ts.observation
                    
                   ### --- FIXED SAM 2 TRACKING & CROPPING --- ###
                    if use_sam2:
                        current_frame = obs['images'][track_cam_name]
                        
                        if t > 0:
                            img_resized = cv2.resize(
                                current_frame, 
                                (sam2_predictor.image_size, sam2_predictor.image_size)
                            )
                            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
                            img_tensor = img_tensor.unsqueeze(0).cpu()
                            inference_state["images"] = torch.cat([inference_state["images"], img_tensor], dim=0)
                            inference_state["num_frames"] = inference_state["images"].shape[0]
                        
                        mask = None
                        for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(
                            inference_state, 
                            start_frame_idx=t, 
                            max_frame_num_to_track=1
                        ):
                            mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                        
                        if mask is not None:
                            cropped_img = get_square_crop(current_frame, mask, padding=10, target_size=(640, 480))
                        else:
                            cropped_img = cv2.resize(current_frame, (640, 480))
                        
                        crop_cam_target = [cam for cam in camera_names if 'cropped' in cam][0]
                        obs['images'][crop_cam_target] = cropped_img
                    ### --------------------------------------- ###
                        
                    qpos_numpy = np.array(obs['qpos'])
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    qpos_history[:, t] = qpos
                    
                    curr_image = get_image(ts, camera_names)

                    # Query policy
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            all_actions, curr_attn_weights, curr_box_hat = policy(qpos, curr_image)
                        
                        if temporal_agg:
                            all_time_actions[[t], t:t+num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                        curr_attn_weights = None
                        curr_box_hat = None
                    else:
                        raise NotImplementedError

                    ### --- DRAW BOUNDING BOX ON RAW OBSERVATION & ATTENTION MAP --- ###
                    if curr_box_hat is not None:
                        # Convert YOLO format (cx, cy, w, h) to standard (x1, y1, x2, y2)
                        box = curr_box_hat.squeeze().cpu().numpy()  
                        
                        # Select the base camera image to draw on
                        draw_cam_name = [cam for cam in camera_names if 'cropped' not in cam][0] if any('cropped' not in cam for cam in camera_names) else camera_names[0]
                        
                        if draw_cam_name in obs['images']:
                            img_to_draw = obs['images'][draw_cam_name].copy()
                            img_h, img_w = img_to_draw.shape[:2]
                            
                            cx, cy, bw, bh = box
                            xmin = int((cx - bw / 2) * img_w)
                            ymin = int((cy - bh / 2) * img_h)
                            xmax = int((cx + bw / 2) * img_w)
                            ymax = int((cy + bh / 2) * img_h)
                            
                            text = f"Box: ({cx:.2f}, {cy:.2f})"
                            
                            # 1. Draw on Original Image
                            cv2.rectangle(img_to_draw, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            cv2.putText(img_to_draw, text, (xmin, max(ymin - 10, 10)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                            obs['images'][draw_cam_name] = img_to_draw
                            
                            # 2. Draw on Attention Map (if enabled)
                            if save_episode and curr_attn_weights is not None:
                                vis_img = visualize_attention(curr_image, curr_attn_weights, camera_names)
                                
                                # Convert float image to uint8 so OpenCV can draw on it
                                vis_img_uint8 = np.uint8(vis_img * 255).copy()
                                
                                # Calculate horizontal offset for the correct camera panel
                                cam_idx = camera_names.index(draw_cam_name)
                                x_offset = cam_idx * img_w
                                
                                # Draw the Box
                                cv2.rectangle(vis_img_uint8, (xmin + x_offset, ymin), (xmax + x_offset, ymax), (0, 255, 0), 2)
                                
                                # Draw High-Contrast Text Background
                                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(vis_img_uint8, 
                                              (xmin + x_offset, max(ymin - 10, 10) - text_h - 2), 
                                              (xmin + x_offset + text_w, max(ymin - 10, 10) + 2), 
                                              (0, 0, 0), -1)  # Solid black background
                                              
                                # Draw Text
                                cv2.putText(vis_img_uint8, text, (xmin + x_offset, max(ymin - 10, 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                                
                                attn_vis_list.append(vis_img_uint8)
                    else:
                        # Fallback if no box is predicted
                        if save_episode and curr_attn_weights is not None:
                            vis_img = visualize_attention(curr_image, curr_attn_weights, camera_names)
                            attn_vis_list.append(np.uint8(vis_img * 255))
                    ### ------------------------------------------------------------ ###

                    if 'images' in obs:
                        image_list.append({k: v.copy() for k, v in obs['images'].items()})
                    else:
                        image_list.append({'main': obs['image']})

                    # Post-process actions
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action

                    # Step the environment
                    ts = env.step(target_qpos)

                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(ts.reward)

                if onscreen_render:
                    plt.close()

            if real_robot:
                from aloha_scripts.robot_utils import move_grippers
                from constants import PUPPET_GRIPPER_JOINT_OPEN
                move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

            # Record stats
            rewards = np.array(rewards)
            episode_return = np.sum(rewards[rewards!=None])
            episode_highest_reward = np.max(rewards)
            
            if has_distractor:
                episode_returns.append(episode_return)
                highest_rewards.append(episode_highest_reward)
                
            print(f'Result -> Return: {episode_return}, Highest Reward: {episode_highest_reward}, Success: {episode_highest_reward==env_max_reward}')

            # 3. Save the videos immediately after the run completes
            if save_episode:
                suffix = "with_distractor" if has_distractor else "no_distractor"
                
                # Save standard top/angle view video
                save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video_{suffix}_{rollout_id}.mp4'))
                
                # Save the full attention video
                if len(attn_vis_list) > 0:
                    print(f"Saving attention video ({suffix}) for rollout {rollout_id}...")
                    attn_video_path = os.path.join(ckpt_dir, f'video_attn_{suffix}_{rollout_id}.mp4')
                    
                    # NOTE: Images are ALREADY uint8 from our drawing logic above
                    attn_frames = attn_vis_list 
                    
                    h, w, _ = attn_frames[0].shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(attn_video_path, fourcc, int(1/DT), (w, h))
                    
                    for frame in attn_frames:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    out.release()
                
                # Cleanup SAM 2 inference state at the end of every episode loop 
                if use_sam2:
                    sam2_predictor.reset_state(inference_state)
                    del inference_state
                    torch.cuda.empty_cache()

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + str(os.getpid()) + '_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    # image_data, qpos_data, action_data, is_pad = data
    # image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None
    if len(data) == 5:
        image_data, qpos_data, action_data, is_pad, box_data = data
        image_data, qpos_data, action_data, is_pad, box_data = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), box_data.cuda()
        return policy(qpos_data, image_data, action_data, is_pad, box_data=box_data)
    else:
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))
