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
from utils_dino_bisim_track import load_data, sample_box_pose, sample_insertion_pose, compute_dict_mean, set_seed, detach_dict 
from policy_dino_bisim_track import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

def update_mujoco_lighting(env, timestep, mode='dynamic'):
    if not hasattr(env, '_physics') or env._physics is None: return
    physics = env._physics
    if mode == 'dynamic':
        brightness_scalar = 0.5 + 0.4 * np.sin(timestep * 0.05) 
    elif mode == 'random':
        brightness_scalar = np.random.uniform(0.2, 0.9)
    else:
        return
    physics.model.light_ambient[0] = np.array([brightness_scalar, brightness_scalar, brightness_scalar])

def main(args):
    set_seed(1)
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    device = args['device'] 

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

    state_dim = 14
    lr_backbone = 1e-5
    backbone = args.get('backbone', 'resnet18')  
    dinov3_downsample = args.get('dinov3_downsample', False)
    # --- UPDATE THIS LINE ---
    lr_backbone = 1e-5 if backbone != 'dinov3' else 0.0
    # ------------------------
    
    if policy_class == 'ACT':
        policy_config = {
            'lr': args['lr'],
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone': lr_backbone,
            'backbone': backbone,
            'dinov3_downsample': dinov3_downsample,
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': camera_names,
            'bisim_weight': args.get('bisim_weight', 1.0),
            'layer_to_align': args.get('layer_to_align', 6),
            'gaze_weight': args.get('gaze_weight', 1.0)
        }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'dinov3_downsample': dinov3_downsample, 'num_queries': 1, 'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs, 'ckpt_dir': ckpt_dir, 'episode_len': episode_len, 'state_dim': state_dim,
        'lr': args['lr'], 'policy_class': policy_class, 'onscreen_render': onscreen_render,
        'policy_config': policy_config, 'task_name': task_name, 'seed': args['seed'],
        'temporal_agg': args['temporal_agg'], 'camera_names': camera_names, 'real_robot': not is_sim, 'device': device 
    }

    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        exit()

    chunk_size = policy_config['num_queries']
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunk_size)

    if not os.path.isdir(ckpt_dir): os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, f'dataset_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    torch.save(best_ckpt_info[2], os.path.join(ckpt_dir, f'policy_best.ckpt'))
    print(f'Best ckpt, val loss {best_ckpt_info[1]:.6f} @ epoch{best_ckpt_info[0]}')

def make_policy(policy_class, policy_config):
    return ACTPolicy(policy_config) if policy_class == 'ACT' else CNNMLPPolicy(policy_config)

def make_optimizer(policy_class, policy):
    return policy.configure_optimizers()

def get_image(ts, camera_names, device='cuda'): 
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    return torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir, state_dim, real_robot, policy_class, onscreen_render, policy_config, camera_names, max_timesteps, task_name, temporal_agg, device = \
        [config[k] for k in ['ckpt_dir', 'state_dim', 'real_robot', 'policy_class', 'onscreen_render', 'policy_config', 'camera_names', 'episode_len', 'task_name', 'temporal_agg', 'device']]
    
    onscreen_cam = 'angle'
    policy = make_policy(policy_class, policy_config)
    device_obj = torch.device(device)
    policy.load_state_dict(torch.load(os.path.join(ckpt_dir, ckpt_name), map_location=device_obj))
    policy.to(device_obj).eval()
    
    with open(os.path.join(ckpt_dir, f'dataset_stats.pkl'), 'rb') as f: stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    if real_robot:
        from aloha_scripts.real_env import make_real_env
        from aloha_scripts.robot_utils import move_grippers
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = 1 if temporal_agg else policy_config['num_queries']
    num_queries = policy_config['num_queries']
    num_rollouts = 50
    episode_returns, highest_rewards = [], []

    for rollout_id in range(num_rollouts):
        if 'sim_transfer_cube' in task_name:
            def sample_non_colliding_pose(existing_poses, min_dist=0.06, max_tries=100):
                for _ in range(max_tries):
                    pose = sample_box_pose()
                    if not any(np.linalg.norm(pose[:2] - ep[:2]) < min_dist for ep in existing_poses): return pose
                return sample_box_pose()
            p_red = sample_box_pose()
            p_dist1 = sample_non_colliding_pose([p_red])
            p_dist2 = sample_non_colliding_pose([p_red, p_dist1])
            p_dist3 = sample_non_colliding_pose([p_red, p_dist1, p_dist2])
            BOX_POSE[0] = np.concatenate([p_red, p_dist1, p_dist2, p_dist3])
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) 

        ts = env.reset()
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        if temporal_agg: all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).to(device_obj)
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).to(device_obj)
        image_list, qpos_list, target_qpos_list, rewards = [], [], [], []

        with torch.inference_mode():
            for t in range(max_timesteps):
                if not real_robot:
                    update_mujoco_lighting(env, t, mode='dynamic')
                    for cam_name in camera_names:
                        ts.observation['images'][cam_name] = env._physics.render(height=480, width=640, camera_id=cam_name)
                
                if onscreen_render:
                    plt_img.set_data(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
                    plt.pause(DT)

                obs = ts.observation
                image_list.append(obs['images'] if 'images' in obs else {'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = torch.from_numpy(pre_process(qpos_numpy)).float().to(device_obj).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names, device_obj) 

                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions, attn_weights = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_for_curr_step = actions_for_curr_step[torch.all(actions_for_curr_step != 0, axis=1)]
                        exp_weights = np.exp(-0.01 * np.arange(len(actions_for_curr_step)))
                        exp_weights = torch.from_numpy(exp_weights / exp_weights.sum()).to(device_obj).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)

                action = post_process(raw_action.squeeze(0).cpu().numpy())
                ts = env.step(action)
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(action)
                rewards.append(ts.reward)
            plt.close()

        if real_robot: move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5) 
        
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        highest_rewards.append(np.max(rewards))
        if save_episode: save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id+1}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    with open(os.path.join(ckpt_dir, f'result_{ckpt_name.split(".")[0]}.txt'), 'w') as f:
        f.write(f'Success rate: {success_rate}\nAverage return: {avg_return}\n\n')
    return success_rate, avg_return

def forward_pass(data, policy, device='cuda'): 
    # Handles dynamic dataset schemas efficiently depending on configs merged.
    # Max configuration matching merged pipeline (Vision + Bisim + Track): Image, Qpos, Action, Is_Pad, Box, Gaze, Next_Image, Next_Qpos, Valid_Next
    if len(data) >= 9:
        image_data, qpos_data, action_data, is_pad, _, gaze_data, image_data_next, qpos_data_next, valid_next = data[:9]
        image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
        return policy(qpos_data, image_data, action_data, is_pad, gaze_data=gaze_data.to(device), next_qpos=qpos_data_next.to(device), next_image=image_data_next.to(device), valid_next=valid_next.to(device))
    elif len(data) == 8: 
        image_data, qpos_data, action_data, is_pad, _, image_data_next, qpos_data_next, valid_next = data
        return policy(qpos_data.to(device), image_data.to(device), action_data.to(device), is_pad.to(device), next_qpos=qpos_data_next.to(device), next_image=image_data_next.to(device), valid_next=valid_next.to(device))
    elif len(data) == 6: 
        image_data, qpos_data, action_data, is_pad, _, gaze_data = data
        return policy(qpos_data.to(device), image_data.to(device), action_data.to(device), is_pad.to(device), gaze_data=gaze_data.to(device))
    else: 
        image_data, qpos_data, action_data, is_pad, _ = data
        return policy(qpos_data.to(device), image_data.to(device), action_data.to(device), is_pad.to(device))

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs, ckpt_dir, seed, policy_class, policy_config, device = config['num_epochs'], config['ckpt_dir'], config['seed'], config['policy_class'], config['policy_config'], config['device']
    set_seed(seed)
    policy = make_policy(policy_class, policy_config).to(device) 
    optimizer = make_optimizer(policy_class, policy)

    train_history, validation_history = [], []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        
        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_summary = compute_dict_mean([forward_pass(data, policy, device) for data in val_dataloader])
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

        # Training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device) 
            forward_dict['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            
        # Calculate training metrics for the epoch
        epoch_train_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_train_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_train_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # Checkpoint and plot at intervals
        if epoch % 100 == 0:
            torch.save(policy.state_dict(), os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt'))
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            
    # Save the final checkpoint
    torch.save(policy.state_dict(), os.path.join(ckpt_dir, f'policy_last.ckpt'))
    
    # Save best checkpoint state distinctively 
    best_epoch, min_val_loss_val, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss_val:.6f} at epoch {best_epoch}')
    
    # Save final training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """Saves plots of the training and validation history to the checkpoint directory."""
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
        plt.close() # Close to avoid consuming memory
    print(f'Saved plots to {ckpt_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, required=True)
    parser.add_argument('--policy_class', action='store', type=str, required=True)
    parser.add_argument('--task_name', action='store', type=str, required=True)
    parser.add_argument('--batch_size', action='store', type=int, required=True)
    parser.add_argument('--seed', action='store', type=int, required=True)
    parser.add_argument('--num_epochs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--device', action='store', type=str, default='cuda:0')
    parser.add_argument('--kl_weight', action='store', type=int)
    parser.add_argument('--chunk_size', action='store', type=int)
    parser.add_argument('--hidden_dim', action='store', type=int)
    parser.add_argument('--dim_feedforward', action='store', type=int)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--backbone', action='store', type=str, default='resnet18')
    parser.add_argument('--dinov3_downsample', action='store_true')
    parser.add_argument('--bisim_weight', action='store', type=float, default=0.5)
    parser.add_argument('--layer_to_align', action='store', type=int, default=4)
    parser.add_argument('--gaze_weight', action='store', type=float, default=1.0)
    main(vars(parser.parse_args()))