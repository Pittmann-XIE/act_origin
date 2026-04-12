import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import cv2

import IPython
e = IPython.embed

def get_attention_weights(reward):
    """Returns cognitive attention weights based on task progress."""
    if reward == 0:
        return [1.0, 0.2, 0.2, 1.0, 1.0] # Reaching (Right arm active)
    elif reward == 1 or reward == 2:
        return [1.0, 0.5, 0.5, 0.8, 0.8] # Grasped
    elif reward == 3:
        return [1.0, 1.0, 1.0, 1.0, 1.0] # Transferring (Both arms)
    elif reward == 4:
        return [1.0, 1.0, 1.0, 0.2, 0.2] # Success (Left arm active)
    else:
        return [1.0, 1.0, 1.0, 1.0, 1.0]

def generate_gaze_heatmap(points, reward, camera_names, sigma=1.0):
    """Generates the multi-camera 1D flattened heatmap for ACT vision tokens."""
    W_img, H_img = 640.0, 480.0
    W_feat, H_feat = 20, 15
    num_cameras = len(camera_names)
    
    # Shape: (num_cameras, 15, 20)
    G_target = torch.zeros((num_cameras, H_feat, W_feat), dtype=torch.float32)
    
    # We only have spatial tracking for the 'top' camera. 
    # Find its index in the camera sequence.
    if 'top' in camera_names:
        cam_idx = camera_names.index('top')
    elif 'top_cropped_1' in camera_names: # Handle your custom names
        cam_idx = camera_names.index('top_cropped_1')
    else:
        return G_target.view(-1) # Return zeros if top camera isn't used
        
    weights = get_attention_weights(reward)
    
    x_grid = torch.arange(W_feat, dtype=torch.float32)
    y_grid = torch.arange(H_feat, dtype=torch.float32)
    y_map, x_map = torch.meshgrid(y_grid, x_grid, indexing='ij')
    
    cam_heatmap = torch.zeros((H_feat, W_feat), dtype=torch.float32)
    
    # Superimpose Gaussian for each tracked point
    for pt_idx, pt in enumerate(points):
        x_raw, y_raw = pt
        if x_raw != -1 and y_raw != -1: # Ignore out-of-bounds points
            x_feat = x_raw * (W_feat / W_img)
            y_feat = y_raw * (H_feat / H_img)
            
            squared_dist = (x_map - x_feat)**2 + (y_map - y_feat)**2
            
            # Since we only pass 1 point (YOLO center), it will use weights[0]
            weight = weights[pt_idx] if pt_idx < len(weights) else 1.0
            gaussian_2d = weight * torch.exp(-squared_dist / (2 * sigma**2))
            
            # Use maximum to prevent additive overlapping
            cam_heatmap = torch.maximum(cam_heatmap, gaussian_2d)
            
    G_target[cam_idx] = cam_heatmap
    
    # Flatten to match ACT's vision tokens (e.g., num_cameras * 300)
    return G_target.view(-1)


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, chunk_size):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
                
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            
            for cam_name in self.camera_names:
                if cam_name == 'top_cropped_1':
                    img = root[f'/observations/images/{cam_name}'][0]
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                if img.shape[0] != 480 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                image_dict[cam_name] = img
                
            # get all actions and rewards after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
                
                if '/reward' in root:
                    reward_seq = root['/reward'][start_ts:]
                else:
                    reward_seq = None
            else:
                action = root['/action'][max(0, start_ts - 1):] 
                action_len = episode_len - max(0, start_ts - 1) 
                reward_seq = None

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # Process sequence of Gaze Heatmaps using YOLO bounding box centers
        gaze_heatmaps = []
        for t in range(self.chunk_size): 
            current_ts = start_ts + t
            x_raw, y_raw = -1, -1 # Default out-of-bounds
            
            if current_ts < episode_len:
                for cam_name in self.camera_names:
                    label_path = os.path.join(self.dataset_dir, 'yolo_labels', f'episode_{episode_id}', cam_name, f'{int(current_ts):05d}.txt')
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 3 and parts[0] == '0':
                                    # Convert normalized YOLO coordinates to raw pixel coordinates
                                    x_raw = float(parts[1]) * 640.0
                                    y_raw = float(parts[2]) * 480.0
                                    break
                        
                        if x_raw != -1:
                            break # Found class 0 box, stop looking in other cameras
                            
            current_reward = reward_seq[t] if (reward_seq is not None and t < action_len) else 0
            
            # Pass the extracted center point into the heatmap generator
            hm = generate_gaze_heatmap([[x_raw, y_raw]], current_reward, self.camera_names, sigma=1.0)
            gaze_heatmaps.append(hm)
            
        # Shape is now natively (chunk_size, num_cameras * 300)
        gaze_data = torch.stack(gaze_heatmaps)

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        
        # Load single-frame YOLO bounding box label (Kept from your original script for box_tensor)
        box_data = np.zeros(5, dtype=np.float32)
        for cam_name in self.camera_names:
            label_path = os.path.join(self.dataset_dir, 'yolo_labels', f'episode_{episode_id}', cam_name, f'{int(start_ts):05d}.txt')
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5 and parts[0] == '0':
                            box_data = np.array([float(x) for x in parts[:5]], dtype=np.float32)
                            break 
                if np.any(box_data[1:]): 
                    break
        box_tensor = torch.from_numpy(box_data).float()

        return image_data, qpos_data, action_data, is_pad, box_tensor, gaze_data

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,chunk_size):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, chunk_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)