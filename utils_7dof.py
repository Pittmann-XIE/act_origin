# import numpy as np
# import torch
# import os
# import h5py
# from torch.utils.data import TensorDataset, DataLoader

# import IPython
# e = IPython.embed

# class EpisodicDataset(torch.utils.data.Dataset):
#     def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
#         super(EpisodicDataset).__init__()
#         self.episode_ids = episode_ids
#         self.dataset_dir = dataset_dir
#         self.camera_names = camera_names
#         self.norm_stats = norm_stats
#         self.is_sim = None
#         self.max_ep_len = norm_stats.get("max_ep_len", 1000) # ADDED: Get global max length
#         self.__getitem__(0)

#     def __len__(self):
#         return len(self.episode_ids)

#     def __getitem__(self, index):
#         sample_full_episode = False

#         episode_id = self.episode_ids[index]
#         dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             is_sim = root.attrs['sim']
#             original_action_shape = root['/action'].shape
#             episode_len = original_action_shape[0]
#             if sample_full_episode:
#                 start_ts = 0
#             else:
#                 start_ts = np.random.choice(episode_len)
            
#             qpos = root['/observations/qpos'][start_ts]
#             qvel = root['/observations/qvel'][start_ts]
#             image_dict = dict()
#             for cam_name in self.camera_names:
#                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
#             if is_sim:
#                 action = root['/action'][start_ts:]
#                 action_len = episode_len - start_ts
#             else:
#                 action = root['/action'][max(0, start_ts - 1):]
#                 action_len = episode_len - max(0, start_ts - 1)

#         self.is_sim = is_sim
        
#         # CHANGED: Pad up to max_ep_len instead of the local original_action_shape
#         padded_action = np.zeros((self.max_ep_len, original_action_shape[1]), dtype=np.float32)
#         padded_action[:action_len] = action
        
#         # CHANGED: Create pad mask up to max_ep_len
#         is_pad = np.zeros(self.max_ep_len)
#         is_pad[action_len:] = 1

#         # new axis for different cameras
#         all_cam_images = []
#         for cam_name in self.camera_names:
#             all_cam_images.append(image_dict[cam_name])
#         all_cam_images = np.stack(all_cam_images, axis=0)

#         # construct observations
#         image_data = torch.from_numpy(all_cam_images)
#         qpos_data = torch.from_numpy(qpos).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()

#         # channel last
#         image_data = torch.einsum('k h w c -> k c h w', image_data)

#         # normalize image and change dtype to float
#         image_data = image_data / 255.0
#         action_data = ((action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]).float()
#         qpos_data = ((qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]).float()

#         return image_data, qpos_data, action_data, is_pad
    
# def get_norm_stats(dataset_dir, num_episodes):
#     all_qpos_data = []
#     all_action_data = []
#     max_ep_len = 0  # ADDED: Track the longest episode
    
#     for episode_idx in range(num_episodes):
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()]
#             qvel = root['/observations/qvel'][()]
#             action = root['/action'][()]
            
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))
        
#         # ADDED: Update max episode length
#         max_ep_len = max(max_ep_len, len(qpos))

#     # CHANGED: Use torch.cat instead of torch.stack to handle variable lengths
#     all_qpos_data = torch.cat(all_qpos_data, dim=0)
#     all_action_data = torch.cat(all_action_data, dim=0)

#     # CHANGED: Calculate mean/std over dim 0 because data is now concatenated
#     action_mean = all_action_data.mean(dim=[0], keepdim=True)
#     action_std = all_action_data.std(dim=[0], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf)

#     qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

#     stats = {
#         "action_mean": action_mean.numpy().squeeze(), 
#         "action_std": action_std.numpy().squeeze(),
#         "qpos_mean": qpos_mean.numpy().squeeze(), 
#         "qpos_std": qpos_std.numpy().squeeze(),
#         "example_qpos": qpos,
#         "max_ep_len": max_ep_len  # ADDED: Pass this to the Dataset for safe padding
#     }

#     return stats

# def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
#     print(f'\nData from: {dataset_dir}\n')
#     # obtain train test split
#     train_ratio = 0.8
#     shuffled_indices = np.random.permutation(num_episodes)
#     train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
#     val_indices = shuffled_indices[int(train_ratio * num_episodes):]

#     # obtain normalization stats for qpos and action
#     norm_stats = get_norm_stats(dataset_dir, num_episodes)

#     # construct dataset and dataloader
#     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
#     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

#     return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# ### env utils

# def sample_box_pose():
#     x_range = [0.0, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     cube_quat = np.array([1, 0, 0, 0])
#     return np.concatenate([cube_position, cube_quat])

# def sample_insertion_pose():
#     # Peg
#     x_range = [0.1, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     peg_quat = np.array([1, 0, 0, 0])
#     peg_pose = np.concatenate([peg_position, peg_quat])

#     # Socket
#     x_range = [-0.2, -0.1]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     socket_quat = np.array([1, 0, 0, 0])
#     socket_pose = np.concatenate([socket_position, socket_quat])

#     return peg_pose, socket_pose

# ### helper functions

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result

# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d

# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)


## color jitter
import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms # ADDED: Import transforms

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, augment_images=False): # ADDED: augment_images flag
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.max_ep_len = norm_stats.get("max_ep_len", 1000)
        self.augment_images = augment_images # ADDED
        
        # ADDED: Define ColorJitter transform
        # These values (0.3) are standard for robotic manipulation robustness
        self.transform = transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3, 
            saturation=0.3, 
            hue=0.05
        )
        
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

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
            
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        
        padded_action = np.zeros((self.max_ep_len, original_action_shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        
        is_pad = np.zeros(self.max_ep_len)
        is_pad[action_len:] = 1

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

        # ADDED: Apply color jitter if augmentation is enabled
        if self.augment_images:
            # Apply jitter to each camera independently to ensure maximum robustness
            # (If applied to the batch k together, they would share the same brightness shift)
            augmented_images = []
            for k in range(image_data.shape[0]):
                augmented_images.append(self.transform(image_data[k]))
            image_data = torch.stack(augmented_images)

        action_data = ((action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]).float()
        qpos_data = ((qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]).float()

        return image_data, qpos_data, action_data, is_pad
    
def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    max_ep_len = 0
    
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        
        max_ep_len = max(max_ep_len, len(qpos))

    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {
        "action_mean": action_mean.numpy().squeeze(), 
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(), 
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
        "max_ep_len": max_ep_len
    }

    return stats

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    # CHANGED: Enable augmentation for training, disable for validation
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, augment_images=True)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, augment_images=False)
    
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