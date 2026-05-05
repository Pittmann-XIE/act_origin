import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, target_camera):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.target_camera = target_camera
        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            if "/observations/masks" not in root:
                raise ValueError(
                    f"Variant A requires recorded masks, but {dataset_path} has no /observations/masks group."
                )
            if self.target_camera not in root["/observations/masks"]:
                raise ValueError(
                    f"Variant A requires masks for target camera {self.target_camera}, but they are missing in {dataset_path}."
                )

            is_sim = root.attrs["sim"]
            original_action_shape = root["/action"].shape
            episode_len = original_action_shape[0]
            start_ts = 0 if sample_full_episode else np.random.choice(episode_len)

            qpos = root["/observations/qpos"][start_ts]
            image_dict = {}
            for cam_name in self.camera_names:
                img = root[f"/observations/images/{cam_name}"][start_ts]
                if img.shape[0] != 480 or img.shape[1] != 640:
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                image_dict[cam_name] = img
            target_mask = root[f"/observations/masks/{self.target_camera}"][start_ts]

            if is_sim:
                action = root["/action"][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root["/action"][max(0, start_ts - 1) :]
                action_len = episode_len - max(0, start_ts - 1)

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len, dtype=np.bool_)
        is_pad[action_len:] = True

        all_cam_images = np.stack([image_dict[cam_name] for cam_name in self.camera_names], axis=0)
        target_rgb = image_dict[self.target_camera]
        roi_weight_mask = (target_mask > 0).astype(np.float32)
        if roi_weight_mask.sum() == 0:
            raise ValueError(
                f"Variant A expected non-empty ROI mask in {dataset_path} for camera {self.target_camera} at step {start_ts}."
            )

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_tensor = torch.from_numpy(is_pad)

        image_data = torch.einsum("k h w c -> k c h w", image_data).float() / 255.0
        target_rgb = torch.from_numpy(target_rgb).permute(2, 0, 1).float() / 255.0
        roi_weight_mask = torch.from_numpy(roi_weight_mask).float()

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        box_data = np.zeros(5, dtype=np.float32)
        for cam_name in self.camera_names:
            label_path = os.path.join(
                self.dataset_dir,
                "yolo_labels",
                f"episode_{episode_id}",
                cam_name,
                f"{int(start_ts):05d}.txt",
            )
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as file_obj:
                    line = file_obj.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5 and parts[0] == "0":
                            box_data = np.array([float(x) for x in parts[:5]], dtype=np.float32)
                            break
        box_tensor = torch.from_numpy(box_data).float()

        return image_data, qpos_data, action_data, is_pad_tensor, box_tensor, target_rgb, roi_weight_mask


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{episode_idx}.hdf5")
        with h5py.File(dataset_path, "r") as root:
            qpos = root["/observations/qpos"][()]
            action = root["/action"][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = torch.clip(all_action_data.std(dim=[0, 1], keepdim=True), 1e-2, np.inf)
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(all_qpos_data.std(dim=[0, 1], keepdim=True), 1e-2, np.inf)

    return {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
        "example_qpos": qpos,
    }


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, target_camera):
    print(f"\nData from: {dataset_dir}\n")
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, target_camera)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, target_camera)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])
    return peg_pose, socket_pose


def compute_dict_mean(epoch_dicts):
    result = {key: None for key in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for key in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[key]
        result[key] = value_sum / num_items
    return result


def detach_dict(data):
    return {key: value.detach() for key, value in data.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
