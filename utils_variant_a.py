import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader


def build_detail_weight_mask(rgb_image, background_weight=1.0, detail_weight=10.0):
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    detail = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    if detail.max() > 0:
        detail = detail / detail.max()
    return (background_weight + detail_weight * detail).astype(np.float32)


def build_focus_region_weight_mask(mask_image, cube_weight=3.0, gripper_weight=1.5):
    focus_weight_mask = np.zeros_like(mask_image, dtype=np.float32)
    focus_weight_mask[mask_image == 1] = cube_weight
    focus_weight_mask[(mask_image == 2) | (mask_image == 3)] = gripper_weight
    return focus_weight_mask


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        target_camera,
        roi_background_weight=1.0,
        roi_detail_weight=10.0,
        focus_masked_region=False,
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.target_camera = target_camera
        self.roi_background_weight = roi_background_weight
        self.roi_detail_weight = roi_detail_weight
        self.focus_masked_region = focus_masked_region
        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f"episode_{episode_id}.hdf5")
        with h5py.File(dataset_path, "r") as root:
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
            target_mask = None
            if self.focus_masked_region:
                mask_path = f"/observations/masks/{self.target_camera}"
                if mask_path not in root:
                    raise KeyError(
                        f"Variant A expected semantic masks at {mask_path} in {dataset_path} "
                        "when focus_masked_region is enabled."
                    )
                target_mask = root[mask_path][start_ts]
                if target_mask.shape[0] != 480 or target_mask.shape[1] != 640:
                    target_mask = cv2.resize(target_mask, (640, 480), interpolation=cv2.INTER_NEAREST)

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
        roi_weight_mask = build_detail_weight_mask(
            target_rgb,
            background_weight=self.roi_background_weight,
            detail_weight=self.roi_detail_weight,
        )
        focus_weight_mask = (
            build_focus_region_weight_mask(target_mask)
            if target_mask is not None
            else np.zeros(target_rgb.shape[:2], dtype=np.float32)
        )
        if roi_weight_mask.sum() == 0:
            raise ValueError(
                f"Variant A expected a non-empty reconstruction weight mask in {dataset_path} "
                f"for camera {self.target_camera} at step {start_ts}."
            )

        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad_tensor = torch.from_numpy(is_pad)

        image_data = torch.einsum("k h w c -> k c h w", image_data).float() / 255.0
        target_rgb = torch.from_numpy(target_rgb).permute(2, 0, 1).float() / 255.0
        roi_weight_mask = torch.from_numpy(roi_weight_mask).float()
        focus_weight_mask = torch.from_numpy(focus_weight_mask).float()

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

        return image_data, qpos_data, action_data, is_pad_tensor, box_tensor, target_rgb, roi_weight_mask, focus_weight_mask


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


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    target_camera,
    roi_background_weight=1.0,
    roi_detail_weight=10.0,
    focus_masked_region=False,
):
    print(f"\nData from: {dataset_dir}\n")
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[: int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes) :]

    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    train_dataset = EpisodicDataset(
        train_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        target_camera,
        roi_background_weight=roi_background_weight,
        roi_detail_weight=roi_detail_weight,
        focus_masked_region=focus_masked_region,
    )
    val_dataset = EpisodicDataset(
        val_indices,
        dataset_dir,
        camera_names,
        norm_stats,
        target_camera,
        roi_background_weight=roi_background_weight,
        roi_detail_weight=roi_detail_weight,
        focus_masked_region=focus_masked_region,
    )

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
