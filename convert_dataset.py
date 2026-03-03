import os
import h5py
import numpy as np
import torch
from tqdm import tqdm

def get_norm_stats(dataset_dir, num_episodes):
    """Calculates normalization stats from the original dataset format."""
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)

    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    return {
        "action_mean": action_mean.numpy().squeeze(), 
        "action_std": action_std.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(), 
        "qpos_std": qpos_std.numpy().squeeze()
    }

def convert_to_prenormalized_format(input_dir, output_filepath, num_episodes, camera_names):
    print(f"Calculating normalization stats from {input_dir}...")
    stats = get_norm_stats(input_dir, num_episodes)
    
    q_mean, q_std = stats['qpos_mean'], stats['qpos_std']
    a_mean, a_std = stats['action_mean'], stats['action_std']

    # --- DEFINE ARM/GRIPPER SPLIT ---
    # Adjust these indices based on your state_dim (7 for single arm, 14 for dual arm)
    state_dim = q_mean.shape[0]
    if state_dim == 14: # Standard ALOHA dual-arm
        arm_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
        gripper_indices = [6, 13]
    elif state_dim == 7: # Single arm
        arm_indices = [0, 1, 2, 3, 4, 5]
        gripper_indices = [6]
    else:
        # Fallback: assume last joint is the gripper
        arm_indices = list(range(state_dim - 1))
        gripper_indices = [state_dim - 1]

    print(f"Creating pre-normalized dataset at {output_filepath}...")
    with h5py.File(output_filepath, 'w') as out_f:
        # 1. Write the 'normalization' group for evaluation/inference
        norm_group = out_f.create_group('normalization')
        norm_group.create_dataset('action_mean', data=a_mean[arm_indices])
        norm_group.create_dataset('action_std', data=a_std[arm_indices])
        norm_group.create_dataset('qpos_mean', data=q_mean[arm_indices])
        norm_group.create_dataset('qpos_std', data=q_std[arm_indices])
        norm_group.create_dataset('gripper_state_mean', data=q_mean[gripper_indices])
        norm_group.create_dataset('gripper_state_std', data=q_std[gripper_indices])

        # 2. Process and restructure each episode
        for i in tqdm(range(num_episodes), desc="Converting episodes"):
            in_filepath = os.path.join(input_dir, f'episode_{i}.hdf5')
            with h5py.File(in_filepath, 'r') as in_f:
                
                # Read raw data
                raw_qpos = in_f['/observations/qpos'][()]
                raw_action = in_f['/action'][()]
                
                # PRE-NORMALIZE the data
                norm_qpos = (raw_qpos - q_mean) / q_std
                norm_action = (raw_action - a_mean) / a_std

                # Split out the arm and gripper data
                qpos_arm = norm_qpos[:, arm_indices]
                action_arm = norm_action[:, arm_indices]
                
                # NOTE: utils_joint_1_step.py reads the gripper action directly 
                # from observations/gripper_state. Therefore, we store the 
                # normalized true action for the gripper in gripper_state to 
                # ensure the policy learns the right target.
                gripper_state_for_action = norm_action[:, gripper_indices]

                # Create structural hierarchy: demo_0, demo_1...
                demo_group = out_f.create_group(f'demo_{i}')
                demo_group.create_dataset('action', data=action_arm)
                
                obs_group = demo_group.create_group('observations')
                obs_group.create_dataset('qpos', data=qpos_arm)
                obs_group.create_dataset('gripper_state', data=gripper_state_for_action)
                
                # Transfer images directly (image normalization /255.0 happens in dataloader)
                img_group = obs_group.create_group('images')
                for cam_name in camera_names:
                    raw_img = in_f[f'/observations/images/{cam_name}'][()]
                    img_group.create_dataset(cam_name, data=raw_img)

    print("Conversion complete!")

if __name__ == "__main__":
    # --- CONFIGURE THESE PATHS ---
    INPUT_DATA_DIR = "./dataset/sim_transfer_cube_scripted" 
    OUTPUT_FILEPATH = "./dataset/prenormalized_dataset.hdf5"
    
    NUM_EPISODES = 50 # Set to match your dataset size
    CAMERA_NAMES = ['top'] # Update with your actual camera names
    
    convert_to_prenormalized_format(INPUT_DATA_DIR, OUTPUT_FILEPATH, NUM_EPISODES, CAMERA_NAMES)