import h5py
import numpy as np
import os
import glob
import argparse

def compute_qvel(qpos, timestamps):
    """Computes velocity via finite differences. Handles boundary by duplicating the first velocity."""
    dt = np.diff(timestamps)
    # Prevent division by zero if timestamps are identical
    dt = np.where(dt <= 0, 1e-3, dt) 
    
    # Compute derivative
    qvel = np.diff(qpos, axis=0) / dt[:, None]
    
    # Pad the first frame so qvel has the same length as qpos
    qvel = np.vstack([qvel[0:1], qvel])
    return qvel

def convert_episode(input_path, output_path):
    with h5py.File(input_path, 'r') as f_in:
        # 1. Read source data
        cam1 = f_in['observations/images/cam1_rgb'][:]
        cam2 = f_in['observations/images/cam2_rgb'][:]
        aria = f_in['observations/images/aria_rgb'][:]
        
        qpos_arm = f_in['observations/qpos'][:]
        gripper_state = f_in['observations/gripper_state'][:]
        
        # Ensure gripper_state is 2D: (N, 1)
        if gripper_state.ndim == 1:
            gripper_state = gripper_state[:, None]
            
        actions_arm = f_in['action'][:]
        timestamps = f_in['timestamp'][:]
        
        # 2. Process to Target Format (7 DoF)
        # Combine arm joints and gripper
        qpos_7dof = np.concatenate([qpos_arm, gripper_state], axis=1)
        
        # Pad actions with the actual gripper state (since target_g wasn't logged in actions)
        action_7dof = np.concatenate([actions_arm, gripper_state], axis=1)
        
        # Calculate velocities
        qvel_7dof = compute_qvel(qpos_7dof, timestamps)
        
        max_timesteps = qpos_7dof.shape[0]

        # 3. Write to new target format
        with h5py.File(output_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as f_out:
            f_out.attrs['sim'] = False # Kept as False since it's real data
            
            obs = f_out.create_group('observations')
            image_grp = obs.create_group('images')
            
            # Save images with chunking (matching record_sim_episodes.py)
            image_grp.create_dataset('cam1_rgb', data=cam1, dtype='uint8', chunks=(1, 480, 640, 3))
            image_grp.create_dataset('cam2_rgb', data=cam2, dtype='uint8', chunks=(1, 480, 640, 3))
            image_grp.create_dataset('aria_rgb', data=aria, dtype='uint8', chunks=(1, 480, 640, 3))
            
            # Save states and actions
            obs.create_dataset('qpos', data=qpos_7dof)
            obs.create_dataset('qvel', data=qvel_7dof)
            f_out.create_dataset('action', data=action_7dof)
            
            # Optional: keep timestamps just in case
            f_out.create_dataset('timestamp', data=timestamps)

    print(f"Converted: {os.path.basename(input_path)} -> {os.path.basename(output_path)} (Length: {max_timesteps})")

def main():
    parser = argparse.ArgumentParser(description="Convert raw real data to sim-compatible 7-DoF HDF5 format.")
    parser.add_argument('--input_dir', type=str, default='/mnt/Ego2Exo/pick_teleop_4/todo/smooth', help='Directory with raw .h5 files')
    parser.add_argument('--output_dir', type=str, default='/mnt/Ego2Exo/pick_teleop_4/todo/smooth/aloha_100', help='Directory to save .hdf5 files')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.h5')))
    
    if not input_files:
        print(f"No .h5 files found in {args.input_dir}")
        return

    print(f"Found {len(input_files)} episodes. Starting conversion...")
    
    for i, file_path in enumerate(input_files):
        # Create continuous naming like episode_0.hdf5, episode_1.hdf5
        out_name = f"episode_{i}.hdf5"
        out_path = os.path.join(args.output_dir, out_name)
        convert_episode(file_path, out_path)
        
    print(f"\nAll files successfully converted and saved to {args.output_dir}")

if __name__ == '__main__':
    main()