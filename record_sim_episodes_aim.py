import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import cv2  # <--- ADD THIS

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython
e = IPython.embed

import ee_sim_env  
import sim_env     

def get_pixel_coords(physics, pos_3d, camera_name='top', width=640, height=480):
    """Projects a 3D world position to 2D pixel coordinates for a given camera."""
    cam_pos = physics.named.data.cam_xpos[camera_name]
    cam_mat = physics.named.data.cam_xmat[camera_name].reshape(3, 3)
    
    pos_cam = cam_mat.T @ (pos_3d - cam_pos)
    x_c, y_c, z_c = pos_cam
    
    if z_c > 0:
        return [-1, -1]  # Point is behind the camera
        
    fovy = physics.named.model.cam_fovy[camera_name]
    f = 0.5 * height / np.tan(fovy * np.pi / 360.0)
    
    pixel_x = width / 2.0 + f * x_c / (-z_c)
    pixel_y = height / 2.0 - f * y_c / (-z_c) 
    
    return [int(pixel_x), int(pixel_y)]

# # --- ADDED: DRAWING FUNCTION ---
# def add_aimbot_line_to_img(img_np, physics, left_gripper_open, right_gripper_open, cam_name):
#     """Draws the AimBot visual cues on the image using the current physics state."""
#     for side, is_open in [('left', left_gripper_open), ('right', right_gripper_open)]:
#         link_name = f'vx300s_{side}/gripper_link'
#         try:
#             ee_pos = physics.named.data.xpos[link_name]
#             ee_mat = physics.named.data.xmat[link_name].reshape(3, 3)
#         except KeyError:
#             continue

#         # 1. Look at the XML: The fingers are at pos="0.0687 0 0" relative to gripper_link.
#         # This means the gripper points ALONG THE LOCAL +X AXIS.
        
#         # 2. Start the line at the fingertips (approx 7cm forward on X)
#         start_vec = ee_mat @ np.array([0.07, 0, 0])
#         pt_3d_start = ee_pos + start_vec
        
#         # 3. End the line 25cm further out from the fingertips
#         end_vec = ee_mat @ np.array([0.32, 0, 0]) 
#         pt_3d_end = ee_pos + end_vec

#         pt_start = get_pixel_coords(physics, pt_3d_start, cam_name, img_np.shape[1], img_np.shape[0])
#         pt_end = get_pixel_coords(physics, pt_3d_end, cam_name, img_np.shape[1], img_np.shape[0])

#         # Ensure both points are in front of the camera before drawing
#         if pt_start != [-1, -1] and pt_end != [-1, -1]:
#             # RGB Color coding from AimBot paper [cite: 107, 118]
#             if is_open:
#                 line_color = (0, 255, 0) # Green 
#                 dot_color = (255, 0, 0)  # Red 
#             else:
#                 line_color = (128, 0, 128) # Purple 
#                 dot_color = (0, 0, 255)    # Blue 

#             cv2.line(img_np, tuple(pt_start), tuple(pt_end), line_color, 2)
#             cv2.circle(img_np, tuple(pt_start), 4, dot_color, -1)

#     return img_np


def add_aimbot_line_to_img(img_np, physics, left_gripper_open, right_gripper_open, cam_name):
    """Draws the AimBot visual cues on the image using the current physics state."""
    for side, is_open in [('left', left_gripper_open), ('right', right_gripper_open)]:
        link_name = f'vx300s_{side}/gripper_link'
        try:
            ee_pos = physics.named.data.xpos[link_name]
            ee_mat = physics.named.data.xmat[link_name].reshape(3, 3)
        except KeyError:
            continue

        # 1. Start the line at the fingertips (Your calibrated origin!)
        start_vec = ee_mat @ np.array([0.070, 0.000, 0.000])
        pt_3d_start = ee_pos + start_vec
        
        # 2. Calculate the Ray Direction
        ray_dir = ee_mat @ np.array([1.0, 0.0, 0.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir) # Ensure it's a unit vector
        
        # 3. Dynamic Depth Stopping (Fixes the Parallax!)
        max_dist = 0.25
        
        # If the ray is pointing downwards towards the table (Z < 0)
        if ray_dir[2] < -1e-4:
            # Calculate exactly how far it can go before hitting the table (Z=0.0)
            # Math: t = (Target_Z - Current_Z) / Direction_Z
            dist_to_table = (0.0 - pt_3d_start[2]) / ray_dir[2]
            
            # Use whichever is shorter: 25cm, or the distance to the table
            actual_dist = min(max_dist, max(0.0, dist_to_table))
        else:
            # If pointing up or level, just use the max distance
            actual_dist = max_dist
            
        pt_3d_end = pt_3d_start + ray_dir * actual_dist

        pt_start = get_pixel_coords(physics, pt_3d_start, cam_name, img_np.shape[1], img_np.shape[0])
        pt_end = get_pixel_coords(physics, pt_3d_end, cam_name, img_np.shape[1], img_np.shape[0])

        # Ensure both points are in front of the camera before drawing
        if pt_start != [-1, -1] and pt_end != [-1, -1]:
            if is_open:
                line_color = (0, 255, 0) # Green 
                dot_color = (255, 0, 0)  # Red 
            else:
                line_color = (128, 0, 128) # Purple 
                dot_color = (0, 0, 255)    # Blue 

            # --- SPLIT LOGIC BASED ON CAMERA TYPE ---
            if 'wrist' in cam_name:
                # LOCAL VIEW: Draw Scope Reticle (Crosshair)
                # Ensure we only draw the reticle for the corresponding arm's camera
                if side in cam_name: 
                    reticle_size = 20 # Static size for now
                    x, y = pt_end
                    
                    # Draw horizontal and vertical lines for the crosshair at the stopping point
                    cv2.line(img_np, (x - reticle_size, y), (x + reticle_size, y), line_color, 2)
                    cv2.line(img_np, (x, y - reticle_size), (x, y + reticle_size), line_color, 2)
                    
                    # Draw center dot
                    cv2.circle(img_np, (x, y), 4, dot_color, -1)
            else:
                # GLOBAL VIEW: Draw Shooting Line
                cv2.line(img_np, tuple(pt_start), tuple(pt_end), line_color, 2)
                cv2.circle(img_np, tuple(pt_start), 4, dot_color, -1)

    return img_np
# -------------------------------
def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError

    success = []
    saved_count = 0
    attempt_idx = 0

    while saved_count < num_episodes:
        print(f'\n--- Attempt {attempt_idx} | Successfully Saved: {saved_count}/{num_episodes} ---')
        print('Rollout out EE space scripted policy')
        
        # --- COLOR SAMPLING BLOCK ---
        colors = [
            [1.0, 0.0, 0.0, 1.0],    # Red
            # [1.0, 0.41, 0.71, 1.0],  # Pink 
            # [1.0, 1.0, 0.0, 1.0],    # Yellow
        ]
        sampled_color = colors[np.random.choice(len(colors))]
        
        ee_sim_env.BOX_COLOR[0] = sampled_color
        sim_env.BOX_COLOR[0] = sampled_color
        # --------------------------------
        
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
            
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"Attempt {attempt_idx} EE Rollout: Successful, {episode_return=}")
        else:
            print(f"Attempt {attempt_idx} EE Rollout: Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() 

        del env
        del episode
        del policy

        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info 
        ts = env.reset()

        # track_targets = [
        #     ("red_box", "geom"),
        #     ("vx300s_left/left_finger_link", "body"),
        #     ("vx300s_left/right_finger_link", "body"),
        #     ("vx300s_right/left_finger_link", "body"),
        #     ("vx300s_right/right_finger_link", "body")
        # ]
        # FIX: Dynamically assign target name based on the active task
        target_obj_name = "red_peg" if 'sim_insertion' in task_name else "red_box"

        track_targets = [
            (target_obj_name, "geom"), # <-- Use the dynamic variable here
            ("vx300s_left/left_finger_link", "body"),
            ("vx300s_left/right_finger_link", "body"),
            ("vx300s_right/left_finger_link", "body"),
            ("vx300s_right/right_finger_link", "body")
        ]

        def extract_2d_points(physics):
            step_pixels = []
            for name, obj_type in track_targets:
                if obj_type == "geom":
                    pos_3d = physics.named.data.geom_xpos[name]
                else:
                    pos_3d = physics.named.data.xpos[name]
                coords = get_pixel_coords(physics, pos_3d, "top")
                step_pixels.append(coords)
            return step_pixels

        episode_replay = [ts]
        attention_trajectory = [extract_2d_points(env.physics)] 
        
        # --- GENERATE INITIAL AIMBOT IMAGES ---
        # USE THE COMMANDED ACTION INSTEAD OF PHYSICAL QPOS
        init_action = joint_traj[0]
        aimbot_imgs_step0 = {cam: add_aimbot_line_to_img(
            ts.observation['images'][cam].copy(), 
            env.physics, 
            init_action[6] > 0.5,    # Left commanded state
            init_action[13] > 0.5,   # Right commanded state
            cam) for cam in camera_names}
        episode_aimbot_images = [aimbot_imgs_step0]
        # --------------------------------------
        
        if onscreen_render:
            ax = plt.subplot()
            # Render the aimbot image to see the lines during collection!
            plt_img = ax.imshow(aimbot_imgs_step0[render_cam_name])
            plt.ion()
            
        for t in range(len(joint_traj)): 
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            attention_trajectory.append(extract_2d_points(env.physics))
            
            # --- GENERATE STEP AIMBOT IMAGES ---
            # USE THE COMMANDED ACTION INSTEAD OF PHYSICAL QPOS
            aimbot_imgs = {cam: add_aimbot_line_to_img(
                ts.observation['images'][cam].copy(), 
                env.physics, 
                action[6] > 0.5,     # Left commanded state
                action[13] > 0.5,    # Right commanded state
                cam) for cam in camera_names}
            episode_aimbot_images.append(aimbot_imgs)
            # -----------------------------------
            
            if onscreen_render:
                plt_img.set_data(aimbot_imgs[render_cam_name])
                plt.pause(0.02)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"Attempt {attempt_idx} Replay: Successful, {episode_return=}. Saving as episode_{saved_count}.hdf5")
            
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
                '/observations/attention_2d': [], 
                '/reward': []
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []
                # ADD AIMBOT DICT KEYS
                data_dict[f'/observations/images_aimbot/{cam_name}'] = []

            # truncate here to be consistent
            joint_traj = joint_traj[:-1]
            episode_replay = episode_replay[:-1]
            attention_trajectory = attention_trajectory[:-1]
            episode_aimbot_images = episode_aimbot_images[:-1]

            max_timesteps = len(joint_traj)
            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode_replay.pop(0)
                attn_points = attention_trajectory.pop(0)
                aim_imgs = episode_aimbot_images.pop(0)
                
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                data_dict['/observations/attention_2d'].append(attn_points)
                data_dict['/reward'].append(ts.reward)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
                    # APPEND AIMBOT IMAGES
                    data_dict[f'/observations/images_aimbot/{cam_name}'].append(aim_imgs[cam_name])

            t0 = time.time()
            dataset_path = os.path.join(dataset_dir, f'episode_{saved_count}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                
                # --- CREATE RAW AND AIMBOT IMAGE GROUPS ---
                image = obs.create_group('images')
                image_aimbot = obs.create_group('images_aimbot')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
                    _ = image_aimbot.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3), )
                # ------------------------------------------
                
                qpos = obs.create_dataset('qpos', (max_timesteps, 14))
                qvel = obs.create_dataset('qvel', (max_timesteps, 14))
                action = root.create_dataset('action', (max_timesteps, 14))
                attn = obs.create_dataset('attention_2d', (max_timesteps, 5, 2), dtype='int32')
                reward_dataset = root.create_dataset('reward', (max_timesteps,), dtype='float32')

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
            
            saved_count += 1
            
        else:
            success.append(0)
            print(f"Attempt {attempt_idx} Replay: Failed. Skipping save and retrying...\n")

        attempt_idx += 1

    print(f'Saved {saved_count} episodes to {dataset_dir}')
    print(f'Total Attempts: {attempt_idx}. Overall Success Rate: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))