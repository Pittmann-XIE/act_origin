import torch
import numpy as np
import os
import time
import argparse
import sys
import cv2
import pickle
import threading
from scipy.spatial.transform import Rotation as R

# --- Project Imports ---
# Make sure this path points to where policy.py is located
from policy import ACTPolicy, CNNMLPPolicy

# --- Driver Imports ---
try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:
    print("⚠️ WARNING: RTDE Drivers not found. Sim-Only.")
    RTDEControlInterface = None
    RTDEReceiveInterface = None

sys.path.append('/home/pengtao/ws_ros2humble-main_lab/ur3_vla_teleop/')

try:
    from drivers_new.wsg_driver import WSGGripperDriver
except ImportError:
    print("⚠️ WARNING: WSG Driver not found.")
    WSGGripperDriver = None

import pybullet as p
import pybullet_data
import mplib
import pyrealsense2 as rs
from multiprocessing import shared_memory, resource_tracker
import struct

# -----------------------------------------------------------------------------
# USER CONFIGURATION & CONSTANTS
# -----------------------------------------------------------------------------
MANUAL_SPEED_RAD_S = 0.1
MANUAL_ACCEL_RAD_S2 = 0.05
SERVO_DT = 0.1
SERVO_LOOKAHEAD = 0.1
SERVO_GAIN = 300
ACTIVE_HORIZON = 100 

ROBOT_IP = "10.0.0.1" 
GRIPPER_PORT = "/dev/ttyACM0" 

CAM_SERIALS = ['104122061227', '105422061000']
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

INIT_JOINTS_DEG = [47.22, -94.58, -125.38, -0.0, -271.32, 4.32]
SIM_ROBOT_WORLD_ROT_Z = np.pi 
WALL_POS_WORLD = [-0.14, -0.14, 0.5]  
WALL_ROT_Z_DEG = 45
WALL_SIZE = [0.02, 1.0, 1.0] 
TABLE_SIZE = [1.0, 1.0, 0.1]
TABLE_POS_WORLD = [0, 0, -TABLE_SIZE[2]/2-0.01]
TABLE_ROT_Z_DEG = 0


# -----------------------------------------------------------------------------
# HARDWARE & SENSOR CLASSES (Preserved from inference_joint_1.py)
# -----------------------------------------------------------------------------
def get_object_transforms(pos_world_arr, rot_z_deg):
    pos_world = np.array(pos_world_arr)
    r_obj_local = R.from_euler('z', rot_z_deg, degrees=True)
    r_world_robot = R.from_euler('z', -SIM_ROBOT_WORLD_ROT_Z, degrees=False)
    pos_robot = r_world_robot.apply(pos_world)
    quat_robot = (r_world_robot * r_obj_local).as_quat()
    quat_world = r_obj_local.as_quat()
    return pos_robot, quat_robot, pos_world, quat_world

class ThreadedSensor:
    def __init__(self, sensor_class, *args, **kwargs):
        self.sensor_name = sensor_class.__name__
        self.sensor = sensor_class(*args, **kwargs)
        self.lock = threading.Lock()
        self.latest_data = None
        self.timestamp = 0
        self.running = False
        self.thread = None
        self.measured_fps = 0.0
        self._frame_counter = 0
        self._last_fps_time = time.time()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            try:
                data = self.sensor.get_frame()
                if data is not None:
                    with self.lock:
                        self.latest_data = data
                        self.timestamp = time.time()
                        self._frame_counter += 1
                        now = time.time()
                        elapsed = now - self._last_fps_time
                        if elapsed >= 1.0:
                            self.measured_fps = self._frame_counter / elapsed
                            self._frame_counter = 0
                            self._last_fps_time = now
            except Exception as e:
                time.sleep(1.0)
            time.sleep(0.001) 

    def get_latest(self):
        with self.lock:
            if self.latest_data is None: return None, 0, 0.0
            return self.latest_data.copy(), self.timestamp, self.measured_fps

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        try: self.sensor.stop()
        except: pass

class SharedMemoryReceiver:
    def __init__(self, shm_name, shape=(480, 640, 3)):
        self.shm_name = shm_name
        self.shape = shape
        self.size = np.prod(shape)
        self.connected = False
        self.shm = None
        self._connect()

    def _connect(self):
        try:
            self.shm = shared_memory.SharedMemory(create=False, name=self.shm_name)
            resource_tracker.unregister(self.shm._name, 'shared_memory')
            self.connected = True
            print(f"[{self.shm_name}] Connected to Shared Memory.")
        except FileNotFoundError:
            print(f"[{self.shm_name}] Waiting for sender...")

    def get_frame(self):
        if not self.connected:
            self._connect()
            if not self.connected: return None
        try:
            version_start = struct.unpack_from('Q', self.shm.buf, 0)[0]
            img_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf, offset=8)
            img_copy = img_buffer.copy()
            version_end = struct.unpack_from('Q', self.shm.buf, 0)[0]
            if version_start != version_end or version_start == 0: return None
            return img_copy
        except Exception:
            self.connected = False
            return None

    def stop(self):
        if self.shm: self.shm.close()

class RealSenseCamera:
    def __init__(self, serial_number, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS):
        self.serial = serial_number
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial_number)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipeline.start(config)

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame: return None
        return np.asanyarray(color_frame.get_data())

    def stop(self):
        self.pipeline.stop()

class RobotSystem:
    def __init__(self, robot_ip, gripper_port):
        p.connect(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(1.2, 90, -30, [0, 0, 0])
        p.loadURDF("plane.urdf")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.urdf_path = os.path.join(current_dir, "ur3_gripper.urdf")
        self.srdf_path = os.path.join(current_dir, "ur3_gripper.srdf")
        
        sim_orn = p.getQuaternionFromEuler([0, 0, SIM_ROBOT_WORLD_ROT_Z])
        self.sim_robot_id = p.loadURDF(self.urdf_path, [0,0,0], baseOrientation=sim_orn, useFixedBase=1)
        self.ghost_robot_id = p.loadURDF(self.urdf_path, [0,0,0], baseOrientation=sim_orn, useFixedBase=1)
        self._make_ghost_transparent()
        self._setup_obstacles()
        
        self.planner = mplib.Planner(urdf=self.urdf_path, srdf=self.srdf_path, move_group="tool0")
        
        self.rtde_c = None
        self.rtde_r = None
        if RTDEControlInterface:
            try:
                self.rtde_c = RTDEControlInterface(robot_ip)
                self.rtde_r = RTDEReceiveInterface(robot_ip)
            except Exception as e: print(f"❌ Robot Connection failed: {e}")
        
        self.gripper = None
        if WSGGripperDriver:
            try:
                self.gripper = WSGGripperDriver(gripper_port)
                self.gripper.move(1.0)
            except Exception as e: print(f"❌ Gripper Connection Failed: {e}")

        joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        name_to_idx = {p.getJointInfo(self.sim_robot_id, i)[1].decode('utf-8'): i for i in range(p.getNumJoints(self.sim_robot_id))}
        self.joint_indices = [name_to_idx[n] for n in joint_names]
        self.gripper_last_cmd = 1.0
        self.go_home_sim_only()

    def go_home_sim_only(self):
        home_rad = np.deg2rad(INIT_JOINTS_DEG)
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.sim_robot_id, idx, home_rad[i])
            p.resetJointState(self.ghost_robot_id, idx, home_rad[i])
        p.stepSimulation()
        
    def go_home(self):
        home_rad = np.deg2rad(INIT_JOINTS_DEG)
        if self.gripper:
            try:
                self.gripper.move(1.0) 
                self.gripper_last_cmd = 1.0 
                time.sleep(0.5) 
            except Exception as e: pass

        if self.rtde_c:
            try: self.rtde_c.moveJ(home_rad, MANUAL_SPEED_RAD_S, MANUAL_ACCEL_RAD_S2)
            except Exception as e: pass
    
        for i, idx in enumerate(self.joint_indices): 
            p.resetJointState(self.sim_robot_id, idx, home_rad[i])
            p.resetJointState(self.ghost_robot_id, idx, home_rad[i])
        p.stepSimulation()

    def _make_ghost_transparent(self):
        for j in range(p.getNumJoints(self.ghost_robot_id)):
            p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 1, 0.4])
        p.setCollisionFilterGroupMask(self.ghost_robot_id, -1, collisionFilterGroup=0, collisionFilterMask=0)
        for j in range(p.getNumJoints(self.ghost_robot_id)):
            p.setCollisionFilterGroupMask(self.ghost_robot_id, j, collisionFilterGroup=0, collisionFilterMask=0)

    def _setup_obstacles(self):
        wall_pos_r, wall_quat_r, wall_pos_w, wall_quat_w = get_object_transforms(WALL_POS_WORLD, WALL_ROT_Z_DEG)
        table_pos_r, table_quat_r, table_pos_w, table_quat_w = get_object_transforms(TABLE_POS_WORLD, TABLE_ROT_Z_DEG)
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE]), 
                          basePosition=wall_pos_w, baseOrientation=wall_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in WALL_SIZE], rgbaColor=[0.8,0.2,0.2,0.5]))
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE]), 
                          basePosition=table_pos_w, baseOrientation=table_quat_w,
                          baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in TABLE_SIZE], rgbaColor=[0.5,0.3,0.1,1.0]))

    def get_qpos_real(self):
        if self.rtde_r:
            q = self.rtde_r.getActualQ()
        else:
            q = [p.getJointState(self.sim_robot_id, i)[0] for i in self.joint_indices]
        if self.gripper:
            try: g_state = self.gripper.get_pos() 
            except: raise RuntimeError('Get not get gripper distance')
        else: g_state = 110.0
        return np.array(q + [g_state])
    
    def check_collision(self, qpos_target):
        res = self.planner.check_for_self_collision(qpos_target[:6])
        if res: return True
        res = self.planner.check_for_env_collision(qpos_target[:6])
        return res

    def update_ghost(self, action):
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.ghost_robot_id, idx, action[i])

    def execute_action(self, action, mode='manual'):
        target_joints = action[:6]
        target_gripper = action[6]

        if self.check_collision(action):
            print("⚠️ COLLISION DETECTED! Stopping.")
            return False

        self.update_ghost(action)
        if mode == 'manual':
            for j in range(p.getNumJoints(self.ghost_robot_id)):
                p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 0, 0.6])
            print(f"   >>> LOOK AT WINDOW: Press 'g' to Execute, 'q/c' to Skip/Quit.")
            user_approved = False
            while True:
                key = cv2.waitKey(10) & 0xFF
                if key == ord('g'):
                    user_approved = True
                    break
                elif key == ord('q') or key == ord('c'):
                    break
            for j in range(p.getNumJoints(self.ghost_robot_id)):
                p.changeVisualShape(self.ghost_robot_id, j, rgbaColor=[0, 1, 1, 0.4])
            if not user_approved: return False

        if self.rtde_c:
            try:
                if mode == 'auto': self.rtde_c.servoJ(target_joints, 0.0, 0.0, SERVO_DT, SERVO_LOOKAHEAD, SERVO_GAIN)
                else: self.rtde_c.moveJ(target_joints, MANUAL_SPEED_RAD_S, MANUAL_ACCEL_RAD_S2)
            except Exception as e: return False

        if self.gripper:
            try:
                TO_CLOSE_THRESH, TO_OPEN_THRESH = 45, 45  
                current_width = float(target_gripper)
                is_currently_open = (self.gripper_last_cmd > 0.5)

                if is_currently_open:
                    cmd_val = 0.0 if current_width < TO_CLOSE_THRESH else 1.0
                else:
                    cmd_val = 1.0 if current_width > TO_OPEN_THRESH else 0.0

                if abs(cmd_val - self.gripper_last_cmd) > 0.1:
                    t = threading.Thread(target=self.gripper.move, args=(cmd_val,), daemon=True)
                    t.start()
                    self.gripper_last_cmd = cmd_val
            except Exception as e: print(f"Gripper Error: {e}")
            
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.sim_robot_id, idx, target_joints[i])
        p.stepSimulation()
        return True

# -----------------------------------------------------------------------------
# UTILS 
# -----------------------------------------------------------------------------
# [ACT FIX] Replicating make_policy from imitate_episodes_7dof.py
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

# -----------------------------------------------------------------------------
# MAIN LOOP
# -----------------------------------------------------------------------------

def main(args):
    print(f'hello')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # [ACT FIX] Ensure camera names match those used in evaluation script.
    camera_names = args.camera_names.split(',')

    # [ACT FIX] Reconstruct dynamic configuration required by imitate_episodes_7dof.py
    policy_config = {
        'lr': 1e-5,
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'lr_backbone': 1e-5,
        'backbone': 'resnet18',
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': camera_names,
        'ckpt_dir ': 'ckpt_100',
        'policy_class': 'ACT',
        'task_name': 'sim_pick_place',
        'seed': 0,
        'num-epochs': 2000
    }
    print(f'hello')
    

    checkpoint_dir = os.path.dirname(args.checkpoint)
    stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
    if not os.path.exists(stats_path):
        stats_path = os.path.join(os.path.dirname(checkpoint_dir), 'dataset_stats.pkl')

    try:
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        
        QPOS_MEAN = stats['qpos_mean']
        QPOS_STD  = stats['qpos_std']
        ACTION_MEAN = stats['action_mean']
        ACTION_STD  = stats['action_std']
        print(f"✅ Loaded stats from: {stats_path}")
    except Exception as e:
        print(f"❌ Failed to load dataset_stats.pkl: {e}")
        sys.exit(1)

    def pre_process(s_qpos):
        return (s_qpos - QPOS_MEAN) / QPOS_STD

    def post_process(a):
        return a * ACTION_STD + ACTION_MEAN

    # Instantiate and load policy
    policy = make_policy(args.policy_class, policy_config)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # FIX: Change 'model_state_dict' to 'policy_state_dict' to match imitate_episodes_7dof.py
    state_dict = payload['policy_state_dict'] if isinstance(payload, dict) and 'policy_state_dict' in payload else payload
    
    policy.load_state_dict(state_dict)
    policy.to(device)
    policy.eval()
    
    # --- PHASE 1: Initialize Cameras ---
    active_sensors = {}
    try:
        if 'aria_rgb' in camera_names:
            active_sensors['aria_rgb'] = ThreadedSensor(SharedMemoryReceiver, shm_name="aria_stream_v1")
            active_sensors['aria_rgb'].start()
        if 'cam1_rgb' in camera_names:
            active_sensors['cam1_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[0])
            active_sensors['cam1_rgb'].start()
        if 'cam2_rgb' in camera_names:
            active_sensors['cam2_rgb'] = ThreadedSensor(RealSenseCamera, serial_number=CAM_SERIALS[1])
            active_sensors['cam2_rgb'].start()
    except Exception as e:
        print(f"❌ Camera Init Failed: {e}")
        return

    while True:
        all_ready = True
        for name, sensor in active_sensors.items():
            _, t_stamp, _ = sensor.get_latest()
            if t_stamp == 0:
                all_ready = False
                break
        if all_ready: break
        time.sleep(0.5)

    # --- PHASE 2 & 3: Connect & Home Robot ---
    input("Press [Enter] ONLY after you have connected the robot cable...")
    env = RobotSystem(args.robot_ip, args.gripper_port)
    env.go_home()

    # --- INFERENCE SETUP ---
    num_queries = policy_config['num_queries']
    query_freq = 1 if args.action_chunking else min(args.steps_per_inference, num_queries)
    
    # [ACT FIX] Temporal buffer initialized to ZEROS, not NaNs.
    state_dim = 7 # 6 joints + 1 gripper
    all_time_actions = torch.zeros(
        [args.max_timesteps, args.max_timesteps + num_queries, state_dim]
    ).to(device)
    
    all_actions = None
    attn_weights = None 
    t = 0
    
    try:
        while t < args.max_timesteps:
            loop_start = time.time()
            
            # 1. State Capture
            qpos_raw = env.get_qpos_real()
            qpos_norm = pre_process(qpos_raw)
            qpos_tensor = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)
            
           # 2. Image Capture & Processing
            tensors_to_stack = []
            latest_bgr_images = {} # ADDED: Store BGR images for OpenCV
            latest_fps = {}        # ADDED: Store FPS for display

            for cam_name in camera_names:
                img_raw, ts, fps = active_sensors[cam_name].get_latest()
                if img_raw is None: continue
                
                # ADDED: Save data for visualization
                latest_bgr_images[cam_name] = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                latest_fps[cam_name] = fps
                
                # [ACT FIX] Removed ImageNet normalization here! 
                # It is already handled inside policy.py's __call__ function
                img_t = torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0
                tensors_to_stack.append(img_t.to(device))
            
            img_all = torch.stack(tensors_to_stack, dim=0).unsqueeze(0)

            # 3. Model Forward Pass
            if args.action_chunking or (t % query_freq == 0):
                with torch.inference_mode():
                    # [ACT FIX] Safely unpack outputs based on Policy Class
                    if args.policy_class == "ACT":
                        all_actions, attn_weights = policy(qpos_tensor, img_all)
                    elif args.policy_class == "CNNMLP":
                        all_actions = policy(qpos_tensor, img_all)
                        attn_weights = None

            # --- ADDED: VISUALIZATION & ATTENTION OVERLAY ---
            if args.visualize and attn_weights is not None:
                # Extract attention for the current prediction query (index 0)
                # Ignore the first 2 tokens (latent, proprio)
                attn = attn_weights[0, 0, 2:].detach().cpu().numpy()
                
                # ResNet18 with 480x640 input -> 32x downsampling = 15x20
                h, w = 15, 20
                num_cams = len(camera_names)
                
                # Reshape to a single wide 2D feature map
                attn_2d_full = attn.reshape(h, w * num_cams)
                
                for cam_id, cam_name in enumerate(camera_names):
                    if cam_name in latest_bgr_images:
                        img_bgr = latest_bgr_images[cam_name]
                        
                        # Add FPS text overlay
                        fps_val = latest_fps.get(cam_name, 0.0)
                        color = (0, 255, 0) if fps_val > 20 else (0, 0, 255)
                        cv2.putText(img_bgr, f"FPS: {fps_val:.1f}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
                        
                        # Slice out the specific camera's attention map
                        start_w = cam_id * w
                        end_w = (cam_id + 1) * w
                        cam_attn = attn_2d_full[:, start_w:end_w]
                        
                        # Normalize to 0-1 for heatmap rendering
                        cam_attn = (cam_attn - cam_attn.min()) / (cam_attn.max() - cam_attn.min() + 1e-8)
                        
                        # Resize to match camera dimensions (640x480)
                        cam_attn_resized = cv2.resize(cam_attn, (CAM_WIDTH, CAM_HEIGHT))
                        heatmap = cv2.applyColorMap(np.uint8(255 * cam_attn_resized), cv2.COLORMAP_JET)
                        
                        # Blend the original image and heatmap
                        overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
                        cv2.imshow(f"Input & Attention: {cam_name}", overlay)
                cv2.waitKey(1)

            # 4. Action Chunking / Temporal Ensembling
            if args.action_chunking:
                all_time_actions[[t], t : t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                
                # [ACT FIX] Filter using non-zero logic instead of ~torch.isnan
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_valid = actions_for_curr_step[actions_populated]
                
                if len(actions_valid) > ACTIVE_HORIZON:
                    actions_valid = actions_valid[-ACTIVE_HORIZON:]
                
                k = 0.01 
                indices = np.arange(len(actions_valid))[::-1]
                exp_weights = np.exp(-k * indices)
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                
                raw_action = (actions_valid * exp_weights).sum(dim=0).cpu().numpy()
            else:
                raw_action = all_actions[:, t % query_freq].squeeze(0).cpu().numpy()
                
            # 5. Execute Action
            action = post_process(raw_action)
            success = env.execute_action(action, mode=args.mode)
            
            if not success: break
            t += 1
            if args.mode == 'auto':
                elapsed = time.time() - loop_start
                time.sleep(max(0, SERVO_DT - elapsed))

    except KeyboardInterrupt:
        print("Stopping...")
        if env.rtde_c: env.rtde_c.stopScript()
    finally:
        for name, sensor in active_sensors.items():
            sensor.stop()
        p.disconnect()
        cv2.destroyAllWindows() 
        if env.gripper: env.gripper.close_connection()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General execution
    parser.add_argument('--checkpoint', type=str, default='/home/pengtao/ws_ros2humble-main_lab/act_new/ckpt_100/policy_epoch_2800_seed_0.ckpt', help='Path to policy.ckpt')
    parser.add_argument('--robot_ip', type=str, default=ROBOT_IP)
    parser.add_argument('--gripper_port', type=str, default=GRIPPER_PORT)
    parser.add_argument('--mode', type=str, default='auto', choices=['manual', 'auto'])
    parser.add_argument('--max_timesteps', type=int, default=10000)
    
    # [ACT FIX] Policy configuration parameters needed to match imitate_episodes_7dof.py
    parser.add_argument('--policy_class', type=str, default='ACT', choices=['ACT', 'CNNMLP'])
    parser.add_argument('--camera_names', type=str, default='cam1_rgb,cam2_rgb,aria_rgb', help='Comma separated list of cameras')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size / num_queries')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Transformer hidden dim')
    parser.add_argument('--dim_feedforward', type=int, default=3200)
    parser.add_argument('--kl_weight', type=int, default=10)
    
    # Chunking
    parser.add_argument('--action_chunking', action='store_true', default=True)
    parser.add_argument('--steps_per_inference', type=int, default=1)
    parser.add_argument('--visualize', action='store_true', default=True, help="Visualize camera streams with attention overlays")
    
    args = parser.parse_args()
    main(args)