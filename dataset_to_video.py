## track heat map
import h5py
import cv2
import argparse
import os
import numpy as np

def get_attention_weights(reward):
    """Returns cognitive attention weights based on task progress."""
    if reward == 0:
        return [1.0, 0.2, 0.2, 1.0, 1.0]
    elif reward == 1 or reward == 2:
        return [1.0, 0.5, 0.5, 0.8, 0.8]
    elif reward == 3:
        return [1.0, 1.0, 1.0, 1.0, 1.0]
    elif reward == 4:
        return [1.0, 1.0, 1.0, 0.2, 0.2]
    else:
        return [1.0, 1.0, 1.0, 1.0, 1.0]

def add_gaussian_patch(heatmap, x, y, sigma=25, weight=1.0):
    """Adds a bounded 2D Gaussian patch to the high-res heatmap."""
    h, w = heatmap.shape
    size = int(sigma * 3)
    
    x0, x1 = max(0, x - size), min(w, x + size + 1)
    y0, y1 = max(0, y - size), min(h, y + size + 1)

    if x0 >= w or y0 >= h or x1 <= 0 or y1 <= 0:
        return 

    patch_x, patch_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    gaussian = weight * np.exp(-((patch_x - x)**2 + (patch_y - y)**2) / (2 * sigma**2))
    
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)

def convert_dataset_to_videos(dataset_path, output_dir, fps, resnet_stride=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]

    try:
        with h5py.File(dataset_path, 'r') as root:
            if 'observations/images' not in root:
                print(f"Error: Could not find 'observations/images' in {dataset_path}")
                return

            image_group = root['observations/images']
            camera_names = list(image_group.keys())
            
            attention_data = root['observations/attention_2d'][:] if 'observations/attention_2d' in root else None
            reward_data = root['reward'][:] if 'reward' in root else None

            for cam_name in camera_names:
                frames = image_group[cam_name][:]
                num_frames, height, width, channels = frames.shape
                
                is_attention_view = (cam_name == 'top' and attention_data is not None)
                out_width = width * 2 if is_attention_view else width

                output_filename = os.path.join(output_dir, f"{base_name}_{cam_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (out_width, height))

                print(f"Writing '{cam_name}' view to {output_filename}...")
                
                # Calculate the exact shape of the ResNet feature map (e.g., 15x20)
                feat_h = height // resnet_stride
                feat_w = width // resnet_stride
                
                for i in range(num_frames):
                    frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

                    if is_attention_view:
                        # 1. Create the high-res 480x640 canvas
                        heatmap_canvas = np.zeros((height, width), dtype=np.float32)
                        points = attention_data[i]
                        
                        current_reward = reward_data[i] if reward_data is not None else 0
                        weights = get_attention_weights(current_reward)
                        
                        for pt_idx, pt in enumerate(points):
                            x, y = pt
                            if x != -1 and y != -1: 
                                add_gaussian_patch(heatmap_canvas, x, y, sigma=25, weight=weights[pt_idx])
                        
                        # 2. Pool down to the ResNet feature shape (15x20)
                        # We use INTER_AREA because it acts like an Average Pooling layer, perfectly mimicking CNN behavior
                        heatmap_lowres = cv2.resize(heatmap_canvas, (feat_w, feat_h), interpolation=cv2.INTER_AREA)

                        # 3. Scale back up to 480x640 using INTER_NEAREST to visualize the blocky grid cells
                        heatmap_blocky = cv2.resize(heatmap_lowres, (width, height), interpolation=cv2.INTER_NEAREST)

                        # 4. Colorize and blend the blocky heatmap
                        heatmap_uint8 = (heatmap_blocky * 255).clip(0, 255).astype(np.uint8)
                        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

                        alpha = (heatmap_uint8 / 255.0)[..., None] 
                        overlay_bgr = (colored_heatmap * alpha + frame_bgr * (1 - alpha)).astype(np.uint8)

                        final_frame = np.hstack((frame_bgr, overlay_bgr))
                    else:
                        final_frame = frame_bgr

                    video_writer.write(final_frame)

                video_writer.release()
                print(f"Successfully saved {output_filename}")

    except Exception as e:
        print(f"An error occurred while processing {dataset_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ALOHA/Sim HDF5 dataset into mp4 videos.")
    parser.add_argument('--dataset', action='store', type=str, default='/mnt/Ego2Exo/sim_cube_transfer_same_objects_track/episode_10.hdf5', 
                        help='Path to the .hdf5 dataset file')
    parser.add_argument('--output_dir', action='store', type=str, default='/mnt/Ego2Exo/sim_cube_transfer_same_objects_track', 
                        help='Directory to save the output videos')
    parser.add_argument('--fps', action='store', type=int, default=50, 
                        help='Frames per second for the output video')
    parser.add_argument('--stride', action='store', type=int, default=32, 
                        help='Downsampling stride of the feature extractor (default: 32 for ResNet18)')
    
    args = parser.parse_args()
    convert_dataset_to_videos(args.dataset, args.output_dir, args.fps)