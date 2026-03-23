import h5py
import cv2
import argparse
import os

def convert_dataset_to_videos(dataset_path, output_dir, fps):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the base filename (e.g., 'episode_0' from 'data/episode_0.hdf5')
    base_name = os.path.splitext(os.path.basename(dataset_path))[0]

    try:
        with h5py.File(dataset_path, 'r') as root:
            # Check if the expected image group exists
            if 'observations/images' not in root:
                print(f"Error: Could not find 'observations/images' in {dataset_path}")
                return

            image_group = root['observations/images']
            camera_names = list(image_group.keys())
            
            print(f"Found {len(camera_names)} camera view(s): {', '.join(camera_names)}")

            # Loop through each camera view and generate a video
            for cam_name in camera_names:
                # Load all frames into memory
                frames = image_group[cam_name][:]
                num_frames, height, width, channels = frames.shape
                
                # Setup video writer
                output_filename = os.path.join(output_dir, f"{base_name}_{cam_name}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
                video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

                print(f"Writing '{cam_name}' view ({num_frames} frames) to {output_filename}...")
                
                for i in range(num_frames):
                    frame_rgb = frames[i]
                    # The simulator saves in RGB, but OpenCV expects BGR to write correctly
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    video_writer.write(frame_bgr)

                video_writer.release()
                print(f"Successfully saved {output_filename}")

    except Exception as e:
        print(f"An error occurred while processing {dataset_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ALOHA/Sim HDF5 dataset into mp4 videos.")
    parser.add_argument('--dataset', action='store', type=str, default='/mnt/Ego2Exo/sim_transfer_new_distractos/episode_1.hdf5', 
                        help='Path to the .hdf5 dataset file (e.g., episode_0.hdf5)')
    parser.add_argument('--output_dir', action='store', type=str, default='.', 
                        help='Directory to save the output videos (default: current directory)')
    parser.add_argument('--fps', action='store', type=int, default=50, 
                        help='Frames per second for the output video (default: 50)')
    
    args = parser.parse_args()
    convert_dataset_to_videos(args.dataset, args.output_dir, args.fps)

