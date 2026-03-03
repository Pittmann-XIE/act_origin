import torch
import numpy as np

def load_state_dict_robust(path):
    """
    Loads a checkpoint and extracts the model weights, regardless of 
    whether it's a full checkpoint or just weights.
    """
    print(f"Loading {path}...")
    try:
        # Load to CPU to avoid CUDA OOM errors during comparison
        checkpoint = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # FIX FOR YOUR ERROR:
    # Check if the file is a nested dict (contains optimizer, steps, etc.)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        print(f"  -> Detected full checkpoint (found keys: {list(checkpoint.keys())})")
        print("  -> Extracting 'model_state_dict'...")
        return checkpoint['model_state_dict']
    
    # Fallback: Assumption that it is a direct state dict
    return checkpoint

def compare_checkpoints(path_a, path_b):
    sd_a = load_state_dict_robust(path_a)
    sd_b = load_state_dict_robust(path_b)

    if sd_a is None or sd_b is None:
        return

    # Get keys
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    # Check for architecture mismatches
    if keys_a != keys_b:
        print("\nWARNING: Checkpoint keys do not match!")
        only_in_a = keys_a - keys_b
        only_in_b = keys_b - keys_a
        if only_in_a: print(f"Keys only in A: {list(only_in_a)[:5]} ...")
        if only_in_b: print(f"Keys only in B: {list(only_in_b)[:5]} ...")
        
        # Intersect keys to continue comparison on common layers
        common_keys = keys_a.intersection(keys_b)
    else:
        print("\nModel Architecture (keys) matches perfectly.")
        common_keys = keys_a

    print("\n--- Comparison Report ---")
    
    changed_count = 0
    unchanged_count = 0
    total_diff = 0.0

    print(f"{'Layer Name':<50} | {'Status':<10} | {'L1 Diff (Sum)':<15}")
    print("-" * 80)

    for key in sorted(common_keys):
        # Convert to float and numpy for calculation
        tensor_a = sd_a[key].float()
        tensor_b = sd_b[key].float()

        # Check shapes
        if tensor_a.shape != tensor_b.shape:
            print(f"{key:<50} | SHAPE MISMATCH ({tensor_a.shape} vs {tensor_b.shape})")
            continue

        # Calculate difference
        diff = torch.abs(tensor_a - tensor_b)
        total_l1 = diff.sum().item()
        max_diff = diff.max().item()

        if total_l1 == 0:
            unchanged_count += 1
        else:
            changed_count += 1
            total_diff += total_l1
            # Print only layers that changed (or remove if check to see all)
            print(f"{key:<50} | CHANGED    | {total_l1:.6f}")

    print("-" * 80)
    print(f"Total Layers Checked: {len(common_keys)}")
    print(f"Unchanged Layers:     {unchanged_count}")
    print(f"Changed Layers:       {changed_count}")
    print(f"Total L1 Difference:  {total_diff:.4f}")

if __name__ == "__main__":
    # REPLACE THESE PATHS WITH YOUR FILE PATHS
    ckpt_1 = "/home/pengtao/ws_ros2humble-main_lab/act/ckpt/policy_best.ckpt" 
    ckpt_2 = "/home/pengtao/ws_ros2humble-main_lab/act/checkpoints/policy_best.ckpt"
    
    compare_checkpoints(ckpt_1, ckpt_2)