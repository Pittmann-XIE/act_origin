import h5py

def print_hdf5_structure(name, obj):
    """
    Callback function to print the name and type of each object in the file.
    """
    indent = name.count('/') * '  '
    if isinstance(obj, h5py.Group):
        print(f"{indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name} (shape: {obj.shape}, type: {obj.dtype})")

# Replace 'your_file.h5' with your actual file path
file_path = '/home/fe/xie/episode_0.hdf5'

with h5py.File(file_path, 'r') as f:
    print(f"Structure of {file_path}:")
    f.visititems(print_hdf5_structure)