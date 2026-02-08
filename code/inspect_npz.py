#!/usr/bin/env python3
"""
Inspect what's stored in .npz files
"""
import pickle
import sys

def inspect_npz_file(npz_path):
    try:
        with open(npz_path, 'rb') as file:
            data = pickle.loads(file.read())

        print(f"Contents of {npz_path}:")
        print("=" * 50)

        for key in data.keys():
            value = data[key]
            if hasattr(value, 'shape'):
                print(f"{key}: {type(value).__name__} with shape {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"{key}: {type(value).__name__} with length {len(value)}")
            else:
                print(f"{key}: {type(value).__name__} = {value}")

        # Check specifically for burst_positions
        if 'burst_positions' in data:
            print("\nBurst positions details:")
            for i, burst in enumerate(data['burst_positions']):
                print(f"  Burst {i}: {burst}")

    except Exception as e:
        print(f"Error reading {npz_path}: {e}")

if __name__ == "__main__":
    # if len(sys.argv) > 1:
    #     inspect_npz_file(sys.argv[1])
    # else:
    #     # Use a default file

    file_path = "/src/data/SenSig-test_burst/0.npz"
    inspect_npz_file(file_path)