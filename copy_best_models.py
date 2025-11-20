#!/usr/bin/env python3
"""
Script to copy best.pth files from output subdirectories to saved_models.
"""
import os
import shutil

# Ensure saved_models directory exists
os.makedirs('saved_models', exist_ok=True)

# Iterate through each subdirectory in output
for subdir in os.listdir('output'):
    subdir_path = os.path.join('output', subdir)
    if os.path.isdir(subdir_path):
        best_pth_path = os.path.join(subdir_path, 'model', 'best.pth')
        if os.path.isfile(best_pth_path):
            dst_path = os.path.join('saved_models', f'{subdir}.pth')
            shutil.copy(best_pth_path, dst_path)
            print(f'Copied {best_pth_path} to {dst_path}')
        else:
            print(f'No best.pth found in {subdir_path}')
    else:
        print(f'Skipping non-directory: {subdir}')
