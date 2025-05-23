
import os
import random

# Set the folder path and number of files to keep
folder_path = "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/AVLips_format/deepspeak_all__train/0_real"
num_files_to_keep = 72558

# Get list of all files in the folder
all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
total_files = len(all_files)

# Sanity check
if num_files_to_keep > total_files:
    raise ValueError(f"Requested to keep {num_files_to_keep} files, but only {total_files} files found.")

# Randomly select files to keep
files_to_keep = set(random.sample(all_files, num_files_to_keep))

# Remove files not selected
for file_name in all_files:
    if file_name not in files_to_keep:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)

print(f"Kept {num_files_to_keep} files and removed {total_files - num_files_to_keep} files.")
