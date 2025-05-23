"""
# TODO: filter real to make it balanced with fake

import pandas as pd

DATASET_PATHS = "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/AVLips_format/deepspeak_all__test/1_fake"

df_11 = pd.read_csv("annotations-v11.csv")
df_2 = pd.read_csv("annotations-v2.csv")

# loop over all png files in these folders

# obtain structured information from the file name (e.g., v1_1_facefusion--118-119--1626-1737--0_0.png, v1_1_facefusion--118-119--1626-1737--0_1.png ...):

    # version: 1.1 (if filename starts with v1_1) or 2 (if filename starts with v2)
    # source_filename_core: everything between the version prefix (v1_1_ or v2_) and the frame number suffix (_{i}.png); above, for example, that would be 'facefusion--118-119--1626-1737--0'
    # frame number: in this case, 0 and 10, respectively

# then, determine the kind of video that this frame comes from (see kind extraction below; kind is a string)

    # in v1.1
    kind = df_11[df_11["video-file"] == source_filename_core + ".mp4"]["kind"].to_list()[0]

    # in v2
    kind = df_2[df_2["video-file"] == source_filename_core + ".mp4"]["kind"].to_list()[0]

# if the frame comes from a lip-sync (i.e., kind == 'lip-sync'), keep it; if it is NOT a lip-sync, however, remove the file!

# additionally -- print statistics on how many frames were in the folder before and after"""

import os
import pandas as pd
from glob import glob

DATASET_PATH = "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/AVLips_format/deepspeak_all__train/1_fake"

# Load the annotation CSVs
df_11 = pd.read_csv("/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/video_experiments/LipFD/annotations-v11.csv")
df_2 = pd.read_csv("/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/video_experiments/LipFD/annotations-v2.csv")

# Collect all PNG files
png_files = glob(os.path.join(DATASET_PATH, "*.png"))

print(f"Number of frames before filtering: {len(png_files)}")

remove_count = 0

for file_path in png_files:
    filename = os.path.basename(file_path)

    if filename.startswith("v1_1_"):
        version = "1.1"
        source_filename_core = filename[len("v1_1_"):].rsplit("_", 1)[0]
        df = df_11
    elif filename.startswith("v2_"):
        version = "2"
        source_filename_core = filename[len("v2_"):].rsplit("_", 1)[0]
        df = df_2
    else:
        print(f"Skipping unknown version file: {filename}")
        continue

    try:
        kind = df[df["video-file"] == source_filename_core + ".mp4"]["kind"].to_list()[0]
    except IndexError:
        print(f"Could not find video {source_filename_core}.mp4 in the annotations.")
        continue

    if kind != "lip-sync":
        os.remove(file_path)
        remove_count += 1

remaining_png_files = glob(os.path.join(DATASET_PATH, "*.png"))

print(f"Number of frames after filtering: {len(remaining_png_files)}")
print(f"Number of frames removed: {remove_count}")
