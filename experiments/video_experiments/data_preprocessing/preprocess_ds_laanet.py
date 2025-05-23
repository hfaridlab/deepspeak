#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse
import sys
import torch
from retinaface.pre_trained_models import get_model

# Constants for saving cropped faces
ROOT = '/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/FreqNet-format/'

IMAGE_H, IMAGE_W, IMAGE_C = 256, 256, 3
PADDING = 0.25


def facecrop(model, org_path, save_path, num_frames=10, dataset='deepspeak', label=None, padding=PADDING):
    """
    Processes a video by extracting selected frames, detecting faces using retinaface,
    cropping the largest detected face (with some padding), resizing it, and then saving
    the result as an image. The output is stored in:

    save_path/frames/{dataset}/{label}/{video_name}/<frame_number>.png
    """
    print(f'Processing video --- {org_path}')
    cap_org = cv2.VideoCapture(org_path)
    if not cap_org.isOpened():
        print(f'Error opening video: {org_path}')
        return

    frame_count_org = int(cap_org.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count:", frame_count_org)

    # Build output folder path
    base_name = os.path.basename(org_path).replace('.mp4', '')
    if label is not None:
        save_path_ = os.path.join(save_path, 'frames', dataset, str(label), base_name)
    else:
        save_path_ = os.path.join(save_path, 'frames', dataset, base_name)
    os.makedirs(save_path_, exist_ok=True)

    # Select indices of frames to process
    frame_idxs = np.linspace(0, frame_count_org - 1, num_frames, endpoint=True, dtype=np.int64)

    for cnt_frame in range(frame_count_org):
        # Only process frames that are in our selected indices
        if cnt_frame not in frame_idxs:
            # Skip reading the frame if not in selected indices
            cap_org.grab()
            continue

        ret_org, frame_org = cap_org.read()
        if not ret_org:
            print(f"Error reading frame {cnt_frame} in {org_path}")
            break

        height, width = frame_org.shape[:2]
        # Convert to RGB for the model
        frame_rgb = cv2.cvtColor(frame_org, cv2.COLOR_BGR2RGB)

        # Detect faces using the retinaface model
        faces = model.predict_jsons(frame_rgb)
        if len(faces) == 0:
            print(f'No faces detected in frame {cnt_frame} of {org_path}')
            continue

        # Select the face with the largest area (and highest score)
        face_s_max = -1
        face_crop = None
        score_max = -1
        for face in faces:
            x0, y0, x1, y1 = face['bbox']
            face_w = x1 - x0
            face_h = y1 - y0
            face_area = face_w * face_h
            score = face['score']
            if face_area > face_s_max and score > score_max:
                f_c_x0 = max(0, int(x0 - face_w * padding))
                f_c_x1 = min(width, int(x1 + face_w * padding))
                f_c_y0 = max(0, int(y0 - face_h * padding))
                f_c_y1 = min(height, int(y1 + face_h * padding))
                face_crop = frame_org[f_c_y0:f_c_y1, f_c_x0:f_c_x1, :]
                face_s_max = face_area
                score_max = score

        if face_crop is None:
            print(f'No valid face crop found for frame {cnt_frame} in {org_path}')
            continue

        # Resize cropped face to fixed dimensions
        face_crop = cv2.resize(face_crop, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
        image_filename = os.path.join(save_path_, str(cnt_frame).zfill(3) + '.png')
        cv2.imwrite(image_filename, face_crop)

    cap_org.release()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert DeepSpeak dataset to new structure with face crops")
    parser.add_argument('--deepspeak_dir', type=str, default='/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/deepspeak_v1_1',
                        help='Path to the DeepSpeak dataset root')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='train',
                        help='Dataset split to process (train or test)')
    parser.add_argument('--save_dir', type=str, required=True) # SAVE_DIR = "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/LAANet_format/deepspeak_v1_1__train"
    parser.add_argument('--label', type=str, choices=['real', 'fake'], required=True,
                        help='Label to process (real or fake)')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to extract from each video')
    args = parser.parse_args()

    # Build the path to the videos for the given split and label
    video_dir = os.path.join(args.deepspeak_dir, args.split, args.label)
    video_files = glob(os.path.join(video_dir, '*.mp4'))
    if not video_files:
        print(f"No video files found in {video_dir}")
        sys.exit(1)

    # Initialize the retinaface model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    model.eval()

    print(f"Processing {len(video_files)} videos from {video_dir}")

    counter_of_unprocessable = 0

    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            facecrop(model, video_file, args.save_dir, num_frames=args.num_frames,
                    dataset='deepspeak', label=args.label)
        except:
            counter_of_unprocessable += 1
            print("an error occurred with the face detection")

    print("total unprocessed", counter_of_unprocessable)
