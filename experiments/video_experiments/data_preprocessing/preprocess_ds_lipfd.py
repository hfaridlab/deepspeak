import os
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from moviepy import VideoFileClip
import argparse


parser = argparse.ArgumentParser(description="")

parser.add_argument('--video_root', type=str, required=True)
parser.add_argument('--output_root', type=str, required=True)
parser.add_argument('--split', type=str, choices=['train', 'test'], default='train', help='Dataset split to process (train or test)')

args = parser.parse_args()


############ Custom parameters ##############
N_EXTRACT = 15  # number of starting points for extraction from video
WINDOW_LEN = 5  # number of consecutive frames per window
MAX_SAMPLE = 5000  # process at most 100 videos per class per split

# Dataset root for DeepSpeak
video_root = args.video_root  # "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/deepspeak_v1_1"
# Output directory (same format as AVLips, i.e. two folders: 0_real and 1_fake)
output_root = args.output_root  # "/home/ubuntu/deepspeak-experiments/deepspeak-neurips-2025/deepspeak_v1_1__test_AVLips_format"

splits = [args.split]

# Temporary folder for intermediate files
temp_dir = "./temp"
temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

# Mapping: DeepSpeak labels to output folder names
# "real" becomes "0_real" and "fake" becomes "1_fake"
label_mapping = {"real": "0_real", "fake": "1_fake"}


def get_spectrogram(audio_file):
    """Extract a mel-spectrogram from the given audio file and save it as a temporary image."""
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(os.path.join(temp_dir, "mel.png"), mel)


def extract_audio_from_video(video_file, audio_temp_path):
    """Extract the audio from the video and save it to a temporary file."""
    clip = VideoFileClip(video_file)
    # Suppress verbose output by setting logger to None
    clip.audio.write_audiofile(audio_temp_path, logger=None)
    clip.close()


def process_video(video_file, output_folder, base_name):
    """
    Process a single video:
      - Extract frames at specified windows.
      - Extract audio, compute its mel-spectrogram.
      - Combine the spectrogram segment with concatenated frames.
      - Save the composite image(s) to the output folder.
    """
    # Open video file for frame extraction
    video_capture = cv2.VideoCapture(video_file)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select N_EXTRACT starting frame indices (ensuring we have enough frames for a full window)
    frame_idx = np.linspace(
        0,
        frame_count - WINDOW_LEN - 1,
        N_EXTRACT,
        endpoint=True,
        dtype=np.uint8,
    ).tolist()
    frame_idx.sort()

    # Create a list of frame indices covering each window
    frame_sequence = [i for num in frame_idx for i in range(num, num + WINDOW_LEN)]
    frame_list = []
    current_frame = 0
    while current_frame <= frame_sequence[-1]:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error reading frame {video_file} at frame {current_frame}")
            break
        if current_frame in frame_sequence:
            # Convert BGR to RGBA and resize frame to 500x500 pixels
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            frame_list.append(cv2.resize(frame, (500, 500)))
        current_frame += 1
    video_capture.release()

    # Extract audio from the video file and save to a temporary file
    extract_audio_from_video(video_file, temp_audio_path)
    # Compute the mel-spectrogram and save it as an image in the temp folder
    get_spectrogram(temp_audio_path)
    mel = plt.imread(os.path.join(temp_dir, "mel.png")) * 255  # scale to int
    mel = mel.astype(np.uint8)

    # Mapping factor from video frames to spectrogram columns
    mapping = mel.shape[1] / frame_count
    group = 0
    # Process frames in groups defined by WINDOW_LEN
    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        if idx == 0:
            try:
                # Calculate corresponding spectrogram segment for this group of frames
                begin = np.round(frame_sequence[i] * mapping)
                end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
                sub_mel = cv2.resize(
                    mel[:, int(begin):int(end)],
                    (500 * WINDOW_LEN, 500)
                )
                # Concatenate the WINDOW_LEN frames horizontally
                x = np.concatenate(frame_list[i: i + WINDOW_LEN], axis=1)
                # Stack the spectrogram segment on top of the frames
                composite = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
                output_path = os.path.join(output_folder, f"{base_name}_{group}.png")
                plt.imsave(output_path, composite)
                group += 1
            except ValueError:
                print(f"ValueError processing {video_file}")
                continue


def run():
    """
    Process both the 'train' and 'test' splits of DeepSpeak.
    Videos from 'real' and 'fake' folders are processed and saved
    into the corresponding output folders ('0_real' and '1_fake').
    """

    for split in tqdm(splits, desc="Splits"):
        for label in tqdm(["real", "fake"], desc=f"Labels in {split}", leave=False):
            video_dir = os.path.join(video_root, split, label)
            output_label = label_mapping[label]
            output_folder = os.path.join(output_root, output_label)
            os.makedirs(output_folder, exist_ok=True)
            video_list = os.listdir(video_dir)
            sample_count = 0
            for video_file in tqdm(video_list, desc=f"Processing {split}/{label}", leave=False):
                if sample_count >= MAX_SAMPLE:
                    break
                video_path = os.path.join(video_dir, video_file)
                base_name = os.path.splitext(video_file)[0]

                try:
                    process_video(video_path, output_folder, base_name)
                    sample_count += 1
                except:
                    print("there was an issue reading this video")


if __name__ == "__main__":
    os.makedirs(output_root, exist_ok=True)
    run()
