"""
Main module for validating deepfake videos and generating review reports.
"""

import os
import ffmpeg
import librosa
import cv2
import tqdm
import argparse

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine
from validation.face_similarity import detect_faces_identical


def parse_args():
    """
    Parses command-line arguments for the validation script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Validate deepfake videos and generate review report.")
    parser.add_argument(
        "--df_generation_path",
        type=str,
        nargs='+',
        help="Paths to one or more CSV files for generation plan",
        default=[
            "/home/ubuntu/deepspeak/deepspeak-orchestrator/orchestration/output/csv/generation_plan-v2-2025-03-25--new-identities-add.csv"
        ]
    )
    parser.add_argument("--deepfake_dataset_path", help="Path to the deepfake dataset.", default="/home/ubuntu/deepspeak/data/deepspeak_v2_output_new_identities")
    parser.add_argument("--source_videos_path", help="Path to the source videos.", default="/home/ubuntu/deepspeak/data/deepspeak_v2_real_video_fake_audio")
    parser.add_argument("--df_mapper", type=str, help="Path to the mapper CSV file.", default="/home/ubuntu/deepspeak/data/deepspeak_v2_real_video_fake_audio/transcript_question_number_mapper.csv")
    parser.add_argument("--report_csv", type=str, help="Path to save the validation report.", default="output-report-v2--new-identities.csv")
    return parser.parse_args()


def extract_audio(input_path: str, sr: int = 22050):
    """
    Extracts audio from a video or loads it directly from an audio file.

    Args:
        input_path (str): Path to the input file (audio or video).
        sr (int): Sampling rate for audio extraction.

    Returns:
        np.ndarray: Extracted audio signal.

    Raises:
        ValueError: If the file format is unsupported.
    """
    file_ext = os.path.splitext(input_path)[1].lower()

    if file_ext in {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}:  # Directly load audio
        audio, _ = librosa.load(input_path, sr=sr)
        return audio

    elif file_ext in {".mp4", ".mkv", ".avi", ".mov", ".webm"}:  # Extract audio from video
        out_audio_path = "temp_audio.wav"
        (
            ffmpeg.input(input_path)
            .output(out_audio_path, format="wav", acodec="pcm_s16le", ac=1, ar=sr)
            .run(overwrite_output=True, quiet=True)
        )
        audio, _ = librosa.load(out_audio_path, sr=sr)
        os.remove(out_audio_path)  # Clean up temporary file
        return audio

    else:
        raise ValueError("Unsupported file format. Provide a valid video or audio file.")


def get_audio_similarity(file_0_path: str, file_1_path: str) -> float:
    """
    Computes the cosine similarity between the mel spectrograms of two audio sources.

    Args:
        file_0_path (str): Path to the first audio file.
        file_1_path (str): Path to the second audio file.

    Returns:
        float: Cosine similarity between the two audio sources.
    """
    if not os.path.exists(file_0_path) or not os.path.exists(file_1_path):
        print("File does not exist.")
        return 0

    audio_0 = extract_audio(file_0_path)
    audio_1 = extract_audio(file_1_path)

    min_length = min(len(audio_0), len(audio_1))
    audio_0 = audio_0[:min_length]
    audio_1 = audio_1[:min_length]

    spec_0 = librosa.feature.melspectrogram(y=audio_0, sr=22050)
    spec_1 = librosa.feature.melspectrogram(y=audio_1, sr=22050)

    spec_0_flat = spec_0.flatten()
    spec_1_flat = spec_1.flatten()

    return 1 - cosine(spec_0_flat, spec_1_flat)


def is_video_black(video_path: str, threshold: float = 10.0) -> bool:
    """
    Checks if a video is almost entirely black.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Intensity threshold for detecting black frames.

    Returns:
        bool: True if the video is mostly black, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    frame_count = 0
    black_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_intensity = np.mean(gray)

        if mean_intensity < threshold:
            black_frames += 1

    cap.release()
    return (black_frames / frame_count) > 0.9 if frame_count > 0 else False


def validate_generated_deepfake(deepfake_vid_path: str, source_vid_path: str, target_vid_path: str):
    """
    Placeholder for deepfake video validation logic.

    Args:
        deepfake_vid_path (str): Path to the generated deepfake video.
        source_vid_path (str): Path to the source video.
        target_vid_path (str): Path to the target video.
    """
    pass


if __name__ == "__main__":
    args = parse_args()

    # Load and merge all generation CSV files into one DataFrame
    df_list = [pd.read_csv(csv_path) for csv_path in args.df_generation_path]
    df_generation = pd.concat(df_list, ignore_index=True)

    df_mapper = pd.read_csv(args.df_mapper)
    deepfake_dataset_path = args.deepfake_dataset_path
    source_videos_path = args.source_videos_path

    issues = []

    for i, row in tqdm.tqdm(df_generation.iterrows(), total=len(df_generation)):

        components = row["video-file"].split("/")
        local_video_path = os.path.join(deepfake_dataset_path, components[-2], components[-1])

        try:
            if not os.path.exists(local_video_path):
                continue

            from orchestration.launch import get_video_file_from_recording_id, get_audio_file_from_recording_id

            source_vid = get_video_file_from_recording_id(
                row["identity-source"], row["recording-source"], source_videos_path, df_mapper
            )
            source_aud = get_audio_file_from_recording_id(
                row["identity-source"], row["recording-source"], source_videos_path, df_mapper, row["audio-config"]
            )
            target_vid = get_video_file_from_recording_id(
                row["identity-target"], row["recording-target"], source_videos_path, df_mapper
            )
            target_aud = get_audio_file_from_recording_id(
                row["identity-target"], row["recording-target"], source_videos_path, df_mapper, row["audio-config"]
            )

            # Check if the video is almost entirely black
            vid_all_black = is_video_black(local_video_path)
            if vid_all_black:
                print("All black")
                issues.append({
                    "video_path": local_video_path,
                    "issue_type": "All black"
                })

            # Check audio similarity based on the engine type
            if row["engine"] in ["latentsync", "diff2lip"]:
                aud_sim = get_audio_similarity(local_video_path, target_aud)
                if aud_sim > 0.4:
                    print(f"Audio similarity review (target aud): {aud_sim:.2f}")
                    issues.append({
                        "video_path": local_video_path,
                        "issue_type": f"Audio similarity review (target aud): {aud_sim:.2f}"
                    })

            if row["engine"] in ["memo", "facefusion", "retalking", "wav2lip"]:
                aud_sim = get_audio_similarity(local_video_path, source_aud)
                if aud_sim > 0.4:
                    print(f"Audio similarity review (source aud): {aud_sim:.2f}")
                    issues.append({
                        "video_path": local_video_path,
                        "issue_type": f"Audio similarity review (source aud): {aud_sim:.2f}"
                    })

            if row["engine"] in ["facefusion", "facefusion_gan", "facefusion_live"]:
                if not os.path.exists(target_vid) or not os.path.exists(local_video_path):
                    print(f"Skipping face identity check. Missing file: {target_vid} or {local_video_path}")
                else:
                    if detect_faces_identical(target_vid, local_video_path):
                        print("Faces identical")
                        issues.append({
                            "video_path": local_video_path,
                            "issue_type": "Faces identical"
                        })

            df_report = pd.DataFrame(issues)
            df_report.to_csv(args.report_csv, index=False)

        except:
            issues.append({
                "video_path": local_video_path,
                "issue_type": f"Validation failed for {local_video_path}"
            })
            df_report = pd.DataFrame(issues)
            df_report.to_csv(args.report_csv, index=False)

    if issues:
        df_report = pd.DataFrame(issues)
        df_report.to_csv(args.report_csv, index=False)
        print(f"Review report generated: {args.report_csv}")
    else:
        print("No review issues identified.")