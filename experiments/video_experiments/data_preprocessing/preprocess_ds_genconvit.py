import os
import cv2
import random
import argparse
import face_recognition  # Make sure to install this: pip install face_recognition

parser = argparse.ArgumentParser(description="Video processing script saving only detected facial regions.")
parser.add_argument('--source_root', type=str, required=True)
parser.add_argument('--dest_root', type=str, required=True)
args = parser.parse_args()

source_root = args.source_root
dest_root = args.dest_root


def process_video(video_path, dest_dir, num_frames=10):
    """Extracts and saves randomly selected facial regions from a video file."""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: No frames found in {video_path}.")
        cap.release()
        return

    # Decide which frame indices to sample
    if total_frames < num_frames:
        indices = list(range(total_frames))
    else:
        indices = random.sample(range(total_frames), num_frames)
    indices.sort()

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    current_sample = 0
    frame_index = 0

    while cap.isOpened() and current_sample < len(indices):
        ret, frame = cap.read()
        if not ret:
            break

        # When the current frame is one of our sample indices, process and save the face region.
        if frame_index == indices[current_sample]:
            # Convert frame from BGR to RGB for face_recognition processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Detect face locations in the frame; assuming one face per video.
            face_locations = face_recognition.face_locations(rgb_frame)
            if not face_locations:
                print(f"No face found in frame {frame_index} of video: {video_path}")
            else:
                # Use the first detected face region
                top, right, bottom, left = face_locations[0]
                # Crop the frame to the facial region
                face_image = frame[top:bottom, left:right]
                image_filename = f"{base_name}_frame_{frame_index}_face.jpg"
                dest_path = os.path.join(dest_dir, image_filename)
                cv2.imwrite(dest_path, face_image)
            current_sample += 1

        frame_index += 1

    cap.release()


def main():
    # Traverse the source directory recursively
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(root, file)
                # Recreate the directory structure in the destination folder
                relative_path = os.path.relpath(root, source_root)
                dest_dir = os.path.join(dest_root, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                print(f"Processing video: {video_path}")
                process_video(video_path, dest_dir, num_frames=10)


if __name__ == '__main__':
    main()