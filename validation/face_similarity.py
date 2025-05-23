"""
Module for face similarity validation using CLIP embeddings and pixel-based comparisons.
"""

import cv2
import random
import torch
import clip

import numpy as np
import mediapipe as mp

from PIL import Image
from scipy.spatial.distance import cosine


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def extract_faces_from_frame(frame):
    """
    Extracts the face region from a given video frame.

    Args:
        frame (np.ndarray): The input video frame.

    Returns:
        np.ndarray or None: Cropped face region or None if no face is detected.
    """
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        h, w, _ = frame.shape
        detection = results.detections[0]  # Assume single face per frame
        bboxC = detection.location_data.relative_bounding_box
        x, y, w_box, h_box = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                              int(bboxC.width * w), int(bboxC.height * h))
        return frame[y:y + h_box, x:x + w_box]  # Return cropped face
    return None


def extract_clip_embedding(image):
    """
    Extracts a CLIP embedding for a given image.

    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The CLIP embedding vector.
    """
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_tensor).cpu().numpy().flatten()
    return embedding


def compute_pixel_similarity(face1, face2, size=(64, 64)):
    """
    Computes pixel-based similarity between two face images.

    Args:
        face1 (np.ndarray): The first face image.
        face2 (np.ndarray): The second face image.
        size (tuple): The size to which the images are resized.

    Returns:
        float: The cosine distance between the two images.
    """
    face1_resized = cv2.resize(face1, size)
    face2_resized = cv2.resize(face2, size)

    face1_flat = face1_resized.flatten().astype(np.float32)
    face2_flat = face2_resized.flatten().astype(np.float32)

    return cosine(face1_flat, face2_flat)


def sample_video_frames(video_path, num_frames=5):
    """
    Samples a specified number of frames from a video.

    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to sample.

    Returns:
        list: List of sampled frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(num_frames, frame_count)
    frame_indices = sorted(random.sample(range(frame_count), sample_count))

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def get_reference_faces(reference_video, num_samples=5):
    """
    Extracts face crops from sampled frames of a reference video.

    Args:
        reference_video (str): Path to the reference video.
        num_samples (int): Number of frames to sample.

    Returns:
        list: List of cropped face images.

    Raises:
        ValueError: If no faces are detected in the video.
    """
    frames = sample_video_frames(reference_video, num_frames=num_samples)
    faces = []
    for frame in frames:
        face_crop = extract_faces_from_frame(frame)
        if face_crop is not None:
            faces.append(face_crop)
    if not faces:
        raise ValueError("No faces detected in the reference video!")
    return faces


def compute_min_similarity(examined_video, reference_faces):
    """
    Computes the minimum similarity between faces in an examined video and reference faces.

    Args:
        examined_video (str): Path to the examined video.
        reference_faces (list): List of reference face images.

    Returns:
        tuple: Minimum CLIP distance and pixel distance.
    """
    ref_clip_embeddings = [extract_clip_embedding(face) for face in reference_faces]

    min_clip_distance = float("inf")
    min_pixel_distance = float("inf")

    cap = cv2.VideoCapture(examined_video)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        examined_face = extract_faces_from_frame(frame)
        if examined_face is None:
            continue

        examined_clip_embedding = extract_clip_embedding(examined_face)

        for ref_face, ref_embedding in zip(reference_faces, ref_clip_embeddings):
            clip_distance = cosine(ref_embedding, examined_clip_embedding)
            pixel_distance = compute_pixel_similarity(ref_face, examined_face)

            if clip_distance < min_clip_distance:
                min_clip_distance = clip_distance
            if pixel_distance < min_pixel_distance:
                min_pixel_distance = pixel_distance

    cap.release()
    return min_clip_distance, min_pixel_distance


def detect_faces_identical(reference_video, examined_video):
    """
    Detects if faces in the examined video are identical to those in the reference video.

    Args:
        reference_video (str): Path to the reference video.
        examined_video (str): Path to the examined video.

    Returns:
        bool: True if faces are identical, False otherwise.
    """
    reference_faces = get_reference_faces(reference_video, num_samples=5)
    min_clip_distance, min_pixel_distance = compute_min_similarity(examined_video, reference_faces)

    return min_clip_distance < 0.03 or min_pixel_distance < 0.002


if __name__ == "__main__":
    pass