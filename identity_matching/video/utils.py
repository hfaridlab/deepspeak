
import os
import yaml
import cv2
import PIL


def extract_nth_frame(video_path: str, n: int) -> PIL.Image:
    """
    Extracts the nth frame from a video file and returns it as a PIL Image.

    :param video_path: path to the video file
    :param n: frame number to extract (1-indexed)
    :return: PIL Image of the extracted frame, or None if the frame cannot be read
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file does not exist, {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Opening of video failed, {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, n-1)
    success, frame = cap.read()
    if not success:
        return None

    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = PIL.Image.fromarray(frame_rgb)

    return img


def load_config(config_path):
    """
    Loads a YAML config from the provided path in a safe mode.

    :param config_path: path to the .yaml file
    :return: parsed YAML Python object
    """

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    pass
