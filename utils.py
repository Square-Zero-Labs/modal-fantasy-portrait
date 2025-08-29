import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from diffsynth import save_video


def merge_audio_to_video(
    driven_video_path, save_video_path, save_video_path_with_audio
):
    audio_path = "temp_audio.aac"
    subprocess.run(
        ["ffmpeg", "-i", driven_video_path, "-vn", "-acodec", "aac", "-y", audio_path]
    )

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            save_video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            "-y",
            save_video_path_with_audio,
        ]
    )

    subprocess.run(["rm", audio_path])


def resize_image_by_longest_edge(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def extract_faces_from_video(video_path, output_dir, face_aligner):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR opening video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_videos = {}
    padding = 100
    bounding_boxes_first = None

    frame_crop = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = frame[:, :, ::-1]

        if bounding_boxes_first is None:
            (
                bounding_boxes,
                _,
                score,
            ) = face_aligner.face_alignment_module.face_detector.detect(rgb_frame)
            bounding_boxes = sorted(
                bounding_boxes, key=lambda box: (box[0] + box[2]) / 2
            )
            if len(bounding_boxes) == 2:
                bounding_boxes_first = bounding_boxes

        if bounding_boxes_first is not None:
            frame_crop.append(rgb_frame)
            for i, (x1, y1, x2, y2) in enumerate(bounding_boxes_first):
                top = max(0, y1 - padding)
                bottom = min(frame_height, y2 + padding)
                left = max(0, x1 - padding)
                right = min(frame_width, x2 + padding)

                face = frame[top:bottom, left:right][:, :, ::-1]

                if i not in face_videos:
                    face_videos[i] = {"frames": []}

                face_videos[i]["frames"].append(face)

    cap.release()

    for person_id, data in face_videos.items():
        if len(data["frames"]) > 0:
            os.makedirs(output_dir, exist_ok=True)
            save_video(
                data["frames"],
                os.path.join(output_dir, f"{person_id}.mp4"),
                fps=fps,
                quality=5,
            )
        else:
            print(f"person_{person_id} video is null.")

    print(f"success generate {len(face_videos)} single video.")


def change_video_fps(input_path, output_path, target_fps=25):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-r",
        str(target_fps),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-c:a",
        "copy",
        output_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Success! The video has been saved as: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error message: {e.stderr}")
        raise


def draw_rects_on_frames(frame_list, rect_lists, color="red", width=3):
    """
    Args:
        frame_list (List[PIL.Image]): A list of original video frames as PIL Images.
        rect_lists (List[List]): A list where each element is a list of rectangles for the corresponding frame, e.g., [[x1, y1, x2, y2], ...].
        color (str or Tuple[int, int, int]): The color of the bounding boxes, supporting string names (e.g., 'red') or RGB tuples (e.g., (255, 0, 0)).
        width (int): The width of the box lines.

    Returns:
        List[PIL.Image]: A new list of frames (PIL.Image) with the boxes drawn.
    """
    drawn_frames = []

    for i, image in enumerate(frame_list):
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        for j in range(len(rect_lists)):
            rect = rect_lists[j]

            x1, y1, x2, y2 = rect
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        drawn_frames.append(draw_image)

    return drawn_frames
