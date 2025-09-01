import cv2
import os
import tempfile
from typing import List, Tuple
import subprocess
import numpy as np


def compute_segments(total_frames: int, window_len: int = 117, overlap: int = 17) -> List[Tuple[int, int]]:
    """Return (start, length) pairs covering total_frames with sliding windows.
    Enforces length and overlap to be 4k+1 for Wan latents.
    """
    # Snap window_len and overlap to 4k+1
    def snap(n: int) -> int:
        if n <= 1:
            return 1
        # nearest below that satisfies (n-1)%4==0
        return n - ((n - 1) % 4)

    L = snap(window_len)
    O = snap(overlap)
    if O >= L:
        O = L - 4  # ensure overlap smaller than window
        O = snap(O)

    starts = list(range(0, max(total_frames - 1, 0), L - O))
    segments: List[Tuple[int, int]] = []
    for s in starts:
        length = min(L, total_frames - s)
        length = snap(length)
        if length <= 0:
            continue
        segments.append((s, length))
        if s + length >= total_frames:
            break
    # Ensure last segment reaches end by adjusting start if needed
    if segments:
        last_s, last_l = segments[-1]
        if last_s + last_l < total_frames:
            new_start = max(0, total_frames - L)
            segments[-1] = (new_start, snap(min(L, total_frames - new_start)))
    return segments


# Removed old crossfade-based stitchers; using precise ffmpeg concat below.


def concat_pair_noblend_ffmpeg(v0_path: str, v1_path: str, overlap_frames: int, fps: int, out_path: str) -> str:
    """Concatenate two segments without crossfade, trimming the last overlap_frames
    from the first segment to avoid duplication. Uses ffmpeg filter_complex for
    precise timing and re-encodes to H.264.
    """
    cap0 = cv2.VideoCapture(v0_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Failed to open segment: {v0_path}")
    frames0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()
    keep0 = max(0, frames0 - overlap_frames)

    # ffmpeg concat via frame-accurate trim + concat filter
    cmd = [
        "ffmpeg", "-y",
        "-i", v0_path,
        "-i", v1_path,
        "-filter_complex",
        f"[0:v]trim=start_frame=0:end_frame={keep0},setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1[out]",
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def concat_no_blend(segment_paths: List[str], overlap_frames: int, fps: int, out_path: str) -> str:
    """Concatenate multiple segments without blending using pairwise ffmpeg trims.
    Produces a single H.264 file at the given fps.
    """
    if not segment_paths:
        raise ValueError("No segment paths provided")
    current = segment_paths[0]
    for idx in range(1, len(segment_paths)):
        nxt = segment_paths[idx]
        tmp_out = out_path if idx == len(segment_paths) - 1 else tempfile.mktemp(suffix=".mp4")
        current = concat_pair_noblend_ffmpeg(current, nxt, overlap_frames, fps, tmp_out)
    return current


def merge_audio_to_video(driven_video_path: str, save_video_path: str, save_video_path_with_audio: str) -> None:
    """Mux audio from driven video with stitched video into a final H.264 file.
    Re-encodes video to H.264/yuv420p to ensure compatibility and sync.
    """
    audio_path = "temp_audio.aac"
    # Extract audio
    subprocess.run([
        "ffmpeg", "-y", "-i", driven_video_path, "-vn", "-acodec", "aac", audio_path
    ], check=False)
    # Mux
    subprocess.run([
        "ffmpeg", "-y",
        "-i", save_video_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-shortest",
        save_video_path_with_audio
    ], check=False)
    try:
        os.remove(audio_path)
    except Exception:
        pass


def color_match_next_to_prev(prev_path: str, next_path: str, overlap_frames: int, fps: int) -> str:
    """Adjust the first `overlap_frames` frames of `next_path` to match the color statistics
    (per-channel mean/std) of the last `overlap_frames` frames of `prev_path`. Returns the path
    to a new adjusted video (temp file).
    """
    cap0 = cv2.VideoCapture(prev_path)
    cap1 = cv2.VideoCapture(next_path)
    if not cap0.isOpened() or not cap1.isOpened():
        raise RuntimeError("Failed to open segment videos for color matching.")
    w = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames0 = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    # Read tail of prev
    tail0 = []
    cap0.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames0 - overlap_frames))
    for _ in range(overlap_frames):
        ok, f = cap0.read()
        if not ok:
            break
        tail0.append(f)
    cap0.release()
    if not tail0:
        cap1.release()
        return next_path
    # Compute mean/std for prev tail
    tail0_np = np.stack(tail0, axis=0).astype(np.float32)
    mean0 = tail0_np.mean(axis=(0, 1, 2))
    std0 = tail0_np.std(axis=(0, 1, 2)) + 1e-6

    # Prepare writer
    tmp_out = tempfile.mktemp(suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap1.release()
        raise RuntimeError("Failed to open VideoWriter for color-matched output.")

    # Process next video
    cnt = 0
    head_buf = []
    while cnt < overlap_frames:
        ok, f = cap1.read()
        if not ok:
            break
        head_buf.append(f)
        cnt += 1
    # Compute stats of next head
    if head_buf:
        head_np = np.stack(head_buf, axis=0).astype(np.float32)
        mean1 = head_np.mean(axis=(0, 1, 2))
        std1 = head_np.std(axis=(0, 1, 2)) + 1e-6
        scale = std0 / std1
        shift = mean0 - mean1 * scale
        # Apply to head frames
        for f in head_buf:
            g = f.astype(np.float32)
            g = g * scale + shift
            g = np.clip(g, 0, 255).astype(np.uint8)
            writer.write(g)
    # Write remainder frames unchanged
    while True:
        ok, f = cap1.read()
        if not ok:
            break
        writer.write(f)
    cap1.release()
    writer.release()
    return tmp_out
