import argparse
import math
import os
import subprocess
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from diffsynth import ModelManager, WanVideoPipeline
from diffsynth.data import save_video
from diffsynth.models.camer import CameraDemo
from diffsynth.models.face_align import FaceAlignment
from diffsynth.models.pdf import (FanEncoder, det_landmarks,
                                  get_drive_expression_pd_fgc)
from diffsynth.pipelines.wan_video import PortraitAdapter
from utils import merge_audio_to_video


def find_replacement(a):
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0


def get_emo_feature(
    video_path, face_aligner, pd_fpg_motion, device=torch.device("cuda")
):
    pd_fpg_motion = pd_fpg_motion.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    ret, frame = cap.read()
    while ret:
        resized_frame = frame
        frame_list.append(resized_frame.copy())
        ret, frame = cap.read()
    cap.release()

    # Apply sliding window start
    start = max(0, int(args.start_frame)) if hasattr(args, "start_frame") else 0
    if start > 0:
        frame_list = frame_list[start:]

    num_frames = min(len(frame_list), args.num_frames)
    num_frames = find_replacement(num_frames)
    frame_list = frame_list[:num_frames]

    landmark_list = det_landmarks(face_aligner, frame_list)[1]
    emo_list = get_drive_expression_pd_fgc(
        pd_fpg_motion, frame_list, landmark_list, device
    )

    emo_feat_list = []
    head_emo_feat_list = []
    for emo in emo_list:
        headpose_emb = emo["headpose_emb"]
        eye_embed = emo["eye_embed"]
        emo_embed = emo["emo_embed"]
        mouth_feat = emo["mouth_feat"]

        emo_feat = torch.cat([eye_embed, emo_embed, mouth_feat], dim=1)
        head_emo_feat = torch.cat([headpose_emb, emo_feat], dim=1)

        emo_feat_list.append(emo_feat)
        head_emo_feat_list.append(head_emo_feat)

    emo_feat_all = torch.cat(emo_feat_list, dim=0)
    head_emo_feat_all = torch.cat(head_emo_feat_list, dim=0)

    return emo_feat_all, head_emo_feat_all, fps, num_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=False,
        help="prompt.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./output",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--portrait_scale",
        type=float,
        default=1.0,
        help="Image width.",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="The cfg of prompt.",
    )
    parser.add_argument(
        "--portrait_cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="The emo cfg.",
    )
    parser.add_argument(
        "--scale_image",
        type=bool,
        default=True,
        required=False,
        help="If scale the image.",
    )
    parser.add_argument(
        "--portrait_in_dim",
        type=int,
        default=768,
        help="The portrait in dim.",
    )
    parser.add_argument(
        "--portrait_proj_dim",
        type=int,
        default=2048,
        help="The portrait proj dim.",
    )
    parser.add_argument(
        "--portrait_checkpoint",
        type=str,
        default=None,
        required=True,
        help="The ckpt of FantasyPortrait",
    )
    parser.add_argument(
        "--alignment_model_path",
        type=str,
        default=None,
        required=True,
        help="The face landmark of pd-fgc.",
    )
    parser.add_argument(
        "--det_model_path",
        type=str,
        default=None,
        required=True,
        help="The det model of pd-fgc.",
    )
    parser.add_argument(
        "--pd_fpg_model_path",
        type=str,
        default=None,
        required=True,
        help="The motion model of pd-fgc.",
    )
    parser.add_argument(
        "--wan_model_path",
        type=str,
        default=None,
        required=True,
        help="The wan model path.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        required=False,
        help="The number of frames.",
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default=None,
        help="Path to precomputed features npz (head_emo_feat_all, fps).",
    )
    parser.add_argument(
        "--warm_video_path",
        type=str,
        default=None,
        help="Optional video to warm-start overlap from (previous segment tail).",
    )
    parser.add_argument(
        "--warm_overlap",
        type=int,
        default=0,
        help="Number of warm-start frames from warm_video_path to use at head.",
    )
    parser.add_argument(
        "--denoising_strength",
        type=float,
        default=1.0,
        help="Denoising strength when using warm video (0..1).",
    )
    parser.add_argument(
        "--noise_path",
        type=str,
        default=None,
        help="Optional path to precomputed global noise tensor (.pt) to slice per window.",
    )
    parser.add_argument(
        "--init_latents_path",
        type=str,
        default=None,
        help="Optional path to previous window's initial latents (.pt) for overlap handoff.",
    )
    parser.add_argument(
        "--init_latents_overlap",
        type=int,
        default=0,
        help="Number of head frames to override from init_latents_path.",
    )
    parser.add_argument(
        "--save_init_latents_path",
        type=str,
        default=None,
        help="If set, saves this window's initial latents (post add_noise) to the given .pt path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        required=False,
        help="The generative seed.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=720,
        help="The max size to scale.",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        default=None,
        required=True,
        help="The input image path.",
    )
    parser.add_argument(
        "--driven_video_path",
        type=str,
        default=None,
        required=True,
        help="The driven video path.",
    )
    parser.add_argument(
        "--no_audio_merge",
        action="store_true",
        help="Skip merging audio (useful for segmented generation).",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=0,
        required=False,
        help="Start frame index for sliding window (0-based).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        required=False,
        help="Override output FPS (e.g., 25).",
    )

    args = parser.parse_args()
    return args


args = parse_args()


def load_wan_video():
    # Load models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00001-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00002-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00003-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00004-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00005-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00006-of-00007.safetensors",
                ),
                os.path.join(
                    args.wan_model_path,
                    "diffusion_pytorch_model-00007-of-00007.safetensors",
                ),
            ],
            os.path.join(
                args.wan_model_path,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            os.path.join(args.wan_model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            os.path.join(args.wan_model_path, "Wan2.1_VAE.pth"),
        ],
        # torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )
    # Aggressively offload DiT weights to reduce resident VRAM. This slows down
    # compute but helps avoid OOM on smaller GPUs.
    pipe.enable_vram_management(
        num_persistent_param_in_dit=0
    )  # Keep 0 persistent params on GPU; spill others and on-demand load.
    return pipe


def load_pd_fgc_model():
    face_aligner = CameraDemo(
        face_alignment_module=FaceAlignment(
            gpu_id=None,
            alignment_model_path=args.alignment_model_path,
            det_model_path=args.det_model_path,
        ),
        reset=False,
    )

    pd_fpg_motion = FanEncoder()
    pd_fpg_checkpoint = torch.load(args.pd_fpg_model_path, map_location="cpu")
    m, u = pd_fpg_motion.load_state_dict(pd_fpg_checkpoint, strict=False)
    pd_fpg_motion = pd_fpg_motion.eval()

    return face_aligner, pd_fpg_motion


os.makedirs(args.output_path, exist_ok=True)
print(f"[infer] start_frame={args.start_frame} num_frames={args.num_frames} max_size={args.max_size}")

# Load models
pipe = load_wan_video()
face_aligner, pd_fpg_motion = load_pd_fgc_model()
device = torch.device("cuda")

portrait_model = PortraitAdapter(
    pipe.dit, args.portrait_in_dim, args.portrait_proj_dim
).to("cuda")
portrait_model.load_portrait_adapter(args.portrait_checkpoint, pipe.dit)
pipe.dit.to("cuda")
print(f"FantasyPortrait model load from checkpoint:{args.portrait_checkpoint}")

image = Image.open(args.input_image_path).convert("RGB")
width, height = image.size
if args.scale_image:
    scale = args.max_size / max(width, height)
    width, height = (int(width * scale), int(height * scale))
    image = image.resize([width, height], Image.LANCZOS)
print(f"[infer] resized image to {width}x{height}")

init_noise_slice = None
if args.noise_path is not None:
    # Load full noise and slice by latent indices for this window
    noise_full = torch.load(args.noise_path, map_location="cpu")  # [1,16,T_lat,H/8,W/8]
    start_lat = int(args.start_frame) // 4
    len_lat = (int(args.num_frames) - 1) // 4 + 1
    init_noise_slice = noise_full[:, :, start_lat : start_lat + len_lat]
    print(f"[infer] noise: full={tuple(noise_full.shape)} slice_start={start_lat} len_lat={len_lat} slice={tuple(init_noise_slice.shape)}")

init_latents_slice = None
if args.init_latents_path and args.init_latents_overlap > 0:
    init_lat_full = torch.load(args.init_latents_path, map_location="cpu")  # expected [1,16,T_lat,H/8,W/8]
    # Ensure 5D shape (B,C,T,H,W)
    if init_lat_full.dim() == 4:
        init_lat_full = init_lat_full.unsqueeze(0)
    # Convert overlap in frames to overlap in latent timesteps
    overlap_frames = int(args.init_latents_overlap)
    t_head_lat = (overlap_frames - 1) // 4 + 1
    init_latents_slice = init_lat_full[:, :, -t_head_lat:]
    print(f"[infer] init_latents: full={tuple(init_lat_full.shape)} overlap_frames={overlap_frames} t_head_lat={t_head_lat} slice={tuple(init_latents_slice.shape)}")

with torch.no_grad():
    if args.features_path is not None:
        import numpy as np
        data = np.load(args.features_path)
        head_emo_feat_all_np = data["head_emo_feat_all"]  # [T_saved, C]
        fps = float(data.get("fps", 25.0))
        saved_start = int(data.get("start_frame", 0))
        # Map absolute indices to the saved range
        abs_start = max(0, int(args.start_frame))
        rel_start = max(0, abs_start - saved_start)
        rel_end = rel_start + int(args.num_frames)
        # Bounds check
        if rel_start < 0 or rel_start >= head_emo_feat_all_np.shape[0]:
            raise ValueError(
                f"features_path does not cover requested start_frame: saved_start={saved_start}, "
                f"requested_start={abs_start}, available={head_emo_feat_all_np.shape[0]} frames"
            )
        rel_end = min(rel_end, head_emo_feat_all_np.shape[0])
        head_emo_feat_all = torch.from_numpy(head_emo_feat_all_np[rel_start:rel_end]).to(torch.float32)
        num_frames = head_emo_feat_all.shape[0]
    else:
        emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(
            args.driven_video_path, face_aligner, pd_fpg_motion
        )
head_emo_feat_all = head_emo_feat_all.unsqueeze(0)

adapter_proj = portrait_model.get_adapter_proj(head_emo_feat_all.to(device))
pos_idx_range = portrait_model.split_audio_adapter_sequence(
    adapter_proj.size(1), num_frames=num_frames
)
proj_split, context_lens = portrait_model.split_tensor_with_padding(
    adapter_proj, pos_idx_range, expand_length=0
)

negative_prompt = "人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
init_head_video = None
if args.warm_video_path and args.warm_overlap > 0:
    # Provide only the last overlap frames for latent warm-start (tail of previous segment)
    import cv2
    from PIL import Image as PILImage
    cap_w = cv2.VideoCapture(args.warm_video_path)
    warm_frames = []
    try:
        total = int(cap_w.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        start_idx = max(0, total - int(args.warm_overlap))
        if start_idx > 0:
            cap_w.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for _ in range(int(args.warm_overlap)):
            ok, fr = cap_w.read()
            if not ok:
                break
            warm_frames.append(PILImage.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    finally:
        cap_w.release()
    if warm_frames:
        init_head_video = warm_frames

result = pipe(
    prompt=args.prompt,
    negative_prompt=negative_prompt,
    input_image=image,
    init_head_video=init_head_video,
    return_latent_slice=bool(args.save_init_latents_path),
    latent_slice_count=(
        ((int(args.init_latents_overlap) - 1) // 4 + 1)
        if args.init_latents_overlap and int(args.init_latents_overlap) > 0
        else None
    ),
    width=width,
    height=height,
    num_frames=num_frames,
    num_inference_steps=30,
    seed=args.seed,
    tiled=True,
    ip_scale=args.portrait_scale,
    cfg_scale=args.cfg_scale,
    ip_cfg_scale=args.portrait_cfg_scale,
    adapter_proj=proj_split,
    adapter_context_lens=context_lens,
    latents_num_frames=(num_frames - 1) // 4 + 1,
    init_noise=init_noise_slice,
    init_latents=init_latents_slice,
)

# Unpack result (video + optional latent_slice)
if isinstance(result, dict):
    video_audio = result.get("video")
    latent_slice = result.get("latent_slice")
else:
    video_audio = result

# Save initial latents for this window (post add_noise at t0)
if args.save_init_latents_path:
    if 'latent_slice' in locals() and latent_slice is not None:
        torch.save(latent_slice, args.save_init_latents_path)

now = datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

image_name = args.input_image_path.split("/")[-1]
video_name = args.driven_video_path.split("/")[-1]

save_image_name = image_name + os.path.basename(args.input_image_path).split(".")[0][:8]
save_video_name = (
    video_name + os.path.basename(args.driven_video_path).split(".")[0][:8]
)
save_name = f"{timestamp_str}_{save_image_name}_{save_video_name}"
save_video_path = os.path.join(args.output_path, f"{save_name}.mp4")
out_fps = args.fps if hasattr(args, "fps") and args.fps is not None else fps
save_video(
    video_audio, os.path.join(args.output_path, f"{save_name}.mp4"), fps=out_fps, quality=5
)

# add Driven Audio to the Result video unless disabled for segment runs.
save_video_path_with_audio = os.path.join(
    args.output_path, f"{save_name}_with_audio.mp4"
)
if not hasattr(args, "no_audio_merge") or not args.no_audio_merge:
    merge_audio_to_video(
        args.driven_video_path, save_video_path, save_video_path_with_audio
    )
