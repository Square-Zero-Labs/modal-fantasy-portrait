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

with torch.no_grad():
    emo_feat_all, head_emo_feat_all, fps, num_frames = get_emo_feature(
        args.driven_video_path, face_aligner, pd_fpg_motion
    )
emo_feat_all, head_emo_feat_all = emo_feat_all.unsqueeze(
    0
), head_emo_feat_all.unsqueeze(0)

adapter_proj = portrait_model.get_adapter_proj(head_emo_feat_all.to(device))
pos_idx_range = portrait_model.split_audio_adapter_sequence(
    adapter_proj.size(1), num_frames=num_frames
)
proj_split, context_lens = portrait_model.split_tensor_with_padding(
    adapter_proj, pos_idx_range, expand_length=0
)

negative_prompt = "人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
video_audio = pipe(
    prompt=args.prompt,
    negative_prompt=negative_prompt,
    input_image=image,
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
)

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
save_video(
    video_audio, os.path.join(args.output_path, f"{save_name}.mp4"), fps=fps, quality=5
)

# add Driven Audio to the Result video.
save_video_path_with_audio = os.path.join(
    args.output_path, f"{save_name}_with_audio.mp4"
)
merge_audio_to_video(
    args.driven_video_path, save_video_path, save_video_path_with_audio
)
