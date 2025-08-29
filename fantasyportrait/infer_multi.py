import argparse
import csv
import math
import os
import random
import sys
from datetime import datetime
from uuid import uuid4

import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers.utils import load_image
from moviepy import *
from PIL import Image

from diffsynth import ModelManager, WanVideoPipeline, save_video
from diffsynth.models.camer import CameraDemo
from diffsynth.models.face_align import FaceAlignment
from diffsynth.models.pdf import (FanEncoder, det_landmarks,
                                  get_drive_expression_pd_fgc)
from diffsynth.pipelines.wan_video import PortraitAdapter
from utils import (change_video_fps, draw_rects_on_frames,
                   extract_faces_from_video)


def find_replacement(a):
    while a > 0:
        if (a - 1) % 4 == 0:
            return a
        a -= 1
    return 0


def resize_mask(mask):
    """
    Downsample the mask both temporally and spatially to match the size of the video compressed by VAE.
    - The first frame is only downsampled spatially, not temporally.
    - Other frames are downsampled using max pooling for both temporal and spatial dimensions.

    Args:
        mask (torch.Tensor): Input mask with shape [f, h, w].

    Returns:
        torch.Tensor: Downsampled mask with shape [1 + f//4, h//16, w//16, 1].
    """
    f, h, w = mask.shape

    first_frame = mask[0].unsqueeze(0).unsqueeze(0)
    first_frame = F.max_pool2d(first_frame, kernel_size=16, stride=16)
    first_frame = first_frame.squeeze(0).squeeze(0)

    mask_rest = mask[1:].unsqueeze(0).unsqueeze(0)
    mask_rest = F.max_pool3d(mask_rest, kernel_size=(4, 16, 16), stride=(4, 16, 16))
    mask_rest = mask_rest.squeeze(0).squeeze(0)
    mask_resized = torch.cat([first_frame.unsqueeze(0), mask_rest], dim=0)

    return mask_resized


def build_attn_mask(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: shape [B, L1]
    b: shape [B, L2]
    return: attn_mask, shape [B, L1, L2], dtype=torch.bool or torch.int
    """
    attn_mask = a.unsqueeze(2) == b.unsqueeze(1)
    return ~attn_mask


def compute_max_xy(rect_list):
    x1_ref, y1_ref, x2_ref, y2_ref = rect_list[0]
    cx_ref = (x1_ref + x2_ref) / 2
    cy_ref = (y1_ref + y2_ref) / 2
    width_ref = x2_ref - x1_ref
    height_ref = y2_ref - y1_ref

    rect_array = np.array(rect_list)
    max_values = np.max(rect_array, axis=0)
    min_values = np.min(rect_array, axis=0)

    max_x1, max_y1, max_x2, max_y2 = max_values
    min_x1, min_y1, min_x2, min_y2 = min_values

    max_rect = [min_x1, min_y1, max_x2, max_y2]

    max_left_move = (x1_ref - min_x1) / width_ref
    max_right_move = (max_x2 - x2_ref) / width_ref
    max_up_move = (y1_ref - min_y1) / height_ref
    max_down_move = (max_y2 - y2_ref) / height_ref
    return (
        min(max_left_move, 0.25),
        min(max_right_move, 0.25),
        min(max_up_move, 0.25),
        min(max_down_move, 0.25),
    )


def create_mask(image, bounding_boxes, proj_split, video_rect_list):
    """
    Computes the maximum range of facial movements from the face regions in the driving video to obtain the corresponding face mask for the reference image.

    Args:
        image (Image): The reference image.
        bounding_boxes (List): The face bounding boxes in the reference image.
        proj_split (Tensor): The facial expression motion feature sequence, with shape [1, f, L2, C].
        video_rect_list (List[List]): The face detection bounding boxes from the driving video, used to calculate the maximum mask range for the corresponding face in the reference image.

    Returns:
        adapter_attn_mask (Tensor): The attention mask for cross-attention between the internal video tokens and adapter tokens in WAN, with shape [f, L1, L2].
    """
    width, height = image.size
    bounding_boxes = sorted(bounding_boxes, key=lambda box: (box[0] + box[2]) / 2)

    num_faces = len(bounding_boxes)
    assert proj_split.shape[2] % num_faces == 0

    mask = torch.zeros((proj_split.size(1) - 1) * 4 + 1, height, width, device="cuda")

    adapter_mask = torch.zeros(proj_split.squeeze(0).shape[:-1], device="cuda")
    f, l = adapter_mask.shape

    extend_bounding_boxes = []

    for i, (face_rect, video_rect) in enumerate(zip(bounding_boxes, video_rect_list)):
        max_left_move, max_right_move, max_up_move, max_down_move = compute_max_xy(
            video_rect
        )

        x1, y1, x2, y2 = face_rect
        width_face, height_face = int(x2 - x1), int(y2 - y1)
        x1 -= width_face * max_left_move
        x2 += width_face * max_right_move
        y1 -= height_face * max_up_move
        y2 += height_face * max_down_move

        x1 = max(0, int(x1))
        x2 = min(width, int(x2))
        y1 = max(0, int(y1))
        y2 = min(height, int(y2))

        extend_bounding_boxes.append([x1, y1, x2, y2])

        mask[:, y1:y2, x1:x2] = i + 1

        adapter_face_index_begin = (l // num_faces) * i
        adapter_face_index_end = (l // num_faces) * (i + 1)

        adapter_mask[:, adapter_face_index_begin:adapter_face_index_end] = i + 1

    mask_latents = resize_mask(mask)
    _, l_h, l_w = mask_latents.shape[:3]

    mask_latents = mask_latents.view(f, -1)
    attn_mask = build_attn_mask(mask_latents, adapter_mask)

    return attn_mask, extend_bounding_boxes


def get_emo_feature(
    video_path, face_aligner, pd_fpg_motion, device=torch.device("cuda")
):
    pd_fpg_motion = pd_fpg_motion.to(device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame.copy())
        ret, frame = cap.read()
    cap.release()

    num_frames = min(len(frame_list), args.num_frames)
    num_frames = find_replacement(num_frames)
    print("The number of frames is ", num_frames)
    frame_list = frame_list[:num_frames]

    landmark_list, rect_list = det_landmarks(face_aligner, frame_list)[1:]

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

    return emo_feat_all, head_emo_feat_all, fps, num_frames, rect_list, frame_list


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

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
        nargs="+",
        required=True,
        help="List of driven video paths.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
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
        help="Portrait condition injection weight.",
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
        default=4.0,
        required=False,
        help="The cfg of portrait condition.",
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
        "--scale_image",
        type=bool,
        default=True,
        required=False,
        help="If scale the image.",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=720,
        help="The max size to scale.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        required=False,
        help="The fps of the generated video",
    )
    args = parser.parse_args()
    return args


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
    pipe.enable_vram_management(
        num_persistent_param_in_dit=None
    )  # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    return pipe


def process_multi_video(portrait_model, face_aligner, pd_fpg_motion, image, video_list):
    """
    Args:
        image (PIL.Image): The reference image.
        video_list (List[str]): A list of file paths to the driving videos.

    Returns:
        proj_split (Tensor): The concatenated expression feature sequence, with shape [1, f, l*b(L2), c].
        adapter_attn_mask (Tensor): The attention mask for cross-attention between the internal video tokens and adapter tokens in WAN, with shape [f, L1, L2].
        extend_bounding_boxes (List[List[float]]): The extended bounding box regions [[x1, x2, y1, y2]] corresponding to the face in the generated video.
    """

    width, height = image.size
    bounding_boxes, _, score = face_aligner.face_alignment_module.face_detector.detect(
        np.array(image)[:, :, ::-1]
    )
    num_faces = len(bounding_boxes)

    if num_faces <= 1:
        raise ValueError(f"the image face number {num_faces} is lower than 1.")
    if len(video_list) > num_faces:
        video_list = video_list[:num_faces]

    if len(video_list) != num_faces:
        raise ValueError(
            f"the length of video_list {len(video_list)} is not equal to num_faces {num_faces}!"
        )

    face_motion_feat = []
    num_frames_list = []
    video_rect_list = []
    frame_list_list = []
    with torch.no_grad():
        for video_path in video_list:
            (
                emo_feat_all,
                head_emo_feat_all,
                fps,
                num_frames,
                rect_list,
                frame_list,
            ) = get_emo_feature(video_path, face_aligner, pd_fpg_motion)
            face_motion_feat.append(head_emo_feat_all)
            num_frames_list.append(num_frames)
            video_rect_list.append(rect_list)
            frame_list_list.append(frame_list)

    num_frames = min(num_frames_list)
    face_motion_feat = [i[:num_frames, :].unsqueeze(0) for i in face_motion_feat]
    video_rect_list = [i[:num_frames] for i in video_rect_list]

    proj_split = []
    for face_motion_feat_ in face_motion_feat:
        adapter_proj = portrait_model.get_adapter_proj(face_motion_feat_.to("cuda"))
        pos_idx_range = portrait_model.split_audio_adapter_sequence(
            adapter_proj.size(1), num_frames=num_frames
        )
        proj_split_, adapter_context_lens = portrait_model.split_tensor_with_padding(
            adapter_proj, pos_idx_range, expand_length=0
        )
        proj_split.append(proj_split_)

    proj_split = torch.cat(proj_split[::-1], dim=-2)

    adapter_attn_mask, extend_bounding_boxes = create_mask(
        image, bounding_boxes, proj_split, video_rect_list
    )

    return (
        proj_split,
        adapter_attn_mask,
        extend_bounding_boxes,
        frame_list_list,
        num_frames,
        video_list,
    )


def main(args):
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    image_name = os.path.splitext(os.path.basename(args.input_image_path))[0]
    video_name = "_".join(
        os.path.splitext(os.path.basename(video_path))[0]
        for video_path in args.driven_video_path
    )
    args.output_path = os.path.join(
        args.output_path, f"{timestamp_str}_{image_name}_{video_name}"
    )
    os.makedirs(args.output_path, exist_ok=True)
    driven_video_dir = os.path.join(args.output_path, "driven_video")
    os.makedirs(driven_video_dir, exist_ok=True)

    driven_video_path = []
    for video_path in args.driven_video_path:
        output_driven_video_path = os.path.join(
            driven_video_dir, os.path.basename(video_path)
        )
        change_video_fps(video_path, output_driven_video_path, args.fps)
        driven_video_path.append(output_driven_video_path)
    args.driven_video_path = driven_video_path

    # Load models
    pipe = load_wan_video()
    face_aligner, pd_fpg_motion = load_pd_fgc_model()

    portrait_model = PortraitAdapter(
        pipe.dit, args.portrait_in_dim, args.portrait_proj_dim
    ).to("cuda")
    portrait_model.load_portrait_adapter(args.portrait_checkpoint, pipe.dit)
    pipe.dit.to("cuda")

    image = Image.open(args.input_image_path).convert("RGB")
    width, height = image.size
    if args.scale_image:
        scale = args.max_size / max(width, height)
        width, height = (int(width * scale), int(height * scale))
        image = image.resize([width, height], Image.LANCZOS)
    height, width = pipe.check_resize_height_width(height, width)
    image = image.resize([width, height], Image.LANCZOS)

    if len(args.driven_video_path) == 1:
        tmp_dir = f"{args.output_path}/face_crop"
        print(args.driven_video_path)

        extract_faces_from_video(args.driven_video_path[0], tmp_dir, face_aligner)
        args.driven_video_path = [
            os.path.join(tmp_dir, p) for p in os.listdir(tmp_dir) if p.endswith(".mp4")
        ]

    (
        proj_split,
        adapter_attn_mask,
        extend_bounding_boxes,
        frame_list_list,
        num_frames,
        video_list_sample,
    ) = process_multi_video(
        portrait_model, face_aligner, pd_fpg_motion, image, args.driven_video_path
    )

    negative_prompt = "人物嘴巴不停地说话，人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
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
        adapter_context_lens=None,
        latents_num_frames=(num_frames - 1) // 4 + 1,
        adapter_attn_mask=adapter_attn_mask,
    )

    save_video(
        video_audio,
        os.path.join(args.output_path, f"output.mp4"),
        fps=args.fps,
        quality=5,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
