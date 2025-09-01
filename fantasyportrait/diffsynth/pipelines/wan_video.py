import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from transformers import (AutoImageProcessor, AutoModel, SiglipImageProcessor,
                          SiglipVisionModel)

from ..models import ModelManager
from ..models.wan_video_dit import (WanLayerNorm, WanModel, WanRMSNorm,
                                    attention, flash_attention)
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_text_encoder import (T5LayerNorm, T5RelativeEmbedding,
                                             WanTextEncoder)
from ..models.wan_video_vae import (CausalConv3d, RMS_norm, Upsample,
                                    WanVideoVAE)
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import (AutoWrappedLinear, AutoWrappedModule,
                               enable_vram_management)
from .base import BasePipeline


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x


class MultiProjModel(nn.Module):
    def __init__(self, adapter_in_dim=1024, cross_attention_dim=1024):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.eye_proj = torch.nn.Linear(6, cross_attention_dim, bias=False)
        self.emo_proj = torch.nn.Linear(30, cross_attention_dim, bias=False)
        self.mouth_proj = torch.nn.Linear(512, cross_attention_dim, bias=False)
        self.headpose_proj = torch.nn.Linear(6, cross_attention_dim, bias=False)

        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, adapter_embeds):
        B, num_frames, C = adapter_embeds.shape
        embeds = adapter_embeds
        split_sizes = [6, 6, 30, 512]
        headpose, eye, emo, mouth = torch.split(embeds, split_sizes, dim=-1)
        headpose = self.norm(self.headpose_proj(headpose))
        eye = self.norm(self.eye_proj(eye))
        emo = self.norm(self.emo_proj(emo))
        mouth = self.norm(self.mouth_proj(mouth))

        all_features = torch.stack([headpose, eye, emo, mouth], dim=2)
        result_final = all_features.view(B, num_frames * 4, self.cross_attention_dim)

        return result_final


class SingleStreamBlockProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        self.ip_adapter_single_stream_k_proj = nn.Linear(
            context_dim, hidden_dim, bias=False
        )
        self.ip_adapter_single_stream_v_proj = nn.Linear(
            context_dim, hidden_dim, bias=False
        )

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        adapter_proj: torch.Tensor,
        adapter_context_lens: torch.Tensor,
        latents_num_frames: int = 21,
        ip_scale: float = 1.0,
        adapter_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), attn.num_heads, attn.head_dim

        # compute query, key, value
        q = attn.norm_q(attn.q(x)).view(b, -1, n, d)
        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)
        v = attn.v(context).view(b, -1, n, d)
        k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)
        v_img = attn.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention with flash attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        x = x.flatten(2)
        img_x = img_x.flatten(2)

        if len(adapter_proj.shape) == 4:
            adapter_q = q.view(b * latents_num_frames, -1, n, d)
            ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(
                b * latents_num_frames, -1, n, d
            )
            ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(
                b * latents_num_frames, -1, n, d
            )
            adapter_x = attention(
                adapter_q, ip_key, ip_value, attn_mask=adapter_attn_mask
            )
            adapter_x = adapter_x.view(b, q.size(1), n, d)
            adapter_x = adapter_x.flatten(2)
        elif len(adapter_proj.shape) == 3:
            ip_key = self.ip_adapter_single_stream_k_proj(adapter_proj).view(
                b, -1, n, d
            )
            ip_value = self.ip_adapter_single_stream_v_proj(adapter_proj).view(
                b, -1, n, d
            )
            adapter_x = attention(q, ip_key, ip_value, attn_mask=adapter_attn_mask)
            adapter_x = adapter_x.flatten(2)

        x = x + img_x + adapter_x * ip_scale
        x = attn.o(x)
        return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):  # x  (b, 512, 1)
        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)  # (b, 512, 1024)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents  # b 16 1024
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class PortraitAdapter(nn.Module):
    def __init__(self, wan_dit: WanModel, adapter_in_dim: int, adapter_proj_dim: int):
        super().__init__()

        self.adapter_in_dim = adapter_in_dim
        self.adapter_proj_dim = adapter_proj_dim
        self.proj_model = self.init_proj(self.adapter_proj_dim)

        self.mouth_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=16,
            embedding_dim=512,
            output_dim=2048,
            ff_mult=4,
        )

        self.emo_proj_model = Resampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=4,
            embedding_dim=30,
            output_dim=2048,
            ff_mult=4,
        )

        self.set_portrait_adapter(wan_dit)

    def init_proj(self, cross_attention_dim=5120):
        proj_model = MultiProjModel(
            adapter_in_dim=self.adapter_in_dim, cross_attention_dim=cross_attention_dim
        )
        return proj_model

    def set_portrait_adapter(self, wan_dit):
        attn_procs = {}
        for name in wan_dit.attn_processors.keys():
            attn_procs[name] = SingleStreamBlockProcessor(
                context_dim=self.adapter_proj_dim, hidden_dim=wan_dit.dim
            )
        wan_dit.set_attn_processor(attn_procs)
        print("set_attn_processor.........")
        for name in wan_dit.attn_processors.keys():
            print(f"{name}: {wan_dit.attn_processors[name]}")

    def load_portrait_adapter(self, ip_ckpt: str, wan_dit):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"proj_model": {}, "ip_adapter": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model."):
                        state_dict["proj_model"][
                            key.replace("proj_model.", "")
                        ] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][
                            key.replace("ip_adapter.", "")
                        ] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")
        self.proj_model.load_state_dict(state_dict["proj_model"])
        self.mouth_proj_model.load_state_dict(state_dict["mouth_proj_model"])
        self.emo_proj_model.load_state_dict(state_dict["emo_proj_model"])
        wan_dit.load_state_dict(state_dict["ip_adapter"], strict=False)

    def get_adapter_proj(self, adapter_fea=None):
        split_sizes = [6, 6, 30, 512]
        headpose, eye, emo, mouth = torch.split(
            adapter_fea, split_sizes, dim=-1
        )
        B, frames, dim = mouth.shape
        mouth = mouth.view(B * frames, 1, 512)
        emo = emo.view(B * frames, 1, 30)

        mouth_fea = self.mouth_proj_model(mouth)
        emo_fea = self.emo_proj_model(emo)

        mouth_fea = mouth_fea.view(B, frames, 16, 2048)
        emo_fea = emo_fea.view(B, frames, 4, 2048)

        adapter_fea = self.proj_model(adapter_fea)

        adapter_fea = adapter_fea.view(B, frames, 4, 2048)

        all_fea = torch.cat([adapter_fea, mouth_fea, emo_fea], dim=2)

        result_final = all_fea.view(B, frames * 24, 2048)

        return result_final

    def split_audio_adapter_sequence(self, adapter_proj_length, num_frames=80):
        tokens_pre_frame = adapter_proj_length / num_frames
        tokens_pre_latents_frame = tokens_pre_frame * 4
        half_tokens_pre_latents_frame = tokens_pre_latents_frame / 2
        pos_idx = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_idx.append(0)
            else:
                begin_token_id = tokens_pre_frame * ((i - 1) * 4 + 1)
                end_token_id = tokens_pre_frame * (i * 4 + 1)
                pos_idx.append(int((sum([begin_token_id, end_token_id]) / 2)) - 1)
        pos_idx_range = [
            [
                idx - int(half_tokens_pre_latents_frame),
                idx + int(half_tokens_pre_latents_frame),
            ]
            for idx in pos_idx
        ]
        pos_idx_range[0] = [
            -(int(half_tokens_pre_latents_frame) * 2 - pos_idx_range[1][0]),
            pos_idx_range[1][0],
        ]
        return pos_idx_range

    def split_tensor_with_padding(self, input_tensor, pos_idx_range, expand_length=0):
        pos_idx_range = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_range
        ]
        sub_sequences = []
        seq_len = input_tensor.size(1)
        max_valid_idx = seq_len - 1
        k_lens_list = []
        for start, end in pos_idx_range:
            pad_front = max(-start, 0)
            pad_back = max(end - max_valid_idx, 0)

            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]
            y = self.vae.encode(
                [
                    torch.concat(
                        [
                            image.transpose(0, 1),
                            torch.zeros(3, num_frames - 1, height, width).to(
                                image.device
                            ),
                        ],
                        dim=1,
                    )
                ],
                device=self.device,
            )[0]
            y = torch.concat([msk, y])
        return {"clip_fea": clip_context, "y": [y]}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            latents = self.vae.encode(
                input_video,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            frames = self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return frames

    def set_ip(self, local_path):
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        ip_cfg_scale=None,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        return_tensor=False,
        **kwargs,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=sigma_shift
        )

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=noise.dtype, device=noise.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            for progress_id, timestep in enumerate(
                progress_bar_cmd(self.scheduler.timesteps)
            ):
                timestep = timestep.unsqueeze(0).to(
                    dtype=torch.float32, device=self.device
                )

                # Inference
                noise_pred_posi = self.dit(
                    latents,
                    timestep=timestep,
                    **prompt_emb_posi,
                    **image_emb,
                    **extra_input,
                    **kwargs,
                )
                if ip_cfg_scale and ip_cfg_scale > 1.0:
                    ip_scale = kwargs["ip_scale"]
                    kwargs["ip_scale"] = 0.0
                    noise_pred_noaudio = self.dit(
                        latents,
                        timestep=timestep,
                        **prompt_emb_posi,
                        **image_emb,
                        **extra_input,
                        **kwargs,
                    )
                    if cfg_scale != 1.0:
                        noise_pred_no_cond = self.dit(
                            latents,
                            timestep=timestep,
                            **prompt_emb_nega,
                            **image_emb,
                            **extra_input,
                            **kwargs,
                        )
                        noise_pred = (
                            noise_pred_no_cond
                            + cfg_scale * (noise_pred_noaudio - noise_pred_no_cond)
                            + ip_cfg_scale * (noise_pred_posi - noise_pred_noaudio)
                        )
                    else:
                        noise_pred = noise_pred_noaudio + ip_cfg_scale * (
                            noise_pred_posi - noise_pred_noaudio
                        )
                    kwargs["ip_scale"] = ip_scale
                else:
                    if cfg_scale != 1.0:
                        noise_pred_nega = self.dit(
                            latents,
                            timestep=timestep,
                            **prompt_emb_nega,
                            **image_emb,
                            **extra_input,
                            **kwargs,
                        )
                        noise_pred = noise_pred_nega + cfg_scale * (
                            noise_pred_posi - noise_pred_nega
                        )
                    else:
                        noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], latents
                )

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        return_frames = self.tensor2video(frames[0])

        if return_tensor:
            return return_frames, frames

        return return_frames
