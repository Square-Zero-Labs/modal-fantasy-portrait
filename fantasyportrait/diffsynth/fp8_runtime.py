import torch
import torch.nn as nn


def _fp8_linear_forward(cls: nn.Linear, base_dtype: torch.dtype, input: torch.Tensor):
    weight_dtype = cls.weight.dtype
    if weight_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Wan DiT linears operate on [B, L, D] tensors
        if input.dim() == 3:
            b, l, d = input.shape

            # Per-layer weight scale (for scaled fp8 exports). Defaults to 1.0
            scale_weight = getattr(cls, "scale_weight", None)
            if scale_weight is None:
                scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
            else:
                scale_weight = scale_weight.to(input.device, dtype=torch.float32)

            # Clamp and cast input to fp8 (always e4m3fn for A)
            scale_input = torch.ones((), device=input.device, dtype=torch.float32)
            inp = input.clamp(min=-448, max=448)
            a = inp.reshape(-1, d).to(torch.float8_e4m3fn).contiguous()

            bias = cls.bias.to(base_dtype) if cls.bias is not None else None

            out = torch._scaled_mm(
                a,
                cls.weight.t(),
                out_dtype=base_dtype,
                bias=bias,
                scale_a=scale_input,
                scale_b=scale_weight,
            )
            return out.reshape(b, l, cls.weight.shape[0])
        # Fallback: expect DiT always calls linears with 3D inputs
        return cls.__original_forward__(input)
    else:
        return cls.__original_forward__(input)


def convert_fp8_linear(module: nn.Module, base_dtype: torch.dtype, scale_weight_map: dict[str, torch.Tensor] | None = None):
    """
    Monkeypatch nn.Linear layers in `module` to use fp8 scaled matmul when weights are fp8.
    - Keeps parameters in float8 (saves VRAM) and computes in `base_dtype` (bf16 recommended).
    - If `scale_weight_map` contains "<module_name>.scale_weight", attaches it to the submodule.
    """
    if scale_weight_map is None:
        scale_weight_map = {}

    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            # Attach per-weight scale if provided
            scale_key = f"{name}.scale_weight"
            if scale_key in scale_weight_map:
                try:
                    setattr(sub, "scale_weight", scale_weight_map[scale_key])
                except Exception:
                    pass
            # Patch forward once
            if not hasattr(sub, "__original_forward__"):
                setattr(sub, "__original_forward__", sub.forward)
                sub.forward = lambda inp, m=sub: _fp8_linear_forward(m, base_dtype, inp)

