import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Lightweight LoRA wrapper for nn.Linear.

    Forward:
      y = base_linear(x) + lora_scale * (dropout(x) @ A^T @ B^T) * (lora_alpha / r)

    - base_linear parameters are frozen by default.
    - lora_scale can be set at inference time (e.g. 0 to disable LoRA).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        lora_init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"base_linear must be nn.Linear, got: {type(base_linear)}")
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got: {r}")

        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(self.r)

        # LoRA parameters: A: [r, in], B: [out, r]
        # Initialize A with small random weights and B to zeros so initial delta ~= 0.
        self.lora_A = nn.Parameter(torch.randn(self.r, self.in_features) * lora_init_std)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r))

        self.lora_dropout = nn.Dropout(p=float(lora_dropout)) if lora_dropout and lora_dropout > 0 else nn.Identity()

        # Inference-time multiplier; can be set by set_lora_scale().
        self._lora_scale: float = 1.0

        # Freeze the original linear weights/bias.
        self.base_linear.weight.requires_grad_(False)
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad_(False)

    @property
    def lora_scale(self) -> float:
        return float(self._lora_scale)

    def set_lora_scale(self, scale: float) -> None:
        self._lora_scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base output
        result = self.base_linear(x)

        # LoRA delta
        if self._lora_scale == 0.0:
            return result

        x_d = self.lora_dropout(x)
        # [..., in] -> [..., r] -> [..., out]
        lora_mid = F.linear(x_d, self.lora_A)  # weight: [r, in]
        lora_out = F.linear(lora_mid, self.lora_B)  # weight: [out, r]
        return result + (lora_out * (self.scaling * self._lora_scale)).to(result.dtype)


def _matches_target_name(child_name: str, target_names: Sequence[str]) -> bool:
    # We match by exact attribute name (common in this codebase: qkv, proj, q_linear, kv_linear, fc1, fc2).
    # Additionally, allow suffix matching to be resilient to refactors.
    return any(child_name == t or child_name.endswith(f".{t}") for t in target_names)


def inject_lora_into_module(
    module: nn.Module,
    target_module_names: Sequence[str],
    r: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
    *,
    freeze_non_lora_params: bool = True,
) -> List[Tuple[str, LoRALinear]]:
    """
    Replace nn.Linear layers inside `module` (recursively) whose attribute name matches
    any item in `target_module_names` with LoRALinear.

    This function ONLY touches layers under the provided `module` object; it does not
    traverse outside its scope.

    Returns a list of (qualified_name, LoRALinear) for injected layers.
    """

    if not isinstance(module, nn.Module):
        raise TypeError(f"module must be nn.Module, got {type(module)}")

    injected: List[Tuple[str, LoRALinear]] = []

    def _inject_rec(parent: nn.Module, prefix: str) -> None:
        for child_name, child in parent.named_children():
            qn = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear) and _matches_target_name(child_name, target_module_names):
                lora = LoRALinear(
                    base_linear=child,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                setattr(parent, child_name, lora)
                injected.append((qn, lora))
            else:
                _inject_rec(child, qn)

    _inject_rec(module, "")

    if freeze_non_lora_params:
        # Freeze everything, then unfreeze only LoRA params inside injected LoRALinear.
        for p in module.parameters():
            p.requires_grad_(False)
        for _, l in injected:
            for p in l.parameters():
                p.requires_grad_(True)

    return injected


def set_lora_scale(module: nn.Module, scale: float) -> None:
    """Set lora_scale for all LoRALinear layers under `module`."""
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.set_lora_scale(scale)


def get_lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Export only LoRA parameters (lora_A/lora_B) for adapter saving.
    """
    out: Dict[str, torch.Tensor] = {}
    for name, p in module.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            out[name] = p.detach().cpu()
    return out


def load_lora_state_dict(
    module: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Load LoRA parameters into existing LoRALinear modules.

    NOTE:
    - You must call inject_lora_into_module(...) before this, otherwise keys won't match.
    - strict=False matches extra/missing keys by default.
    """
    missing: List[str] = []
    unexpected: List[str] = []

    # Build current parameter map
    cur = dict(module.named_parameters())

    for k, v in lora_state_dict.items():
        if k not in cur:
            unexpected.append(k)
            continue
        if cur[k].shape != v.shape:
            raise ValueError(f"LoRA param shape mismatch for {k}: cur={cur[k].shape}, ckpt={v.shape}")
        cur[k].data.copy_(v.to(cur[k].device, dtype=cur[k].dtype))

    if strict:
        for k in cur.keys():
            if k.endswith("lora_A") or k.endswith("lora_B"):
                if k not in lora_state_dict:
                    missing.append(k)

    return missing, unexpected

