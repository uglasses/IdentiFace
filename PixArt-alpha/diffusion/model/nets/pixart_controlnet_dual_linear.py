"""
Dual ControlNet with linear sum of skip connections: two independently trained
controlnet stacks (e.g. edges + LQ checkpoints) share the same base DiT; each
layer adds w_e * skip_e + w_l * skip_l to the main stream.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch
from torch import Tensor
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.nets.PixArtMS import PixArtMS
from diffusion.model.utils import auto_grad_checkpoint

from .pixart_controlnet import ControlPixArtHalf, ControlPixArtMSHalf


class ControlPixArtMSHalfDualLinearSum(ControlPixArtMSHalf):
    """
    Two parallel ``controlnet`` copies (``controlnet`` = first branch, ``controlnet_lq`` = second).
    ``c`` = edges (or first modality) VAE latent; ``c2`` = LQ VAE latent. No latent fusion module:
    each branch embeds its own latent with ``forward_c`` and runs its own ControlNet stack; skips are
    combined with ``dual_linear_w_edges`` and ``dual_linear_w_lq``.
    """

    def __init__(
        self,
        base_model: PixArtMS,
        copy_blocks_num: int = 13,
    ) -> None:
        super().__init__(
            base_model=base_model,
            copy_blocks_num=copy_blocks_num,
            use_vae_dual_fusion=False,
            use_dual_nonlinear_control=False,
            use_frequency_control_fusion=False,
        )
        self.controlnet_lq = copy.deepcopy(self.controlnet)
        self.dual_linear_w_edges: float = 1.0
        self.dual_linear_w_lq: float = 1.0

    def __getattr__(self, name: str):
        # ControlPixArtHalf.__getattr__ only whitelists ``controlnet`` for nn.Module lookup and
        # forwards every other unknown name to ``base_model``, which breaks ``controlnet_lq``.
        if name == "controlnet_lq":
            return super(ControlPixArtHalf, self).__getattr__(name)
        return super().__getattr__(name)

    def forward(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        data_info: Optional[Any] = None,
        c: Optional[Tensor] = None,
        c2: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        we = float(kwargs.get("dual_linear_w_edges", self.dual_linear_w_edges))
        wl = float(kwargs.get("dual_linear_w_lq", self.dual_linear_w_lq))
        kwargs_net = {
            k: v
            for k, v in kwargs.items()
            if k not in ("dual_linear_w_edges", "dual_linear_w_lq")
        }

        if c is None or c2 is None:
            raise ValueError(
                "ControlPixArtMSHalfDualLinearSum requires both c (edges) and c2 (LQ) VAE latents."
            )

        c = c.to(self.dtype)
        c2 = c2.to(self.dtype)
        c_e = self.forward_c(c)
        c_l = self.forward_c(c2)

        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        assert data_info is not None
        c_size, ar = data_info["img_hw"].to(self.dtype), data_info["aspect_ratio"].to(self.dtype)
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size

        pos_embed = (
            torch.from_numpy(
                get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1],
                    (self.h, self.w),
                    lewei_scale=self.lewei_scale,
                    base_size=self.base_size,
                )
            )
            .unsqueeze(0)
            .to(x.device)
            .to(self.dtype)
        )
        x = self.x_embedder(x) + pos_embed
        bs = x.shape[0]
        t = self.t_embedder(timestep)
        csize = self.csize_embedder(c_size, bs)
        ar_emb = self.ar_embedder(ar, bs)
        t = t + torch.cat([csize, ar_emb], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        x = auto_grad_checkpoint(
            self.base_model.blocks[0], x, y, t0, y_lens, **kwargs_net
        )

        for index in range(1, self.copy_blocks_num + 1):
            c_e, skip_e = auto_grad_checkpoint(
                self.controlnet[index - 1],
                x,
                y,
                t0,
                y_lens,
                c_e,
                **kwargs_net,
            )
            c_l, skip_l = auto_grad_checkpoint(
                self.controlnet_lq[index - 1],
                x,
                y,
                t0,
                y_lens,
                c_l,
                **kwargs_net,
            )
            c_skip = we * skip_e + wl * skip_l
            x = auto_grad_checkpoint(
                self.base_model.blocks[index],
                x + c_skip,
                y,
                t0,
                y_lens,
                **kwargs_net,
            )

        for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
            x = auto_grad_checkpoint(
                self.base_model.blocks[index], x, y, t0, y_lens, **kwargs_net
            )

        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        return x

    def forward_with_dpmsolver(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        data_info: Any,
        c: Tensor,
        c2: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        model_out = self.forward(
            x,
            t,
            y,
            data_info=data_info,
            c=c,
            c2=c2,
            **kwargs,
        )
        return model_out.chunk(2, dim=1)[0]
