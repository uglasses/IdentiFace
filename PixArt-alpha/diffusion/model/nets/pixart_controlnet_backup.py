import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping, Optional

from diffusion.model.nets import PixArtMSBlock, PixArtMS, PixArt
from diffusion.model.nets.PixArt import get_2d_sincos_pos_embed
from diffusion.model.utils import auto_grad_checkpoint


class VaeLatentFusion3x3(nn.Module):
    """
    Fuse two 4-channel VAE latents (e.g. LQ + sketch/edges) into one 4-channel control latent.

    Stack: Conv 8->mid, GN+SiLU, Conv mid->mid, GN+SiLU, Conv mid->4 (delta).
    Output: c2 + delta (residual on the second modality — typically structure/edges — for stable optimization).
    Last conv is zero-initialized so training starts near the second-branch latent.
    """

    def __init__(self, mid_channels: int = 32, num_groups: int = 8) -> None:
        super().__init__()
        assert mid_channels % num_groups == 0, "mid_channels must be divisible by num_groups"
        # Delta branch: predict residual delta from concatenated [c1,c2]
        self.delta_net = nn.Sequential(
            nn.Conv2d(8, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, mid_channels),
            nn.SiLU(),
            nn.Conv2d(mid_channels, 4, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)
        # Gate branch: predict per-pixel gate g in [0,1]
        self.gate_net = nn.Sequential(
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.gate_net[0].weight)
        nn.init.zeros_(self.gate_net[0].bias)
        # Timestep weighting parameters: w(t) = sigmoid(a * t_hat + b), t_hat in [0,1] (late steps -> larger)
        self.w_a = nn.Parameter(torch.tensor(2.0))
        self.w_b = nn.Parameter(torch.tensor(-1.0))

    def forward(self, c1: Tensor, c2: Tensor, timestep: Optional[Tensor] = None) -> Tensor:
        """
        Gated residual fusion with timestep-aware weighting:
          delta = Delta([c1,c2])
          g = sigmoid(Gate([c1,c2]))        # [N,1,H,W]
          w = sigmoid(a * t_hat + b)        # [N,1,1,1], t_hat increases towards later steps
          fused = c1 + (w * g) * delta
        """
        x = torch.cat([c1, c2], dim=1)
        delta = self.delta_net(x)
        g = torch.sigmoid(self.gate_net(x))  # [N,1,H,W]
        if timestep is not None:
            # Normalize to [0,1] within batch, then invert so late steps get larger weights
            t = timestep.to(dtype=c1.dtype).view(-1, 1, 1, 1)
            t_min = t.amin(dim=0, keepdim=True)
            t_max = t.amax(dim=0, keepdim=True)
            denom = torch.clamp(t_max - t_min, min=1e-6)
            t_norm = (t - t_min) / denom
            t_hat = 1.0 - t_norm
            w = torch.sigmoid(self.w_a.to(c1.dtype) * t_hat + self.w_b.to(c1.dtype))
        else:
            # Default moderate weight if timestep not provided
            w = torch.sigmoid(self.w_b.to(c1.dtype)).view(1, 1, 1, 1)
        fused = c1 + (w * g) * delta
        return fused.to(c1.dtype)


class DualNonlinearControlFusion(nn.Module):
    """
    Two independent residual transforms on the two control latents, then a convex blend.

    Convention (matches dataset / train_controlnet): ``c1`` = edges (``condition``), ``c2`` = LQ (``condition2``).

      r1 = Net1(c1), r2 = Net2(c2)   # separate conv stacks, last layer zero-init
      z1 = c1 + r1, z2 = c2 + r2     # residual branches (stable at initialization)
      fused = (1 - α) * z1 + α * z2  with α = sigmoid(alpha_logit) trainable scalar

    This replaces :class:`VaeLatentFusion3x3` when ``use_dual_nonlinear_control=True``; mutually exclusive in the wrapper.
    """

    def __init__(self, mid_channels: int = 32, num_groups: int = 8) -> None:
        super().__init__()
        assert mid_channels % num_groups == 0, "mid_channels must be divisible by num_groups"

        def branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(4, mid_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, mid_channels),
                nn.SiLU(),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, mid_channels),
                nn.SiLU(),
                nn.Conv2d(mid_channels, 4, kernel_size=3, padding=1),
            )

        self.net_c1 = branch()
        self.net_c2 = branch()
        nn.init.zeros_(self.net_c1[-1].weight)
        nn.init.zeros_(self.net_c1[-1].bias)
        nn.init.zeros_(self.net_c2[-1].weight)
        nn.init.zeros_(self.net_c2[-1].bias)
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, c1: Tensor, c2: Tensor, timestep: Optional[Tensor] = None) -> Tensor:
        del timestep  # reserved for future time-conditioned α
        # Residual branches: keep original modality signal and learn residual refinements.
        z1 = c1 + self.net_c1(c1)
        z2 = c2 + self.net_c2(c2)
        a = torch.sigmoid(self.alpha_logit).to(dtype=c1.dtype, device=c1.device)
        fused = (1.0 - a) * z1 + a * z2
        return fused.to(c1.dtype)


class FrequencyControlFusion(nn.Module):
    """
    Frequency-domain fusion for dual controls:
      - high frequency from edges (c1)
      - low frequency from LQ (c2)

    Pipeline:
      Fe, Fl = FFT2(c1), FFT2(c2)
      M_low = sigmoid(k * (r0 - r)), M_high = 1 - M_low
      Fused_freq = M_high * Fe + M_low * Fl
      freq_fuse = IFFT2(Fused_freq).real
      out = freq_fuse + small_residual([c1, c2])

    ``r0`` and ``k`` are per-channel learnable vectors and can be frozen at
    early training then unfrozen later.
    """

    def __init__(
        self,
        init_r0: float = 0.25,
        init_k: float = 20.0,
        residual_mid_channels: int = 16,
        residual_scale: float = 0.1,
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        init_r0 = float(max(min(init_r0, 1.0 - 1e-4), 1e-4))
        r0 = torch.full((self.latent_channels,), init_r0, dtype=torch.float32)
        self.r0_logit = nn.Parameter(torch.log(r0 / (1.0 - r0)))
        self.k = nn.Parameter(torch.full((self.latent_channels,), float(init_k), dtype=torch.float32))
        self.residual_scale = float(residual_scale)
        self._freq_params_frozen = False

        self.small_residual = nn.Sequential(
            nn.Conv2d(8, residual_mid_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(residual_mid_channels, 4, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.small_residual[-1].weight)
        nn.init.zeros_(self.small_residual[-1].bias)

    def freeze_frequency_params(self) -> None:
        self.r0_logit.requires_grad_(False)
        self.k.requires_grad_(False)
        self._freq_params_frozen = True

    def unfreeze_frequency_params(self) -> None:
        self.r0_logit.requires_grad_(True)
        self.k.requires_grad_(True)
        self._freq_params_frozen = False

    def frequency_params_frozen(self) -> bool:
        return bool(self._freq_params_frozen)

    def _radial_map(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
        fx = torch.fft.fftfreq(w, d=1.0, device=device).view(1, w)
        r = torch.sqrt(fx * fx + fy * fy)
        r = r / (r.max() + 1e-8)
        return r.to(dtype=dtype)

    def forward(self, c1: Tensor, c2: Tensor, timestep: Optional[Tensor] = None) -> Tensor:
        del timestep  # reserved for future time-conditioned cutoff
        _, c, h, w = c1.shape
        dtype = c1.dtype
        device = c1.device
        if c != self.latent_channels:
            raise ValueError(
                f"FrequencyControlFusion expected {self.latent_channels} channels, got {c}. "
                "Please set latent_channels to match control latent channels."
            )

        fe = torch.fft.fft2(c1.float(), dim=(-2, -1))
        fl = torch.fft.fft2(c2.float(), dim=(-2, -1))

        r = self._radial_map(h, w, device=device, dtype=torch.float32).view(1, 1, h, w)
        r0 = torch.sigmoid(self.r0_logit).to(dtype=torch.float32, device=device).view(1, c, 1, 1)
        k = torch.clamp(self.k, 1.0, 80.0).to(dtype=torch.float32, device=device).view(1, c, 1, 1)
        m_low = torch.sigmoid(k * (r0 - r))
        m_high = 1.0 - m_low

        fused_freq = m_high * fe + m_low * fl
        freq_fuse = torch.fft.ifft2(fused_freq, dim=(-2, -1)).real.to(dtype=dtype)

        small_residual = self.small_residual(torch.cat([c1, c2], dim=1)).to(dtype=dtype)
        out = freq_fuse + self.residual_scale * small_residual
        return out.to(dtype)


# The implementation of ControlNet-Half architrecture
# https://github.com/lllyasviel/ControlNet/discussions/188
class ControlT2IDitBlockHalf(Module):
    def __init__(self, base_block: PixArtMSBlock, block_index: 0) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size = base_block.hidden_size
        if self.block_index == 0:
            self.before_proj = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(hidden_size, hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(x + c, y, t, mask)
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(c, y, t, mask)
            c_skip = self.after_proj(c)
        
        return c, c_skip
        

# The implementation of ControlPixArtHalf net
class ControlPixArtHalf(Module):
    # only support single res model
    def __init__(
        self,
        base_model: PixArt,
        copy_blocks_num: int = 13,
        use_vae_dual_fusion: bool = False,
        use_dual_nonlinear_control: bool = False,
        use_frequency_control_fusion: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(base_model.blocks)
        self.use_vae_dual_fusion = use_vae_dual_fusion
        self.use_dual_nonlinear_control = use_dual_nonlinear_control
        self.use_frequency_control_fusion = use_frequency_control_fusion
        # Runtime switch used by training schedule:
        # True -> bypass frequency fusion and use edges-only control latent.
        self.disable_freq_fusion_runtime = False
        enabled_fusions = int(use_vae_dual_fusion) + int(use_dual_nonlinear_control) + int(use_frequency_control_fusion)
        if enabled_fusions > 1:
            raise ValueError(
                "Choose only one fusion: use_vae_dual_fusion, "
                "use_dual_nonlinear_control, or use_frequency_control_fusion."
            )
        self.dual_nl_fusion = DualNonlinearControlFusion() if use_dual_nonlinear_control else None
        self.vae_fusion = VaeLatentFusion3x3() if use_vae_dual_fusion else None
        self.freq_fusion = FrequencyControlFusion() if use_frequency_control_fusion else None
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        # Copy first copy_blocks_num block
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlT2IDitBlockHalf(base_model.blocks[i], i))
        self.controlnet = nn.ModuleList(self.controlnet)
    
        # Original logic: use the same x_embedder as base_model (4 channels)
        # vae_feat_loader_npz returns mean + std * sample (4 channels after adding noise)
        # So we don't need a separate c_x_embedder, just use the base model's x_embedder
    
    def __getattr__(self, name: str) -> Tensor or Module:
        # Original logic
        if name in ['forward', 'forward_with_dpmsolver', 'forward_with_cfg', 'forward_c', 'load_state_dict']:
            return self.__dict__[name]
        elif name in ['base_model', 'controlnet', 'vae_fusion', 'dual_nl_fusion', 'freq_fusion']:
            return super().__getattr__(name)
        else:
            return getattr(self.base_model, name)

    def _prepare_control_latent(self, c: Tensor, c2: Optional[Tensor], timestep: Optional[Tensor]) -> Tensor:
        """Merge two 4ch latents when dual path is enabled; else pass ``c`` through."""
        if self.dual_nl_fusion is not None:
            if c2 is None:
                c2 = torch.zeros_like(c)
            return self.dual_nl_fusion(c, c2, timestep)
        if self.freq_fusion is not None:
            if bool(getattr(self, "disable_freq_fusion_runtime", False)):
                return c
            if c2 is None:
                c2 = torch.zeros_like(c)
            return self.freq_fusion(c, c2, timestep)
        if self.vae_fusion is None:
            if c2 is not None:
                raise ValueError(
                    "c2 was passed but model was built without dual conditioning "
                    "(set use_vae_dual_fusion or use_dual_nonlinear_control)."
                )
            return c
        if c2 is None:
            c2 = torch.zeros_like(c)
        return self.vae_fusion(c, c2, timestep)

    def forward_c(self, c):
        self.h, self.w = c.shape[-2]//self.patch_size, c.shape[-1]//self.patch_size
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(c.device).to(self.dtype)
        # Original logic: use base model's x_embedder (expects 4 channels)
        return self.x_embedder(c) + pos_embed if c is not None else c

    # def forward(self, x, t, c, **kwargs):
    #     return self.base_model(x, t, c=self.forward_c(c), **kwargs)
    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, c2=None, **kwargs):
        # modify the original PixArtMS forward function
        if c is not None:
            c = c.to(self.dtype)
            if c2 is not None:
                c2 = c2.to(self.dtype)
            c = self._prepare_control_latent(c, c2, timestep)
            c = self.forward_c(c)
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep.to(x.dtype))  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_dpmsolver(self, x, t, y, data_info, c, c2=None, **kwargs):
        model_out = self.forward(x, t, y, data_info=data_info, c=c, c2=c2, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    # def forward_with_dpmsolver(self, x, t, y, data_info, c, **kwargs):
    #     return self.base_model.forward_with_dpmsolver(x, t, y, data_info=data_info, c=self.forward_c(c), **kwargs)

    def forward_with_cfg(self, x, t, y, cfg_scale, data_info, c, **kwargs):
        return self.base_model.forward_with_cfg(x, t, y, cfg_scale, data_info, c=self.forward_c(c), **kwargs)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # New ControlNet checkpoints may contain base_model/controlnet/vae_fusion or dual_nl_fusion.
        # Old checkpoints may be an unwrapped base model (blocks.N.xxx; remap to blocks.N.base_block.xxx).
        if all(
            k.startswith(("base_model", "controlnet", "vae_fusion", "dual_nl_fusion", "freq_fusion"))
            for k in state_dict.keys()
        ):
            return super().load_state_dict(state_dict, strict)

        remapped_state_dict = dict(state_dict)
        renamed_pairs = []
        for k in list(remapped_state_dict.keys()):
            v = re.sub(r"(blocks\.\d+)(.*)", r"\1.base_block\2", k)
            if k != v:
                renamed_pairs.append((k, v))

        for k, v in renamed_pairs:
            print(f"replace {k} to {v}")
            remapped_state_dict[v] = remapped_state_dict.pop(k)

        # If the old format contains only the base model, load into base_model; otherwise prefer loading the full wrapped model.
        if all(k.startswith("base_model") for k in remapped_state_dict.keys()):
            return self.base_model.load_state_dict(remapped_state_dict, strict)
        return super().load_state_dict(remapped_state_dict, strict)
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    @property
    def dtype(self):
        # Return the dtype of model parameters
        return next(self.parameters()).dtype


# The implementation for PixArtMS_Half + 1024 resolution
class ControlPixArtMSHalf(ControlPixArtHalf):
    # support multi-scale res model (multi-scale model can also be applied to single reso training & inference)
    def __init__(
        self,
        base_model: PixArtMS,
        copy_blocks_num: int = 13,
        use_vae_dual_fusion: bool = False,
        use_dual_nonlinear_control: bool = False,
        use_frequency_control_fusion: bool = False,
    ) -> None:
        super().__init__(
            base_model=base_model,
            copy_blocks_num=copy_blocks_num,
            use_vae_dual_fusion=use_vae_dual_fusion,
            use_dual_nonlinear_control=use_dual_nonlinear_control,
            use_frequency_control_fusion=use_frequency_control_fusion,
        )

    def forward(self, x, timestep, y, mask=None, data_info=None, c=None, c2=None, **kwargs):
        # modify the original PixArtMS forward function
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        c2: optional second 4ch VAE control latent (LQ) when dual fusion or dual nonlinear control is enabled.
        """
        if c is not None:
            c = c.to(self.dtype)
            if c2 is not None:
                c2 = c2.to(self.dtype)
            c = self._prepare_control_latent(c, c2, timestep)
            c = self.forward_c(c)
        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size

        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.h, self.w), lewei_scale=self.lewei_scale, base_size=self.base_size)).unsqueeze(0).to(x.device).to(self.dtype)
        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(timestep)  # (N, D)
        csize = self.csize_embedder(c_size, bs)  # (N, D)
        ar = self.ar_embedder(ar, bs)  # (N, D)
        t = t + torch.cat([csize, ar], dim=1)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, D)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # define the first layer
        x = auto_grad_checkpoint(self.base_model.blocks[0], x, y, t0, y_lens, **kwargs)  # (N, T, D) #support grad checkpoint

        if c is not None:
            # update c
            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = auto_grad_checkpoint(self.controlnet[index - 1], x, y, t0, y_lens, c, **kwargs)
                x = auto_grad_checkpoint(self.base_model.blocks[index], x + c_skip, y, t0, y_lens, **kwargs)
        
            # update x
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)
        else:
            for index in range(1, self.total_blocks_num):
                x = auto_grad_checkpoint(self.base_model.blocks[index], x, y, t0, y_lens, **kwargs)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
