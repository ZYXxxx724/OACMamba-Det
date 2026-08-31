# Copyright (c) OACM-Det authors. All rights reserved.
"""Core OACM (Orientation-Aware ConvMamba) modules.

This file contains the core building block of OACM-Det:

- ``RMSNorm``: scale-only normalization that preserves rotation equivariance
  (LayerNorm would subtract the mean and break per-direction statistics).
- ``ImprovedSRCMLayer``: a single shared Mamba transformation (no residual).
- ``OACMBlock``: the orientation-aware ConvMamba block. It rotates the input
  feature into several canonical directions, applies the *same* shared Mamba
  transformation to each branch, rotates the outputs back, averages them and
  adds a residual. All branches share one Mamba core, which keeps the block
  compact while making the transformation rotation-equivariant.
- ``ParameterSharedOACM``: a compatibility alias of ``OACMBlock``.
"""

import torch
import torch.nn as nn
import warnings
from typing import Optional

from mmengine.model import BaseModule
from mmyolo.registry import MODELS

# ==================== dependency check ====================
try:
    from mamba_ssm import Mamba as OfficialMamba
    MAMBA_AVAILABLE = True
except ImportError:
    raise RuntimeError(
        'Critical dependency mamba-ssm is not installed. '
        'Please run: pip install mamba-ssm'
    )

# torchvision is only used for the (optional) non-orthogonal 8-direction mode.
try:
    import torchvision.transforms.functional as TF
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn('torchvision is not installed; 8-direction rotation is unavailable.')

__all__ = ['ImprovedSRCMLayer', 'OACMBlock', 'ParameterSharedOACM', 'RMSNorm']


# ==================== 1. Equivariant normalization (RMSNorm) ====================
class RMSNorm(nn.Module):
    """Scale-only normalization that keeps rotation equivariance.

    LayerNorm subtracts the mean and would make the local statistics of
    different rotated directions inconsistent. RMSNorm only rescales by the
    root-mean-square and leaves the mean untouched, which is better suited to
    the equivariance constraint of OACM.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: [B, L, C]
        rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ==================== 2. Basic Mamba transform (residual-free) ====================
class ImprovedSRCMLayer(nn.Module):
    """Core Mamba transformation layer.

    The residual is handled by ``OACMBlock`` so that the direction transform
    logic stays clean; this layer only applies RMSNorm + Mamba.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim

        if MAMBA_AVAILABLE:
            self.mamba = OfficialMamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.is_fallback = False
        else:
            # Unreachable in practice because of the import guard above.
            self.mamba = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.SiLU(),
                nn.Conv2d(dim, dim, 1),
            )
            self.is_fallback = True

        self.norm = RMSNorm(dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        if self.is_fallback:
            return self.mamba(x)

        # 2D -> 1D sequence -> RMSNorm -> Mamba -> 2D
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)   # [B, HW, C]
        x_seq = self.norm(x_seq)
        x_seq = self.mamba(x_seq)

        out = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


# ==================== 3. Core OACMBlock ====================
@MODELS.register_module()
class OACMBlock(BaseModule):
    """Orientation-Aware ConvMamba Block (OACM).

    Args:
        dim: Number of feature channels.
        num_directions: Number of rotation directions.
            - 1: no rotation [0]
            - 2: horizontal symmetry [0, 180]
            - 4: full equivariance [0, 90, 180, 270] (recommended)
            - 8: dense sampling [0, 45, ..., 315] (requires interpolation)
        d_state: Mamba state dimension.
        d_conv: Mamba convolution kernel size.
        expand: Mamba expansion factor.
        dropout: Reserved interface (unused).
        use_learnable_gamma: If True, use a learnable residual scale ``gamma``;
            otherwise use the fixed scalar ``alpha``.
        gamma_init: Initial value of ``gamma`` (only when learnable).
        alpha: Fixed residual scale (used when ``use_learnable_gamma=False``).
    """

    ROTATION_CONFIGS = {
        1: [0],
        2: [0, 180],
        4: [0, 90, 180, 270],
        8: [0, 45, 90, 135, 180, 225, 270, 315],
    }

    def __init__(self,
                 dim,
                 num_directions=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dropout=0.0,
                 use_learnable_gamma=False,
                 gamma_init=1.0,
                 alpha=1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        if num_directions not in self.ROTATION_CONFIGS:
            raise ValueError(
                f'num_directions must be one of {list(self.ROTATION_CONFIGS.keys())}, '
                f'got {num_directions}'
            )

        self.num_directions = num_directions
        self.angles = self.ROTATION_CONFIGS[num_directions]

        self.use_learnable_gamma = use_learnable_gamma
        self.alpha = alpha
        if self.use_learnable_gamma:
            self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        else:
            self.gamma = None

        # All directions share this single Mamba core.
        self.mamba_core = ImprovedSRCMLayer(
            dim=dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.use_interpolation = any(angle % 90 != 0 for angle in self.angles)

        if self.use_interpolation:
            if not TORCHVISION_AVAILABLE:
                raise RuntimeError(
                    f'num_directions={num_directions} requires torchvision. '
                    f'Please run: pip install torchvision'
                )
            warnings.warn(
                f'[OACMBlock] num_directions={num_directions} uses non-orthogonal '
                f'rotations (angles: {self.angles}) and falls back to bilinear '
                f'interpolation, which may introduce minor accuracy loss.'
            )

        self._zero_init_mamba_output()

    def _zero_init_mamba_output(self):
        """Zero-initialize the output projection of Mamba for stable warm-up."""
        if self.mamba_core.is_fallback:
            nn.init.constant_(self.mamba_core.mamba[-1].weight, 0.0)
            if self.mamba_core.mamba[-1].bias is not None:
                nn.init.constant_(self.mamba_core.mamba[-1].bias, 0.0)
        else:
            if hasattr(self.mamba_core.mamba, 'out_proj'):
                nn.init.constant_(self.mamba_core.mamba.out_proj.weight, 0.0)
                if hasattr(self.mamba_core.mamba.out_proj, 'bias') and \
                        self.mamba_core.mamba.out_proj.bias is not None:
                    nn.init.constant_(self.mamba_core.mamba.out_proj.bias, 0.0)

    def _rotate(self, x, angle, inverse=False):
        if inverse:
            angle = -angle

        angle = angle % 360

        if angle % 90 == 0:
            k = int(angle // 90)
            return torch.rot90(x, k=k, dims=[2, 3])
        else:
            return TF.rotate(
                x,
                angle=-angle,
                interpolation=TF.InterpolationMode.BILINEAR,
            )

    def forward(self, x, direction_weights: Optional[torch.Tensor] = None):
        identity = x

        if direction_weights is None:
            features_sum = 0
            for angle in self.angles:
                x_rot = self._rotate(x, angle, inverse=False)
                feat = self.mamba_core(x_rot)
                feat_back = self._rotate(feat, angle, inverse=True)
                features_sum = features_sum + feat_back
            out = features_sum / float(self.num_directions)
        else:
            assert direction_weights.shape[1] == self.num_directions, (
                f'direction_weights channels ({direction_weights.shape[1]}) must '
                f'equal num_directions ({self.num_directions})'
            )

            features_weighted = 0
            weight_sum = 0

            for idx, angle in enumerate(self.angles):
                w_k = direction_weights[:, idx:idx + 1, :, :]
                x_rot = self._rotate(x, angle, inverse=False)
                feat = self.mamba_core(x_rot)
                feat_back = self._rotate(feat, angle, inverse=True)
                features_weighted = features_weighted + feat_back * w_k
                weight_sum = weight_sum + w_k

            out = features_weighted / (weight_sum + 1e-7)

        self.last_modulation_strength = out.abs().mean().detach().cpu().item()

        if self.use_learnable_gamma:
            return identity + self.gamma * out
        else:
            return identity + self.alpha * out


# ==================== 4. Parameter-shared alias ====================
@MODELS.register_module(force=True)
class ParameterSharedOACM(OACMBlock):
    """Compatibility alias for legacy config names.

    ``OACMBlock`` already shares a single Mamba core across all directions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
