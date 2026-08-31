# Copyright (c) OACM-Det authors. All rights reserved.
"""Orientation-Aware FPN (OA-FPN).

An extension of ``CSPNeXtPAFPN`` used by RTMDet-R. It appends an ``OACMBlock``
to selected output levels (by default P4 and P5) to add orientation-aware
multi-scale feature fusion. The detection head and losses are left unchanged.

``oacm_layers_override`` maps to neck output levels [P3, P4, P5], e.g.
``[False, True, True]`` applies OACM on P4 and P5 only.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Sequence

from mmyolo.registry import MODELS
from mmyolo.models.necks.cspnext_pafpn import CSPNeXtPAFPN
from mmyolo.models.backbones.mamba_modules import OACMBlock


@MODELS.register_module(force=True)
class OAFPN(CSPNeXtPAFPN):
    """PAFPN neck with orientation-aware OACM calibration blocks.

    Args:
        use_oacm: Enable OACM calibration.
        oacm_cfg: Keyword arguments passed to ``OACMBlock``.
        oacm_layers_override: Per-level switch for [P3, P4, P5].
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 out_channels: int,
                 *args,
                 use_oacm: bool = True,
                 oacm_cfg: Optional[Dict] = None,
                 oacm_layers_override=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            *args,
            **kwargs,
        )
        self.use_oacm = use_oacm

        if oacm_cfg is None:
            oacm_cfg = dict(num_directions=4, d_state=16)

        num_out_levels = len(in_channels)
        if oacm_layers_override is None:
            oacm_layers_override = [True] * num_out_levels
        assert len(oacm_layers_override) == num_out_levels, \
            f'oacm_layers_override must have length {num_out_levels}'
        self.oacm_layers_override = oacm_layers_override

        self.oacm_layers = nn.ModuleList()
        actual_channels = self.out_channels

        for i in range(num_out_levels):
            if self.use_oacm and self.oacm_layers_override[i]:
                self.oacm_layers.append(
                    OACMBlock(dim=actual_channels, **oacm_cfg))
            else:
                self.oacm_layers.append(nn.Identity())

        print(
            f'[OAFPN] oacm_layers_override={self.oacm_layers_override}, '
            f'out_channels={actual_channels}'
        )

    def forward(self, inputs: Sequence[torch.Tensor]):
        outs = super().forward(inputs)
        calibrated_outs = []
        for i, feat in enumerate(outs):
            feat = self.oacm_layers[i](feat)
            calibrated_outs.append(feat)
        return tuple(calibrated_outs)
