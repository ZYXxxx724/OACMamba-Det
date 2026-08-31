# Copyright (c) OACM-Det authors. All rights reserved.
"""MambaCSPNeXt backbone (OACM-CSPNeXt).

A thin extension of the CSPNeXt backbone used by RTMDet-R. It injects an
``OACMBlock`` at the end of selected stages (by default stages 3 and 4, which
produce P4 and P5) to add orientation-aware feature modeling without touching
the detection head or the training pipeline.
"""

from typing import Dict, Optional, Tuple

from mmyolo.registry import MODELS
from mmyolo.models.backbones.cspnext import CSPNeXt
from mmyolo.models.backbones.mamba_modules import OACMBlock


@MODELS.register_module(force=True)
class MambaCSPNeXt(CSPNeXt):
    """CSPNeXt backbone with pluggable OACM blocks.

    Args:
        insert_oacm_indices: 0-based stage indices where OACM is injected.
            ``(2, 3)`` means stage3 (P4) and stage4 (P5).
        oacm_cfg: Keyword arguments passed to ``OACMBlock``.
    """

    def __init__(self,
                 *args,
                 insert_oacm_indices: Tuple[int] = (2, 3),
                 oacm_cfg: Optional[Dict] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if oacm_cfg is None:
            self.oacm_cfg = dict(num_directions=4, d_state=16)
        else:
            self.oacm_cfg = oacm_cfg

        self.insert_oacm_indices = insert_oacm_indices
        self._inject_oacm_blocks()

    def _inject_oacm_blocks(self):
        """Append an OACM block to the end of each selected stage."""
        for idx in self.insert_oacm_indices:
            stage_name = f'stage{idx + 1}'
            if not hasattr(self, stage_name):
                continue

            stage_sequential = getattr(self, stage_name)

            # Infer the output channels of the current stage.
            try:
                # In CSPNeXt the first layer of a stage is usually a ConvModule.
                out_channels = stage_sequential[0].conv.out_channels
            except Exception:
                out_channels = stage_sequential[-1].main_conv.out_channels

            block_instance = OACMBlock(dim=out_channels, **self.oacm_cfg)
            stage_sequential.add_module('oacm_block', block_instance)

            print(
                f'[MambaCSPNeXt] injected OACMBlock into {stage_name} '
                f'(output P{idx + 2}, channels={out_channels})'
            )
