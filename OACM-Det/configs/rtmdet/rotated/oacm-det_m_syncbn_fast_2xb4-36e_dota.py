# OACM-Det (M) for rotated object detection on DOTA-v1.0.
#
# OACM-Det = RTMDet-R baseline + OACM-CSPNeXt backbone (MambaCSPNeXt) + OA-FPN
# neck (OAFPN). Only the backbone and neck are changed; the head, losses,
# optimizer, schedule and data pipeline are inherited unchanged from the
# baseline config below.

_base_ = './rtmdet-r_m_syncbn_fast_2xb4-36e_dota.py'

# Register the OACM-Det custom modules. Importing them at config time triggers
# registration, so there is no need to modify mmyolo's ``__init__.py``.
import mmyolo.models.backbones.mamba_modules      # noqa: F401  -> OACMBlock
import mmyolo.models.backbones.mamba_cspnext      # noqa: F401  -> MambaCSPNeXt
import mmyolo.models.necks.orientation_aware_fpn  # noqa: F401  -> OAFPN

model = dict(
    # --- Backbone: CSPNeXt + OACM blocks on stage3 (P4) and stage4 (P5). ---
    backbone=dict(
        type='MambaCSPNeXt',
        insert_oacm_indices=(2, 3),
        oacm_cfg=dict(
            num_directions=4,
            d_state=16,
            expand=2,
            use_learnable_gamma=False,
        ),
        # init_cfg is inherited from the baseline: the same COCO-pretrained
        # cspnext-m weight is loaded, so OACM-Det shares the baseline's
        # initialization (MMEngine ignores the newly added OACM parameters).
    ),

    # --- Neck: PAFPN + OACM blocks on P4 and P5. ---
    neck=dict(
        type='OAFPN',
        use_oacm=True,
        oacm_layers_override=[False, True, True],  # P4 + P5
        oacm_cfg=dict(
            num_directions=4,
            d_state=16,
            expand=2,
            use_learnable_gamma=False,
        ),
    ),
)
