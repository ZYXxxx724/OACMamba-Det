# RTMDet-R (M) baseline for rotated object detection on DOTA-v1.0.
#
# This is the strong one-stage rotated detector baseline used in OACM-Det.
# OACM-Det keeps every setting here (head, loss, optimizer, schedule, data
# pipeline) unchanged and only swaps the backbone and neck.

_base_ = './rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py'

# Backbone pretrained weights (shared by the baseline and OACM-Det, see README).
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth'  # noqa

# ========================modified parameters======================
deepen_factor = 0.67
widen_factor = 0.75

# Submission dir for result submit
submission_dir = './work_dirs/{{fileBasenameNoExtension}}/submission'

# =======================Unmodified in most cases==================
model = dict(
    backbone=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        init_cfg=dict(checkpoint=checkpoint)),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
