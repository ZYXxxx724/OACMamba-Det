# OACM-Det

**OACM-Det: Orientation-Aware ConvMamba for Rotated Object Detection in Remote Sensing Images**

OACM-Det is a plug-and-play orientation-aware ConvMamba framework for rotated
object detection. It improves the strong one-stage baseline RTMDet-R on
DOTA-v1.0, HRSC2016, and DIOR-R by modifying **only the backbone and neck**; the
detection head, label assignment, losses, and training pipeline stay unchanged.

- Backbone: **OACM-CSPNeXt** ([`MambaCSPNeXt`](mmyolo/models/backbones/mamba_cspnext.py))
- Neck: **OA-FPN** ([`OAFPN`](mmyolo/models/necks/orientation_aware_fpn.py))
- Core block: **OACM** ([`OACMBlock`](mmyolo/models/backbones/mamba_modules.py))

## Core implementation

| Component | File | Description |
| --- | --- | --- |
| OACM block | [`mmyolo/models/backbones/mamba_modules.py`](mmyolo/models/backbones/mamba_modules.py) | Four rotation branches + shared ConvMamba + inverse rotation + average fusion + residual |
| Backbone | [`mmyolo/models/backbones/mamba_cspnext.py`](mmyolo/models/backbones/mamba_cspnext.py) | CSPNeXt with OACM injected at P4/P5 |
| Neck | [`mmyolo/models/necks/orientation_aware_fpn.py`](mmyolo/models/necks/orientation_aware_fpn.py) | PAFPN with OACM calibrated on P4/P5 |

## Configs

| Config | Description |
| --- | --- |
| [`configs/rtmdet/rotated/rtmdet-r_m_syncbn_fast_2xb4-36e_dota.py`](configs/rtmdet/rotated/rtmdet-r_m_syncbn_fast_2xb4-36e_dota.py) | RTMDet-R (M) baseline |
| [`configs/rtmdet/rotated/oacm-det_m_syncbn_fast_2xb4-36e_dota.py`](configs/rtmdet/rotated/oacm-det_m_syncbn_fast_2xb4-36e_dota.py) | OACM-Det (M) |

The OACM-Det config only overrides `backbone` and `neck`; every other setting is
inherited unchanged from the baseline config.

## Installation

OACM-Det is a drop-in extension of [MMYOLO](https://github.com/open-mmlab/mmyolo)
and relies on [MMRotate](https://github.com/open-mmlab/mmrotate) /
[MMDetection](https://github.com/open-mmlab/mmdetection) for rotated detection.

1. Install the required frameworks following the official docs (MMCV / MMEngine /
   MMDetection / MMRotate / MMYOLO).
2. Install the Mamba backend:

   ```bash
   pip install mamba-ssm
   ```

3. Copy this repository's `mmyolo/` and `configs/` folders into your MMYOLO
   installation (merge into the existing `mmyolo/` and `configs/` directories).
   The base configs (`rtmdet-r_l_syncbn_fast_2xb4-36e_dota.py` and
   `../../_base_/default_runtime.py`) are part of the standard MMYOLO package.

## Data preparation

Prepare DOTA-v1.0 (split-SS) as described in the MMYOLO rotated detection
documentation. By default the base config reads from `data/split_ss_dota/`; set
`data_root` in the base config to your own path if needed.

## Pretrained weights / checkpoint

Both the baseline and OACM-Det are initialized from the **same** COCO-pretrained
CSPNeXt-M backbone weight:

- **checkpoint (shared init weight):**
  [`cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth`](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-m_8xb256-rsb-a1-600e_in1k-ecb3bbd9.pth)

Notes:

- The **baseline checkpoint and the OACM-Det initialization weight are
  identical**. To reproduce the experiments, load the same pretrained weight for
  both the baseline and OACM-Det.
- MMEngine automatically ignores the newly added OACM parameters when loading
  this weight, so the convolutional layers are warm-started while the OACM
  blocks are trained from scratch.

## Train

```bash
# Baseline
python tools/train.py configs/rtmdet/rotated/rtmdet-r_m_syncbn_fast_2xb4-36e_dota.py --work-dir work_dirs/rtmdet-r_m

# OACM-Det
python tools/train.py configs/rtmdet/rotated/oacm-det_m_syncbn_fast_2xb4-36e_dota.py --work-dir work_dirs/oacm-det_m
```

## Test

```bash
# OACM-Det on the validation set
python tools/test.py configs/rtmdet/rotated/oacm-det_m_syncbn_fast_2xb4-36e_dota.py \
    work_dirs/oacm-det_m/best_dota_mAP_epoch_*.pth
```

## License

This project is released for research purposes. The code builds on MMYOLO,
MMDetection, and MMRotate (Apache License 2.0).
