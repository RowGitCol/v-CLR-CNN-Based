# v-CLR-CNN-Based

This is an attempt at recreating the "v-CLR: View-Consistent Learning for Open-World Instance Segmentation" paper ([arXiv:2504.01383](https://arxiv.org/abs/2504.01383)) using a CNN (ConvNeXt) instead of a transformer.

## Overview

The v-CLR approach uses view-consistent learning to improve open-world instance segmentation. This implementation adapts the methodology to use a ConvNeXt-tiny backbone instead of the original MaskDINO/Swin Transformer architecture.

## Key Differences from Original Paper

1. **Architecture**: Uses ConvNeXt-tiny backbone instead of MaskDINO/Swin Transformer
2. **Detection Focus**: Current implementation focuses on bounding box detection (masks not yet implemented)
3. **Dense Prediction**: Uses dense per-pixel predictions instead of DETR-style object queries

## Paper's Methodology (v-CLR)

Per the paper, v-CLR uses a two-branch architecture:

1. **Natural Image Branch**: Always receives natural images, updated as EMA of transformed branch
2. **Transformed Image Branch**: Randomly processes transformed views OR natural image with equal probability

### Loss Functions (from paper)

- **L_sim** = (1/N̄) Σ (1 - cos(q1, q2)) — Cosine similarity between matched queries
- **L_obj** = λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_giou — Supervision from proposals
- **L_match** = λ_obj·L_obj + λ_sim·L_sim — Matching objective
- **L_gt** = Same as L_obj but using ground truth on **transformed branch queries**
- **Total**: L = λ_match·L_match + λ_gt·L_gt

## Implementation Notes

### Corrections Applied

1. **VOC Class Names**: Fixed to use the correct 20 PASCAL VOC categories in COCO naming conventions:
   - Removed incorrect classes: `truck`, `elephant`, `bear`, `zebra`, `giraffe`
   - Added missing VOC classes: `bottle`, `chair`, `couch`, `potted plant`, `dining table`

2. **GT Supervision**: Per paper, L_gt is computed on transformed branch queries. The student (transformed branch) receives GT supervision on all processed views.

3. **Temperature Scaling**: Added temperature scaling parameter to the cosine similarity loss.

### Known Limitations

1. **No Mask Prediction**: The original v-CLR paper includes instance segmentation masks. This implementation only produces bounding boxes.

2. **Dense vs Query-based**: The paper uses transformer object queries (DETR-style), while this implementation uses dense CNN predictions.

## Requirements

- PyTorch with CUDA
- torchvision
- pycocotools
- COCO dataset with depth and stylized views
- Ultralytics API
- Ultralytics YOLOv11 small-seg checkpoint
- YOLACT ResNet50 80000 checkpoint

## Usage

- Run the Jupyter notebook `vclr_convnext_rev2_teach_stud_6epoch.ipynb` to train a ConvNext Tiny based model.
- Run the Jupyter notebook `vclr_rev6_YoLo.ipynb` to train a YOLOv11s-seg based model.
- Run the Jupyter notebook `vclr_YOLCAT.ipynb` inside a YOLCAT model folder to train a YOLCAT based model.
