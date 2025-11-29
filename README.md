# v-CLR-CNN-Based

This is an attempt at recreating the "v-CLR: View-Consistent Learning for Open-World Instance Segmentation" paper ([arXiv:2504.01383](https://arxiv.org/abs/2504.01383)) using a CNN (ConvNeXt) instead of a transformer.

## Overview

The v-CLR approach uses view-consistent learning to improve open-world instance segmentation. This implementation adapts the methodology to use a ConvNeXt-tiny backbone instead of the original MaskDINO/Swin Transformer architecture.

## Key Differences from Original Paper

1. **Architecture**: Uses ConvNeXt-tiny backbone instead of MaskDINO/Swin Transformer
2. **Detection Focus**: Current implementation focuses on bounding box detection (masks not yet implemented)
3. **Dense Prediction**: Uses dense per-pixel predictions instead of DETR-style object queries

## Implementation Notes

### Corrections Applied

The following issues were identified and corrected from the original implementation:

1. **VOC Class Names**: Fixed to use the correct 20 PASCAL VOC categories in COCO naming conventions:
   - Removed incorrect classes: `truck`, `elephant`, `bear`, `zebra`, `giraffe`
   - Added missing VOC classes: `bottle`, `chair`, `couch`, `potted plant`, `dining table`

2. **Ground-Truth Target Application**: Fixed training loop to apply GT supervision only to the natural view as specified in v-CLR. Depth and stylized views are now trained via view-consistency losses (L_obj, L_sim) only.

3. **Temperature Scaling**: Added temperature scaling parameter to the cosine similarity loss for better contrastive learning behavior.

### Known Limitations

1. **No Mask Prediction**: The original v-CLR paper includes instance segmentation masks. This implementation only produces bounding boxes.

2. **Dense vs Query-based**: The paper uses object queries (like DETR/MaskDINO), while this implementation uses dense per-pixel predictions.

## Loss Functions

The v-CLR loss consists of:
- **L_gt**: Ground-truth detection loss (applied only to natural view)
- **L_obj**: Proposal-prediction alignment loss across views
- **L_sim**: Query feature cosine similarity loss between views

## Requirements

- PyTorch with CUDA
- torchvision
- pycocotools
- COCO dataset with depth and stylized views

## Usage

Run the Jupyter notebook `vclr_convnext_rev2_teach_stud_1epoch.ipynb` to train the model.
