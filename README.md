# v-CLR-CNN-Based

This is an attempt at recreating the "v-CLR: View-Consistent Learning for Open-World Instance Segmentation" paper ([arXiv:2504.01383](https://arxiv.org/abs/2504.01383)) using a CNN (ConvNeXt) instead of a transformer.

## Overview

The v-CLR approach uses view-consistent learning to improve open-world instance segmentation. This implementation adapts the methodology to use a ConvNeXt-tiny backbone instead of the original MaskDINO/Swin Transformer architecture.

## Paper vs Implementation Comparison

### Architecture

| Component | Original Paper | This Implementation |
|-----------|---------------|---------------------|
| Backbone | Swin Transformer | ConvNeXt-tiny |
| Detector | MaskDINO / DeformableDETR | Dense CNN head |
| Feature Representation | Object queries (DETR-style) | Dense per-pixel predictions |
| Output | Bounding boxes + Instance masks | Bounding boxes only |

### Two-Branch Architecture

| Component | Original Paper | This Implementation | Status |
|-----------|---------------|---------------------|--------|
| Teacher branch | EMA-updated, processes natural images | EMA-updated, processes natural images | ✅ Matches |
| Student branch | Processes transformed views | Processes all views (nat + transformed) | ✅ Matches |
| EMA momentum | 0.999 | 0.999 | ✅ Matches |

### View Sampling Strategy

| Aspect | Original Paper | This Implementation | Status |
|--------|---------------|---------------------|--------|
| Available views | Depth, Stylized, Edge | Depth, Stylized, (Edge optional) | ✅ Matches |
| Selection strategy | Random selection: transformed view OR natural with equal probability | Random selection between available transformed views | ⚠️ Minor difference |

**Note on view sampling**: The paper states the transformed branch randomly selects ONE transformed view OR the natural image with equal probability. This implementation randomly picks between depth and style when both exist, but always processes at least one transformed view when available. This is a minor deviation that may provide richer consistency learning.

### Loss Functions

| Loss Component | Original Paper | This Implementation | Status |
|----------------|---------------|---------------------|--------|
| **L_sim** | (1/N̄) Σ (1 - cos(q1, q2)) | (1/N̄) Σ (1 - cos(q1, q2)) / temperature | ✅ Matches (with temp scaling) |
| **L_obj** | λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_giou | L1 box alignment | ⚠️ Adapted for CNN |
| **L_match** | λ_obj·L_obj + λ_sim·L_sim | LAMBDA_OBJ·L_obj + LAMBDA_SIM·L_sim | ✅ Matches |
| **L_gt** | Same as L_obj with ground truth | L_cls + L_box + L_giou | ⚠️ Adapted for CNN |
| **Total Loss** | λ_match·L_match + λ_gt·L_gt | LAMBDA_GT·L_gt + LAMBDA_MATCH·L_match | ✅ Matches |

**Notes on loss adaptation**:
- L_obj (proposal alignment): Uses L1 loss on normalized box coordinates instead of the full MaskDINO loss suite. This is appropriate for a box-only detector.
- L_gt (ground truth supervision): Uses classification BCE, L1 box, and GIoU losses. Mask-related losses (L_dice, L_mask) are omitted since this implementation doesn't predict masks.

### Default Hyperparameters

| Parameter | Original Paper | This Implementation |
|-----------|---------------|---------------------|
| LAMBDA_GT | 1.0 | 1.0 |
| LAMBDA_OBJ | 1.0 | 1.0 |
| LAMBDA_SIM | 1.0 | 1.0 |
| LAMBDA_MATCH | 1.0 | 1.0 |
| Image size | 800 | 800 |
| Temperature (L_sim) | Not specified | 0.07 |

### Object Feature Matching

| Aspect | Original Paper | This Implementation | Status |
|--------|---------------|---------------------|--------|
| Proposal source | CutLER | CutLER | ✅ Matches |
| Matching method | IoU-based | IoU-based (greedy) | ✅ Matches |
| IoU threshold | Not specified | 0.3 | - |

## Implementation Notes

### Corrections Applied

1. **VOC Class Names**: Fixed to use the correct 20 PASCAL VOC categories in COCO naming conventions:
   - Removed incorrect classes: `truck`, `elephant`, `bear`, `zebra`, `giraffe`
   - Added missing VOC classes: `bottle`, `chair`, `couch`, `potted plant`, `dining table`

2. **GT Supervision**: Per paper, L_gt is computed on transformed branch queries. The student (transformed branch) receives GT supervision on all processed views.

3. **Temperature Scaling**: Added temperature scaling parameter (τ=0.07) to the cosine similarity loss for better gradient dynamics.

4. **NaN-safe Loss Computation**: Added checks to handle non-finite loss terms gracefully.

### Key Adaptations for CNN vs Transformer

1. **Query Features**: Instead of transformer decoder object queries, this implementation extracts "query features" from the CNN feature map at each spatial location. Each (x, y) position in the feature map acts as a query.

2. **Dense Prediction**: The ConvNeXt backbone produces a feature map of size [B, C, Hf, Wf]. Each spatial position predicts:
   - A classification score (objectness)
   - A bounding box (center-based cxcywh format)

3. **Feature Matching**: For L_sim computation, features are matched based on IoU between predicted boxes and CutLER proposals, similar to the paper's approach but using dense predictions instead of sparse queries.

### Known Limitations

1. **No Mask Prediction**: The original v-CLR paper includes instance segmentation masks. This implementation only produces bounding boxes.

2. **Dense vs Query-based**: The paper uses transformer object queries (DETR-style, typically 300 queries), while this implementation uses dense CNN predictions (Hf × Wf positions, e.g., 25×25 = 625 for 800px input).

3. **No Hungarian Matching**: The paper likely uses Hungarian matching for loss computation. This implementation uses simpler greedy IoU-based matching.

## Purpose

This implementation serves to:
1. **Compare architectures**: Evaluate CNN vs Transformer performance on the v-CLR methodology
2. **Validate the approach**: Confirm that view-consistent learning transfers to CNN-based detectors
3. **Provide a baseline**: Enable fair comparison between architectural choices while keeping the training methodology consistent

## Requirements

- PyTorch with CUDA
- torchvision
- pycocotools
- COCO dataset with depth and stylized views

## Usage

Run the Jupyter notebook `vclr_convnext_rev2_teach_stud_1epoch.ipynb` to train the model.

## Evaluation Metrics

The implementation uses the same evaluation protocol as the paper:
- **AR^b_k**: Average Recall for bounding boxes at k proposals
- **AR^m_k**: Average Recall for masks at k proposals (using rectangular proxy masks)
- Evaluation on Non-VOC categories (UVO subset) for open-world performance
