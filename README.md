# v-CLR-CNN-Based

This is an attempt at recreating the "v-CLR: View-Consistent Learning for Open-World Instance Segmentation" paper ([arXiv:2504.01383](https://arxiv.org/abs/2504.01383)) using a CNN (ConvNeXt) instead of a transformer.

## Overview

The v-CLR approach uses view-consistent learning to improve open-world instance segmentation. This implementation adapts the methodology to use a ConvNeXt-tiny backbone instead of the original DeformableDETR/DINO-DETR transformer architecture.

## Detailed Comparison: Paper vs. CNN Implementation

| Component | Original Paper (Transformer) | This Implementation (CNN) |
|-----------|------------------------------|---------------------------|
| **Backbone** | DeformableDETR / DINO-DETR | ConvNeXt-tiny |
| **Detection Head** | Transformer decoder with prototypes | Dense convolutional heads |
| **Object Representation** | Learnable object queries → prototypes | Dense per-pixel predictions |
| **Mask Generation** | Prototype-pyramid similarity maps | No (bounding boxes only) |
| **Query Matching** | Hungarian matching (bipartite) | Greedy IoU-based matching |
| **Feature Extraction** | Transformer decoder queries | CNN feature map spatial locations |
| **Attention Mechanism** | Multi-head self/cross attention | Convolutional layers only |
| **Positional Encoding** | Sinusoidal/learned positional embeddings | Implicit (CNN receptive fields) |
| **Model Parameters** | ~50M | ~28M (ConvNeXt-tiny) |
| **Teacher-Student** | EMA teacher on natural branch | EMA teacher on natural branch ✓ |

## Key Differences from Original Paper

### 1. Architecture
- **Paper**: Decorates DeformableDETR and DINO-DETR into an instance segmentation model. Each query predicts a prototype for a corresponding instance, and the model predicts instance segmentation maps by computing similarity between output prototypes and pyramid features of the transformer encoder.
- **CNN**: Uses ConvNeXt-tiny backbone with dense prediction heads. Each spatial location in the feature map acts as an implicit "query".

### 2. Object Query System
- **Paper**: Learnable object queries pass through transformer decoder layers. Each query predicts a prototype for an instance, with masks generated via prototype-pyramid feature similarity.
- **CNN**: Dense predictions at every spatial location (e.g., 25×25 = 625 predictions for 800px input). No explicit queries—predictions are spatially anchored.

### 3. Query Matching Strategy
- **Paper**: Hungarian algorithm for optimal bipartite matching between predictions and ground truth/proposals.
- **CNN**: Greedy IoU-based matching for computational efficiency. May produce suboptimal assignments.

### 4. Instance Segmentation
- **Paper**: Prototype-based instance segmentation. Each query predicts a prototype, and the final mask is computed by measuring similarity between the prototype and pyramid features from the transformer encoder.
- **CNN**: Bounding box detection only. Proxy rectangular masks used for segmentation metrics.

### 5. Feature Similarity (L_sim)
- **Paper**: Computes cosine similarity between matched transformer query embeddings from natural and transformed views.
- **CNN**: Computes cosine similarity between CNN feature vectors at matched spatial locations.

### 6. Cross-View Consistency
- **Paper**: Object queries provide view-invariant representations through learned attention patterns.
- **CNN**: Relies on ConvNeXt's translation-equivariant features. Consistency is enforced through L_sim loss on matched detections.

## Paper's Methodology (v-CLR)

Per the paper, v-CLR uses a two-branch architecture:

1. **Natural Image Branch (Teacher)**: Always receives natural images, updated as EMA of transformed branch
2. **Transformed Image Branch (Student)**: Randomly processes transformed views OR natural image with equal probability

### Appearance-Invariant Transformation

The paper leverages off-the-shelf image transformations to overwrite appearance while preserving structural content. This circumvents texture bias by enabling the model to learn consistent, transferable representations across different views:

- **Primary transformation**: Colorized depth maps
- **Auxiliary transformations**: Art-stylizing, edge maps
- **Three views per sample**: Natural image, colorized depth map, one auxiliary view
- **View selection**: One view randomly selected per sample with equal probability during training
- **Additional augmentation**: Random cropping and resizing of image patches, integrated with original image to further destroy object appearance

### Loss Functions (from paper)

| Loss | Formula | Description |
|------|---------|-------------|
| **L_sim** | (1/N̄) Σ (1 - cos(q1, q2)) | Cosine similarity between matched queries |
| **L_obj** | λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_GIoU | Supervision from CutLER proposals |
| **L_match** | λ_obj·L_obj + λ_sim·L_sim | Matching objective |
| **L_gt** | Same as L_obj but using ground truth G on transformed branch queries Q̂_2 | Ground truth supervision |
| **Total** | L = λ_match·L_match + λ_gt·L_gt | Combined loss |

### Loss Weights (from paper)
- λ1 (dice) = 5.0
- λ2 (mask BCE) = 5.0
- λ3 (score/class) = 4.0
- λ4 (box L1) = 5.0
- λ5 (GIoU) = 2.0
- λ_obj = 1.0
- λ_sim = 1.0
- λ_match = 1.0
- λ_gt = 1.0

## Implementation Notes

### What's Preserved from the Paper

1. **Two-Branch Architecture**: Teacher-student setup with EMA updates ✓
2. **View Consistency Learning**: L_sim loss enforces feature consistency across views ✓
3. **CutLER Proposals**: Uses CutLER pseudo-labels for unsupervised object discovery ✓
4. **Multi-View Training**: Depth and stylized views used as transformed inputs ✓
5. **VOC→Non-VOC Evaluation**: Known classes (VOC 20) for training, evaluated on novel categories ✓
6. **Loss Structure**: L_total = L_gt + L_obj + L_sim follows paper's formulation ✓

### Corrections Applied

1. **VOC Class Names**: Fixed to use the correct 20 PASCAL VOC categories in COCO naming conventions:
   - Removed incorrect classes: `truck`, `elephant`, `bear`, `zebra`, `giraffe`
   - Added missing VOC classes: `bottle`, `chair`, `couch`, `potted plant`, `dining table`

2. **GT Supervision**: Per paper, L_gt is computed on transformed branch queries. The student (transformed branch) receives GT supervision on all processed views.

3. **Temperature Scaling**: Added temperature scaling parameter to the cosine similarity loss.

### Known Limitations

1. **No Mask Prediction**: The original v-CLR paper includes instance segmentation masks. This implementation only produces bounding boxes and uses rectangular proxy masks for AR^m metrics.

2. **Dense vs Query-based**: The paper uses transformer object queries (DETR-style), while this implementation uses dense CNN predictions. This affects:
   - How objects are represented
   - How matching is performed
   - The inductive biases of the model

3. **No Hungarian Matching**: Greedy matching may produce suboptimal assignments compared to the optimal Hungarian algorithm.

4. **Smaller Capacity**: ConvNeXt-tiny (~28M params) is smaller than typical MaskDINO configurations (~50M+ params).

## Expected Performance Differences

Due to architectural differences, the CNN implementation may show:

| Aspect | Expected Behavior |
|--------|-------------------|
| **AR^b (box recall)** | Comparable or slightly lower |
| **AR^m (mask recall)** | Lower (no true mask prediction) |
| **Training Speed** | Faster (CNNs are more efficient) |
| **Memory Usage** | Lower (no attention matrices) |
| **Generalization** | May differ (different inductive biases) |

## Requirements

- PyTorch with CUDA
- torchvision
- pycocotools
- COCO dataset with depth and stylized views

## Usage

Run the Jupyter notebook `vclr_convnext_rev2_teach_stud_1epoch.ipynb` to train the model.

## References

- v-CLR Paper: [arXiv:2504.01383](https://arxiv.org/abs/2504.01383)
- DeformableDETR: [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
- DINO-DETR: [DINO: DETR with Improved DeNoising Anchor Boxes](https://arxiv.org/abs/2203.03605)
- ConvNeXt: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- CutLER: [Cut and Learn for Unsupervised Object Detection and Instance Segmentation](https://arxiv.org/abs/2301.11320)
