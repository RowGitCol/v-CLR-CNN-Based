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

### Core Principle

Neural networks are biased toward learning appearance information (e.g., texture) to differentiate objects, which inhibits generalization to novel classes with unseen textures. v-CLR overcomes this by learning **appearance-invariant representations** that complement appearance information, making the model generalizable and unbiased during inference.

**The key to this learning framework is to enforce representation consistency by maximizing query feature similarity between transformed views and the natural image.**

### Two-Branch Architecture

Per the paper, v-CLR uses a two-branch architecture:

1. **Natural Image Branch (Teacher)**: Always receives natural images as input. Updated as an exponential moving average (EMA) of the transformed image branch to prevent feature collapsing (following self-supervised learning frameworks).

2. **Transformed Image Branch (Student)**: Randomly processes any of the transformed images OR the original natural image with equal probability.

Both branches use adapted DETR transformer architectures to make sets of predictions, where **each prediction consists of**:
- A classification score
- A predicted bounding box
- A predicted segmentation mask

### Appearance-Invariant Transformation

The paper leverages off-the-shelf image transformations to overwrite appearance while preserving structural content. This circumvents texture bias by enabling the model to learn consistent, transferable representations across different views:

- **Primary transformation**: Colorized depth maps
- **Auxiliary transformations**: Art-stylizing, edge maps
- **Three views per sample**: Natural image, colorized depth map, one auxiliary view
- **View selection**: One view randomly selected per sample with equal probability during training
- **Additional augmentation**: Random cropping and resizing of image patches, integrated with original image to further destroy object appearance

### Object-centric Learning by Object Proposals

High similarity between matched queries does not necessarily mean the model learns informative representations—models can capture shortcut solutions where representations are irrelevant to objects. In open-world learning, lack of correlation with objects causes failure in generalization.

**Solution**: Leverage large-scale pre-trained instance detectors (e.g., CutLER) to provide **object proposals**. These proposals serve as a medium to match object-related queries from both branches, ensuring the learning framework learns meaningful object-oriented representations transferable to open-world settings.

### Object Feature Matching

The matching process forms one-to-one triplets between proposals and predictions from both branches:

**Notation**:
- **P₁, P₂**: Prediction sets from natural and transformed branches
- **Pₒ**: Object proposals from pre-trained detector
- Each set P = {(p̂ᵢ, b̂ᵢ, m̂ᵢ)} contains tuples of:
  - p̂ᵢ: Classification score
  - b̂ᵢ: Bounding box
  - m̂ᵢ: Segmentation mask
- **Q₁, Q₂**: Query feature sets associated with P₁, P₂ (|Qᵢ| = |Pᵢ|)

**Matching Process**:
1. For each proposal in Pₒ, find optimal matched predictions P̂₁ and P̂₂ by minimizing matching cost
2. Form N̄ one-to-one triplets: (Pₒ, P̂₁, P̂₂)

### Loss Functions (from paper)

The training objectives ensure queries capture **object-oriented appearance-invariant representations**:

| Loss | Formula | Description |
|------|---------|-------------|
| **L_sim** | (1/N̄) Σ (1 - cos(q1, q2)) | Cosine similarity between matched queries Q̂₁ and Q̂₂ |
| **L_obj** | λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_GIoU | Supervision from object proposals (assumed reliably object-related) |
| **L_match** | λ_obj·L_obj + λ_sim·L_sim | Total matching objective |
| **L_gt** | Same as L_obj but using ground truth G instead of proposals Pₒ | Ground truth supervision on transformed branch queries Q̂₂ |
| **Total** | L = λ_match·L_match + λ_gt·L_gt | Combined training objective |

**Training Flow**:
1. **Matching objective (L_match)**: Ensures queries capture object-oriented appearance-invariant representations by (a) enforcing similarity between matched queries across views and (b) supervising predictions against object proposals
2. **Ground truth objective (L_gt)**: Standard segmentation loss using ground truth labels on the optimized transformed image queries Q̂₂

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

## Detailed Code-to-Paper Analysis

Based on the complete v-CLR paper methodology, here is a detailed analysis of how this CNN implementation differs from the paper:

### ✅ What the Code Implements Correctly

| Paper Component | Code Implementation | Status |
|-----------------|---------------------|--------|
| **Two-branch architecture** | `TeacherStudentVCLR` class with `student` and `teacher` | ✅ Correct |
| **EMA teacher updates** | `update_teacher()` with momentum 0.999 | ✅ Correct |
| **Natural image branch** | Teacher always processes natural images | ✅ Correct |
| **Transformed image branch** | Student processes depth/stylized views | ✅ Correct |
| **View selection** | Random selection between depth and style per batch | ✅ Correct |
| **L_sim formula** | `cosine_sim_loss()`: (1 - cos(q1, q2)) averaged | ✅ Correct |
| **L_obj components** | L1 box alignment with proposals | ⚠️ Partial (missing dice/mask) |
| **L_match structure** | `λ_obj·L_obj + λ_sim·L_sim` | ✅ Correct |
| **L_gt supervision** | GT loss on student's transformed branch | ✅ Correct |
| **Total loss** | `L = λ_match·L_match + λ_gt·L_gt` | ✅ Correct |
| **CutLER proposals** | Loaded from `vCLR_coco_train2017_top5.json` | ✅ Correct |
| **Object feature matching** | Greedy IoU matching forms triplets | ⚠️ Simplified |

### ❌ Key Differences from Paper

| Paper Component | Paper Description | Code Implementation | Impact |
|-----------------|-------------------|---------------------|--------|
| **Architecture** | DeformableDETR/DINO-DETR transformer | ConvNeXt-tiny CNN | **Major**: Different inductive biases |
| **Object queries** | 300 learnable queries → prototypes | Dense per-pixel predictions (~625) | **Major**: Changes object representation |
| **Mask prediction** | Prototype-pyramid similarity maps | None (bounding boxes only) | **Major**: Missing L_dice, L_mask |
| **Matching algorithm** | Hungarian optimal bipartite matching | Greedy IoU-based matching | **Medium**: Suboptimal assignments |
| **L_obj components** | λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_GIoU | Only L1 box alignment | **Major**: Missing 4 of 5 components |
| **Query features** | Transformer decoder query embeddings | CNN spatial feature vectors | **Medium**: Different feature semantics |

### Detailed Loss Function Comparison

**Paper's L_obj** (Equation 1):
```
L_obj = λ1·L_dice + λ2·L_mask + λ3·L_score + λ4·L_box + λ5·L_GIoU
     = 5.0·L_dice + 5.0·L_mask + 4.0·L_score + 5.0·L_box + 2.0·L_GIoU
```

**Code's L_obj** (in `compute_vclr_losses`):
```python
L_obj = F.l1_loss(sel_preds_norm, sel_props_norm)  # Only box L1 alignment
```

**Missing components**:
- ❌ `L_dice`: Dice loss for mask prediction
- ❌ `L_mask`: BCE/focal loss for mask prediction  
- ❌ `L_score`: Classification score loss against proposals
- ❌ `L_giou`: GIoU loss for box regression (only in L_gt, not L_obj)

### View Selection Comparison

**Paper**: Randomly selects ONE view per sample with equal probability:
- 1/3 natural image
- 1/3 colorized depth map
- 1/3 auxiliary view (art-stylized or edge)

**Code**: Randomly selects ONE auxiliary view per batch:
```python
if has_depth and has_style:
    if random.random() < 0.5:
        use_depth, use_style = True, False
    else:
        use_depth, use_style = False, True
```
Note: The code doesn't include the case where the transformed branch receives the natural image (1/3 probability in paper).

### Matching Process Comparison

**Paper** (Object Feature Matching):
1. For each proposal in Pₒ, find optimal matched predictions P̂₁ and P̂₂ by minimizing the matching cost (combination of classification, box, and mask losses)
2. Uses Hungarian algorithm for optimal bipartite matching
3. Forms N̄ one-to-one triplets: (Pₒ, P̂₁, P̂₂)

**Code** (`match_proposals_to_predictions`):
```python
ious = box_iou(proposal_boxes, pred_boxes)
best_iou, best_idx = ious.max(dim=1)  # Greedy: best pred for each proposal
keep = best_iou > iou_thresh
```
- Greedy matching instead of Hungarian
- May assign same prediction to multiple proposals
- Truncated to `max_pairs=32` matches

### Summary of Alignment

| Aspect | Alignment | Notes |
|--------|-----------|-------|
| **Core v-CLR principle** | ✅ High | Appearance-invariant learning via view consistency |
| **Two-branch architecture** | ✅ High | Teacher-student with EMA |
| **Loss structure** | ⚠️ Medium | Correct formula, missing components |
| **Feature representation** | ❌ Low | CNN vs transformer queries |
| **Mask prediction** | ❌ None | Bounding boxes only |
| **Matching algorithm** | ⚠️ Medium | Greedy vs Hungarian |

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
