---
sidebar_position: 3
title: InSegt Segmentation
---

# InSegt: Interactive Segmentation

This document explains the InSegt (Interactive Segmentation) method used in VMM-FRC for advanced fiber segmentation.

## Overview

InSegt is a dictionary-based interactive segmentation method that uses Gaussian derivative features and KM-tree (K-Means tree) for efficient image segmentation [[1]](#references). It provides more robust segmentation than simple thresholding, especially for images with varying contrast or noise.

The VMM-FRC implementation is based on [InSegtPy](https://github.com/vedranaa/InSegtpy).

## Method

The InSegt pipeline consists of three main components:

1. **Gaussian Derivative Features**: Multi-scale feature extraction
2. **KM-Tree Dictionary**: Hierarchical k-means clustering of image patches
3. **Label Propagation**: Spreading labels through similar patches

### Gaussian Derivative Features

For each pixel, a 15-dimensional feature vector is computed using Gaussian derivatives up to 4th order:

$$
\mathbf{f}(x, y) = [L, L_x, L_y, L_{xx}, L_{xy}, L_{yy}, L_{xxx}, L_{xxy}, L_{xyy}, L_{yyy}, L_{xxxx}, L_{xxxy}, L_{xxyy}, L_{xyyy}, L_{yyyy}]
$$

where $L$ is the Gaussian-smoothed image and subscripts denote partial derivatives:

$$
L = G_\sigma * I, \quad L_x = \frac{\partial G_\sigma}{\partial x} * I, \quad L_{xx} = \frac{\partial^2 G_\sigma}{\partial x^2} * I, \quad \text{etc.}
$$

Using multiple scales $\sigma \in \{1, 2, 4\}$, this produces a 45-dimensional feature vector per pixel.

### KM-Tree Dictionary

The KM-tree is a hierarchical k-means structure that efficiently clusters image patches:

1. **Patch Extraction**: Extract overlapping patches of size $p \times p$ from feature images
2. **Hierarchical Clustering**: Build a tree with branching factor $k$ and depth $L$
3. **Dictionary Creation**: Each leaf node represents a "visual word"

The tree structure allows fast assignment of new patches to dictionary entries.

### Label Propagation

Given sparse user labels, InSegt propagates them through the dictionary:

1. **Label Collection**: Gather labels from annotated pixels
2. **Dictionary Update**: Accumulate label probabilities in dictionary entries
3. **Probability Propagation**: Assign probabilities to all pixels based on their dictionary entry

This creates a dense probability map from sparse annotations.

## Implementation

### Using InSegt for Fiber Detection

```python
from vmm.fiber_trajectory import detect_fiber_centers_insegt

# Detect fiber centers using InSegt
centers, diameters = detect_fiber_centers_insegt(
    image,
    min_diameter=5.0,
    max_diameter=20.0,
    patch_size=9,
    branching_factor=5,
    number_layers=5,
    sigmas=[1, 2, 4]
)

print(f"Detected {len(centers)} fibers")
```

### Interactive Annotation with InSegtAnnotator

For manual annotation and model training:

```python
from vmm.insegt.annotators.insegtannotator import insegt

# Launch interactive annotator
# Returns probability map and labels
probs, labels = insegt(image)
```

### Using FiberSegmentationModel

For batch processing with a trained model:

```python
from vmm.insegt.fiber_model import FiberSegmentationModel

# Create and train model
model = FiberSegmentationModel(
    sigmas=[1, 2, 4],
    patch_size=9,
    branching_factor=5,
    number_layers=5
)

# Build model from training image
model.build_from_image(training_image)

# Apply to new images
model.set_image(new_image)
probs = model.process(labels)
```

## Parameters

### Gaussian Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sigmas` | [1, 2, 4] | List of Gaussian scales for feature extraction |

Larger sigma values capture coarser features. Multiple scales provide scale-invariant representation.

### KM-Tree

| Parameter | Default | Description |
|-----------|---------|-------------|
| `patch_size` | 9 | Patch size (must be odd) |
| `branching_factor` | 5 | Number of clusters per level |
| `number_layers` | 5 | Tree depth |
| `training_patches` | 30000 | Number of patches for training |

**Dictionary size**: $\text{branching\_factor}^{\text{number\_layers}} = 5^5 = 3125$ entries

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from vmm.io import import_image_sequence
from vmm.fiber_trajectory import detect_fiber_centers, detect_fiber_centers_insegt

# Load CT data
volume = import_image_sequence(
    path_template="data/ct_scan/slice",
    number_of_images=100,
    number_of_digits=4,
    format="tif"
)

# Select a slice
slice_idx = 50
image = volume[slice_idx]

# Compare Otsu vs InSegt detection
centers_otsu, diameters_otsu = detect_fiber_centers(
    image, min_diameter=5.0, max_diameter=15.0
)

centers_insegt, diameters_insegt = detect_fiber_centers_insegt(
    image, min_diameter=5.0, max_diameter=15.0
)

print(f"Otsu detected: {len(centers_otsu)} fibers")
print(f"InSegt detected: {len(centers_insegt)} fibers")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(image, cmap='gray')
axes[0].scatter(centers_otsu[:, 0], centers_otsu[:, 1], c='red', s=5)
axes[0].set_title(f'Otsu ({len(centers_otsu)} fibers)')

axes[1].imshow(image, cmap='gray')
axes[1].scatter(centers_insegt[:, 0], centers_insegt[:, 1], c='blue', s=5)
axes[1].set_title(f'InSegt ({len(centers_insegt)} fibers)')

plt.tight_layout()
plt.show()
```

## When to Use InSegt

InSegt is recommended when:
- Simple thresholding produces poor results
- Image has varying contrast across regions
- Fibers have similar intensity to some matrix regions
- Higher accuracy is required

For most cases with good image quality, Otsu-based detection is faster and sufficient.

## References

1. Emerson, M.J., Jespersen, K.M., Dahl, A.B., Conradsen, K., & Mikkelsen, L.P. (2017). Individual fibre segmentation from 3D X-ray computed tomography for characterising the fibre orientation in unidirectional composite materials. *Composites Part A: Applied Science and Manufacturing*, 97, 83-92. [https://doi.org/10.1016/j.compositesa.2016.12.028](https://doi.org/10.1016/j.compositesa.2016.12.028)
