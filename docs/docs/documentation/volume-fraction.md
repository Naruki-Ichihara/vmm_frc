---
sidebar_position: 4
title: Volume Fraction Estimation
---

# Fiber Volume Fraction Estimation

This document explains the methods for estimating fiber volume fraction (Vf) from segmented CT images.

## Overview

Fiber volume fraction is a critical parameter in composite materials, defined as the ratio of fiber volume to total composite volume:

$$
V_f = \frac{V_{\text{fiber}}}{V_{\text{total}}}
$$

For CT images, this is computed from binary segmentation as the ratio of fiber pixels to total pixels.

## Global Volume Fraction

The simplest approach computes a single Vf value for the entire image or volume:

$$
V_f^{\text{global}} = \frac{N_{\text{fiber}}}{N_{\text{total}}}
$$

where $N_{\text{fiber}}$ is the number of fiber pixels and $N_{\text{total}}$ is the total number of pixels.

```python
from vmm.segment import threshold_otsu

# Segment image
binary, threshold = threshold_otsu(image)

# Compute global Vf
global_vf = binary.mean()
print(f"Global Vf: {global_vf:.3f}")
```

## Local Volume Fraction

Local Vf analysis reveals spatial variations in fiber distribution, which is important for understanding material homogeneity.

### Box Averaging Method

The local Vf at position $(x, y, z)$ is computed as the average within a cubic window:

$$
V_f(x, y, z) = \frac{1}{W^3} \sum_{i,j,k \in \text{window}} B(x+i, y+j, z+k)
$$

where $W$ is the window size and $B$ is the binary segmentation.

```python
from vmm.segment import estimate_local_vf

# Compute local Vf with 50-pixel window
vf_map = estimate_local_vf(binary, window_size=50)

print(f"Vf range: [{vf_map.min():.3f}, {vf_map.max():.3f}]")
```

### Gaussian-Weighted Averaging

For smoother results, use Gaussian-weighted averaging:

$$
V_f(x, y, z) = (G_\sigma * B)(x, y, z)
$$

where $G_\sigma$ is a Gaussian kernel with standard deviation $\sigma$.

```python
# Compute local Vf with Gaussian weighting
vf_map = estimate_local_vf(binary, gaussian_sigma=25.0)
```

## Statistical Analysis

### Vf Distribution

Analyze the distribution of local Vf values:

```python
from vmm.segment import estimate_vf_distribution
import matplotlib.pyplot as plt

# Get Vf distribution statistics
hist, bin_edges, stats = estimate_vf_distribution(
    binary,
    window_size=50,
    bins=50
)

print(f"Vf Statistics:")
print(f"  Global Vf: {stats['global_vf']:.3f}")
print(f"  Mean:      {stats['mean']:.3f}")
print(f"  Std:       {stats['std']:.3f}")
print(f"  Median:    {stats['median']:.3f}")
print(f"  Min:       {stats['min']:.3f}")
print(f"  Max:       {stats['max']:.3f}")

# Plot histogram
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist, width=bin_edges[1]-bin_edges[0], alpha=0.7)
plt.axvline(stats['global_vf'], color='red', linestyle='--', label=f"Global Vf = {stats['global_vf']:.3f}")
plt.xlabel('Local Volume Fraction')
plt.ylabel('Count')
plt.title('Local Vf Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Slice-by-Slice Analysis

Analyze Vf variation through the sample thickness:

```python
from vmm.segment import estimate_vf_slice_by_slice
import matplotlib.pyplot as plt

# Compute Vf per slice along Z-axis
slice_indices, vf_per_slice = estimate_vf_slice_by_slice(
    binary,
    window_size=50,
    axis=0  # Z-axis
)

# Plot
plt.figure(figsize=(12, 5))
plt.plot(slice_indices, vf_per_slice, 'b-', linewidth=1)
plt.axhline(vf_per_slice.mean(), color='red', linestyle='--',
            label=f"Mean Vf = {vf_per_slice.mean():.3f}")
plt.fill_between(slice_indices, vf_per_slice, alpha=0.3)
plt.xlabel('Slice Index (Z)')
plt.ylabel('Volume Fraction')
plt.title('Vf Variation Through Thickness')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from vmm.io import import_image_sequence
from vmm.segment import (
    threshold_otsu,
    estimate_local_vf,
    estimate_vf_distribution,
    estimate_vf_slice_by_slice,
    apply_morphological_cleaning
)

# 1. Load CT data
volume = import_image_sequence(
    path_template="data/ct_scan/slice",
    number_of_images=100,
    number_of_digits=4,
    format="tif"
)

# 2. Segment fibers using Otsu thresholding
binary, threshold = threshold_otsu(volume)

# 3. Clean up segmentation (optional)
binary_clean = apply_morphological_cleaning(binary, opening_size=2, closing_size=2)

# 4. Compute global Vf
global_vf = binary_clean.mean()
print(f"Global Vf: {global_vf:.3f} ({global_vf*100:.1f}%)")

# 5. Compute local Vf map
vf_map = estimate_local_vf(binary_clean, window_size=50)

# 6. Get distribution statistics
hist, bin_edges, stats = estimate_vf_distribution(binary_clean, window_size=50)

print(f"\nLocal Vf Statistics:")
print(f"  Mean:   {stats['mean']:.3f}")
print(f"  Std:    {stats['std']:.3f}")
print(f"  Range:  [{stats['min']:.3f}, {stats['max']:.3f}]")

# 7. Slice-by-slice analysis
slice_idx, vf_slices = estimate_vf_slice_by_slice(binary_clean, axis=0)

# 8. Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Original slice
mid_slice = volume.shape[0] // 2
axes[0, 0].imshow(volume[mid_slice], cmap='gray')
axes[0, 0].set_title('Original CT Image')

# Segmentation
axes[0, 1].imshow(binary_clean[mid_slice], cmap='gray')
axes[0, 1].set_title(f'Segmentation (Global Vf = {global_vf:.3f})')

# Local Vf map
im = axes[1, 0].imshow(vf_map[mid_slice], cmap='viridis', vmin=0, vmax=1)
axes[1, 0].set_title('Local Vf Map')
plt.colorbar(im, ax=axes[1, 0], label='Vf')

# Vf through thickness
axes[1, 1].plot(slice_idx, vf_slices, 'b-')
axes[1, 1].axhline(stats['global_vf'], color='red', linestyle='--')
axes[1, 1].set_xlabel('Slice Index')
axes[1, 1].set_ylabel('Volume Fraction')
axes[1, 1].set_title('Vf Through Thickness')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Parameter Selection

### Window Size

The `window_size` parameter should be chosen based on the feature size in the image. Larger windows produce smoother Vf maps.

### Gaussian Sigma

For Gaussian-weighted averaging, `sigma ≈ window_size / 3` gives similar spatial scale to box averaging.
