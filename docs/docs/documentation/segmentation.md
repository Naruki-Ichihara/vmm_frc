---
sidebar_position: 2
title: Segmentation and Fiber Detection
---

# Segmentation and Fiber Center Detection

This document explains the segmentation methods and fiber center detection algorithms used in VMM-FRC.

## Overview

Fiber center detection from CT images involves two main steps:
1. **Segmentation**: Separating fiber regions from the matrix using thresholding
2. **Center Detection**: Identifying individual fiber centers using watershed segmentation

## Otsu's Thresholding Method

### Mathematical Foundation

Otsu's method [[1]](#references) automatically determines the optimal threshold value by maximizing the between-class variance. Given a grayscale image with $L$ gray levels, the algorithm finds the threshold $t^*$ that maximizes:

$$
\sigma_B^2(t) = \omega_0(t) \omega_1(t) [\mu_0(t) - \mu_1(t)]^2
$$

where:
- $\omega_0(t)$ and $\omega_1(t)$ are the probabilities of the two classes (background and foreground)
- $\mu_0(t)$ and $\mu_1(t)$ are the mean intensities of each class

The class probabilities are computed as:

$$
\omega_0(t) = \sum_{i=0}^{t} p(i), \quad \omega_1(t) = \sum_{i=t+1}^{L-1} p(i)
$$

where $p(i)$ is the probability of gray level $i$.

### Implementation

```python
from vmm.segment import threshold_otsu

# Apply Otsu's thresholding
binary, threshold = threshold_otsu(image)

print(f"Computed threshold: {threshold}")
print(f"Fiber pixels: {binary.sum()}")
```

## Fiber Center Detection Pipeline

The fiber center detection algorithm uses a multi-step approach:

### Step 1: Binary Segmentation

Apply Otsu's thresholding to separate fibers from matrix:

$$
B(x, y) = \begin{cases}
1 & \text{if } I(x, y) > t^* \\
0 & \text{otherwise}
\end{cases}
$$

### Step 2: Distance Transform

Compute the Euclidean distance transform of the binary image:

$$
D(x, y) = \min_{(x', y') \in \text{background}} \sqrt{(x - x')^2 + (y - y')^2}
$$

The distance transform assigns each fiber pixel the distance to the nearest background pixel. Fiber centers will have local maxima in this transform.

### Step 3: Local Maxima Detection

Find local maxima in the distance transform using `peak_local_max`. These points serve as markers for watershed segmentation:

$$
\text{markers} = \{(x, y) : D(x, y) > D(x', y') \text{ for all } (x', y') \in N(x, y)\}
$$

where $N(x, y)$ is a neighborhood of minimum distance `min_distance`.

### Step 4: Watershed Segmentation

Apply watershed segmentation using the detected markers on the inverted distance transform:

$$
L = \text{watershed}(-D, \text{markers}, \text{mask}=B)
$$

The watershed algorithm "floods" from each marker, creating labeled regions that correspond to individual fibers.

### Step 5: Centroid Calculation

For each labeled region, compute the centroid and equivalent diameter:

$$
\bar{x} = \frac{1}{A} \sum_{(x,y) \in R} x, \quad \bar{y} = \frac{1}{A} \sum_{(x,y) \in R} y
$$

$$
d_{eq} = 2\sqrt{\frac{A}{\pi}}
$$

where $A$ is the region area and $R$ is the set of pixels in the region.

## Implementation

### Basic Usage

```python
from vmm.fiber_trajectory import detect_fiber_centers

# Detect fiber centers
centers, diameters = detect_fiber_centers(
    image,
    min_diameter=5.0,   # Minimum fiber diameter (pixels)
    max_diameter=20.0,  # Maximum fiber diameter (pixels)
    min_distance=5      # Minimum distance between detected peaks
)

print(f"Detected {len(centers)} fibers")
print(f"Mean diameter: {diameters.mean():.2f} pixels")
```

### With Label Output

```python
# Get watershed labels for visualization
centers, diameters, labels = detect_fiber_centers(
    image,
    min_diameter=5.0,
    max_diameter=20.0,
    return_labels=True
)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')

axes[1].imshow(labels, cmap='nipy_spectral')
axes[1].set_title('Watershed Labels')

axes[2].imshow(image, cmap='gray')
axes[2].scatter(centers[:, 0], centers[:, 1], c='red', s=10)
axes[2].set_title('Detected Centers')

plt.tight_layout()
plt.show()
```

### Alternative Thresholding

For images where Otsu's method does not perform well, use percentile-based thresholding:

```python
from vmm.segment import threshold_percentile

# Use 70th percentile as threshold
binary, threshold = threshold_percentile(image, percentile=70.0)

# Or directly in detect_fiber_centers
centers, diameters = detect_fiber_centers(
    image,
    min_diameter=5.0,
    max_diameter=20.0,
    threshold_percentile=70.0  # Use percentile instead of Otsu
)
```

## Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from vmm.io import import_image_sequence
from vmm.fiber_trajectory import detect_fiber_centers
from vmm.segment import threshold_otsu

# 1. Load CT data
volume = import_image_sequence(
    path_template="data/ct_scan/slice",
    number_of_images=100,
    number_of_digits=4,
    format="tif"
)

# 2. Select a slice for analysis
slice_idx = 50
image = volume[slice_idx]

# 3. Detect fiber centers
centers, diameters, labels = detect_fiber_centers(
    image,
    min_diameter=5.0,
    max_diameter=15.0,
    min_distance=3,
    return_labels=True
)

# 4. Analyze results
print(f"Number of fibers detected: {len(centers)}")
print(f"Diameter statistics:")
print(f"  Mean: {diameters.mean():.2f} pixels")
print(f"  Std:  {diameters.std():.2f} pixels")
print(f"  Min:  {diameters.min():.2f} pixels")
print(f"  Max:  {diameters.max():.2f} pixels")

# 5. Estimate volume fraction from detected fibers
total_fiber_area = np.sum(np.pi * (diameters / 2) ** 2)
image_area = image.shape[0] * image.shape[1]
vf_estimated = total_fiber_area / image_area
print(f"Estimated Vf: {vf_estimated:.3f}")

# 6. Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original CT Image')

# Binary segmentation
binary, _ = threshold_otsu(image)
axes[0, 1].imshow(binary, cmap='gray')
axes[0, 1].set_title('Otsu Segmentation')

# Watershed labels
axes[1, 0].imshow(labels, cmap='nipy_spectral')
axes[1, 0].set_title(f'Watershed Labels ({len(centers)} fibers)')

# Detected centers with diameter circles
axes[1, 1].imshow(image, cmap='gray')
for (x, y), d in zip(centers, diameters):
    circle = plt.Circle((x, y), d/2, fill=False, color='red', linewidth=0.5)
    axes[1, 1].add_patch(circle)
axes[1, 1].set_title('Detected Fibers')

plt.tight_layout()
plt.show()
```

## Parameter Selection

### min_diameter / max_diameter

These parameters filter detected regions by equivalent diameter:
- Set based on expected fiber diameter in your images
- Too narrow a range may miss valid fibers
- Too wide a range may include noise or merged fibers

### min_distance

Controls the minimum distance between detected peaks:
- Should be approximately half the expected fiber diameter
- Too small: may detect multiple peaks per fiber
- Too large: may miss closely packed fibers

## References

1. Otsu, N. (1979). A Threshold Selection Method from Gray-Level Histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66. [https://doi.org/10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)
