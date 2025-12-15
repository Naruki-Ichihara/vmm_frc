---
sidebar_position: 1
title: Orientation Analysis
---

# Fiber Orientation Analysis

This document explains the mathematical foundation and implementation of fiber orientation analysis using the structure tensor method.

## Overview

Fiber orientation analysis is essential for understanding the microstructure of fiber-reinforced composites. The structure tensor method provides a robust approach to extract local fiber orientations from 3D CT scan data [[1]](#references).

## Mathematical Foundation

### Structure Tensor

The structure tensor (also known as the second moment matrix) captures local orientation information by analyzing intensity gradients. For a 3D grayscale image $I(x, y, z)$, the structure tensor $\mathbf{S}$ is defined as:

$$
\mathbf{S} = \begin{pmatrix}
S_{xx} & S_{xy} & S_{xz} \\
S_{xy} & S_{yy} & S_{yz} \\
S_{xz} & S_{yz} & S_{zz}
\end{pmatrix}
$$

where each component is computed from the image gradients:

$$
S_{ij} = G_\sigma * (\nabla_i I \cdot \nabla_j I)
$$

Here:
- $\nabla_i I$ is the partial derivative of image intensity in direction $i$
- $G_\sigma$ is a Gaussian smoothing kernel with scale $\sigma$
- $*$ denotes convolution

### Gradient Computation

In the VMM-FRC implementation (via scikit-image), the gradients are computed using the **Sobel operator**, a discrete differentiation filter:

$$
\mathbf{K}_x = \begin{pmatrix}
-1 & 0 & +1 \\
-2 & 0 & +2 \\
-1 & 0 & +1
\end{pmatrix}, \quad
\mathbf{K}_y = \begin{pmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
+1 & +2 & +1
\end{pmatrix}
$$

The gradient in each direction is computed as:

$$
\nabla_i I = \mathbf{K}_i * I
$$

For 3D volumes, the Sobel operator is extended to 3×3×3 kernels for each axis.

### Two-Step Process

The structure tensor computation involves two steps:

1. **Gradient computation**: Sobel filters compute derivatives $\nabla_i I$
2. **Outer product smoothing**: Gaussian filter $G_\sigma$ smooths the gradient products

$$
S_{ij} = G_\sigma * (\nabla_i I \cdot \nabla_j I)
$$

The `sigma` parameter controls only the **second step** (smoothing of gradient products), not the gradient computation itself. This smoothing integrates gradient information over a local neighborhood, providing more robust orientation estimates.

### Physical Interpretation

The structure tensor encodes directional information about intensity variations:

- **Large eigenvalue**: Direction of maximum intensity change (perpendicular to fiber axis)
- **Small eigenvalue**: Direction of minimum intensity change (along fiber axis)

For cylindrical fibers in CT images:
- Intensity varies rapidly perpendicular to the fiber axis
- Intensity remains relatively constant along the fiber axis

### Eigenvalue Decomposition

The structure tensor is a symmetric positive semi-definite matrix. Its eigenvalue decomposition yields:

$$
\mathbf{S} \mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad i = 1, 2, 3
$$

where $\lambda_1 \geq \lambda_2 \geq \lambda_3 \geq 0$ are the eigenvalues and $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$ are the corresponding orthonormal eigenvectors.

**The fiber direction corresponds to the eigenvector $\mathbf{v}_3$ associated with the smallest eigenvalue $\lambda_3$.**

### Orientation Angles

#### Spherical Coordinates

The fiber direction vector $\mathbf{v} = (v_x, v_y, v_z)$ can be expressed in spherical coordinates:

**Azimuthal angle (θ):**
$$
\theta = \arctan2(v_z, v_x)
$$

**Elevation angle (φ):**
$$
\phi = \arctan2(v_y, v_x)
$$

#### Misalignment Angle

When a reference direction $\mathbf{r}$ is specified (e.g., the nominal fiber axis), the misalignment angle $\alpha$ is computed as:

$$
\alpha = \arccos(\mathbf{v} \cdot \mathbf{r})
$$

This gives the angle between the local fiber direction and the reference axis, ranging from $0°$ to $180°$.

## Implementation

### Computing the Structure Tensor

```python
import numpy as np
from vmm.io import import_image_sequence
from vmm.analysis import compute_structure_tensor, compute_orientation

# Load CT volume
volume = import_image_sequence(
    path_template="path/to/slice",
    number_of_images=100,
    number_of_digits=4,
    format="tif"
)

# Compute structure tensor
# noise_scale controls Gaussian smoothing (larger = more smoothing)
tensor = compute_structure_tensor(volume, noise_scale=2)

print(f"Input volume shape: {volume.shape}")
print(f"Structure tensor shape: {tensor.shape}")
# tensor shape is (6, depth, height, width)
# Components: [S_xx, S_xy, S_xz, S_yy, S_yz, S_zz]
```

### Extracting Orientation Angles

#### Without Reference Vector

Returns spherical angles (θ, φ) for each voxel:

```python
# Compute orientation without reference
theta, phi = compute_orientation(tensor)

print(f"Theta range: [{theta.min():.1f}, {theta.max():.1f}] degrees")
print(f"Phi range: [{phi.min():.1f}, {phi.max():.1f}] degrees")
```

#### With Reference Vector

Returns misalignment angle relative to a reference direction:

```python
# Define reference direction (e.g., Z-axis for UD composites)
reference_vector = [0, 0, 1]  # [x, y, z]

# Compute misalignment angle
misalignment = compute_orientation(tensor, reference_vector=reference_vector)

print(f"Mean misalignment: {misalignment.mean():.2f} degrees")
print(f"Std misalignment: {misalignment.std():.2f} degrees")
```

### Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from vmm.io import import_image_sequence
from vmm.analysis import compute_structure_tensor, compute_orientation, drop_edges_3D

# 1. Load CT data
volume = import_image_sequence(
    path_template="data/ct_scan/slice",
    number_of_images=200,
    number_of_digits=4,
    format="tif"
)

# 2. Compute structure tensor with noise reduction
tensor = compute_structure_tensor(volume, noise_scale=2)

# 3. Define fiber reference direction (Z-axis)
reference = [0, 0, 1]

# 4. Compute misalignment angles
orientation = compute_orientation(tensor, reference_vector=reference)

# 5. Remove edge artifacts
orientation_cropped = drop_edges_3D(10, orientation)

# 6. Analyze results
print(f"Orientation statistics:")
print(f"  Mean: {orientation_cropped.mean():.2f}°")
print(f"  Std:  {orientation_cropped.std():.2f}°")
print(f"  Max:  {orientation_cropped.max():.2f}°")

# 7. Visualize histogram
plt.figure(figsize=(10, 6))
plt.hist(orientation_cropped.ravel(), bins=100, density=True, alpha=0.7)
plt.xlabel('Misalignment Angle (degrees)')
plt.ylabel('Probability Density')
plt.title('Fiber Orientation Distribution')
plt.xlim(0, 30)
plt.grid(True, alpha=0.3)
plt.show()
```

## Parameter Selection

### Noise Scale (σ)

The `noise_scale` parameter controls Gaussian smoothing of the gradient products. This parameter should be chosen based on the feature size (e.g., fiber diameter) in the image. If `noise_scale` is too small, high-frequency noise will appear in the orientation results.

### Edge Artifacts

Structure tensor computation produces artifacts near volume boundaries due to gradient estimation. Use `drop_edges_3D()` to remove these:

```python
# Remove 10 pixels from each edge
orientation_clean = drop_edges_3D(10, orientation)
```

## References

1. Jeppesen, N., Mikkelsen, L.P., Dahl, A.B., Christensen, A.N., & Dahl, V.A. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis. *Composites Part A: Applied Science and Manufacturing*, 149, 106541. [https://doi.org/10.1016/j.compositesa.2021.106541](https://doi.org/10.1016/j.compositesa.2021.106541)
