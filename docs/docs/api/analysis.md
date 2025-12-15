---
sidebar_position: 3
title: vmm.analysis
---

# vmm.analysis

Structure tensor computation and fiber orientation analysis.

## Functions

### drop_edges_3D

```python
def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray
```

Remove edge pixels from all sides of a 3D volume to eliminate boundary artifacts.

**Args:**
- `width` (int): Number of pixels to remove from each edge.
- `volume` (np.ndarray): Input 3D array with shape (depth, height, width).

**Returns:**
- `np.ndarray`: Cropped 3D volume with shape (depth-2*width, height-2*width, width-2*width).

**Example:**

```python
import numpy as np
from vmm.analysis import drop_edges_3D

vol = np.random.rand(100, 200, 200)
cropped = drop_edges_3D(10, vol)
print(cropped.shape)  # (80, 180, 180)
```

---

### compute_structure_tensor

```python
def compute_structure_tensor(volume: np.ndarray, noise_scale: int, mode: str = 'nearest') -> np.ndarray
```

Compute 3D structure tensor for fiber orientation analysis using gradient-based methods.

The structure tensor is a symmetric 3x3 matrix field that captures local orientation information in 3D volumes. Each voxel gets a tensor describing the predominant direction of intensity gradients in its neighborhood.

**Args:**
- `volume` (np.ndarray): Input 3D grayscale volume (depth, height, width).
- `noise_scale` (int): Gaussian smoothing sigma for noise reduction before gradient computation. Larger values provide more smoothing but reduce spatial resolution.
- `mode` (str, optional): Boundary condition for Gaussian filtering. Options: 'constant', 'edge', 'wrap', 'reflect', 'mirror', 'nearest'. Default is 'nearest'.

**Returns:**
- `np.ndarray`: Structure tensor array with shape (6, depth, height, width). Components represent the upper triangular part of the symmetric 3x3 tensor: [T_xx, T_xy, T_xz, T_yy, T_yz, T_zz].

**Raises:**
- `ValueError`: If the input volume is not exactly 3D.

**Example:**

```python
from vmm.analysis import compute_structure_tensor

tensor = compute_structure_tensor(volume, noise_scale=2)
print(tensor.shape)  # (6, depth, height, width)
```

---

### compute_orientation

```python
def compute_orientation(
    structure_tensor: np.ndarray,
    reference_vector: Optional[List[float]] = None
) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
```

Extract fiber orientation angles from structure tensor using eigenvalue decomposition.

Computes orientation angles by finding the eigenvector corresponding to the smallest eigenvalue of each structure tensor, which represents the direction of minimal intensity variation (i.e., the fiber direction).

**Args:**
- `structure_tensor` (np.ndarray): 4D array with shape (6, depth, height, width) containing the symmetric structure tensor components.
- `reference_vector` (List[float], optional): 3D reference direction vector [x, y, z]. If provided, returns only the angle relative to this reference. If None, returns both theta and phi spherical angles.

**Returns:**
- If `reference_vector` is None: Tuple of (theta, phi) arrays with shape (depth, height, width).
  - `theta`: Azimuthal angle in degrees (-180 to 180).
  - `phi`: Elevation angle in degrees (-90 to 90).
- If `reference_vector` is provided: Single array of angles in degrees (0 to 180) relative to reference.

**Example:**

```python
from vmm.analysis import compute_structure_tensor, compute_orientation

# Compute structure tensor
tensor = compute_structure_tensor(volume, noise_scale=2)

# Get orientation relative to Z-axis
reference_vector = [0, 0, 1]
orientation = compute_orientation(tensor, reference_vector)
print(f"Mean orientation: {orientation.mean():.2f} degrees")
```
