---
sidebar_position: 4
title: vmm.segment
---

# vmm.segment

Segmentation utilities for fiber volume fraction estimation.

## Functions

### estimate_local_vf

```python
def estimate_local_vf(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, ...]] = 50,
    gaussian_sigma: Optional[float] = None,
    normalize: bool = True
) -> np.ndarray
```

Estimate local fiber volume fraction from segmented image.

**Args:**
- `segmentation` (np.ndarray): Segmented image/volume where fiber regions are labeled.
- `fiber_label` (int): Label value representing fiber regions. Default is 1.
- `window_size` (int or tuple): Size of the window for local Vf calculation.
- `gaussian_sigma` (float, optional): If provided, use Gaussian-weighted averaging.
- `normalize` (bool): If True, normalize output to [0, 1] range. Default is True.

**Returns:**
- `np.ndarray`: Local fiber volume fraction map with same shape as input.

**Example:**

```python
from vmm.segment import estimate_local_vf

vf_map = estimate_local_vf(binary_segmentation, window_size=50)
```

---

### estimate_vf_distribution

```python
def estimate_vf_distribution(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, ...]] = 50,
    gaussian_sigma: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, dict]
```

Estimate the distribution of local fiber volume fraction.

**Args:**
- `segmentation` (np.ndarray): Segmented image/volume.
- `fiber_label` (int): Label value representing fiber regions.
- `window_size` (int or tuple): Size of the window for local Vf calculation.
- `gaussian_sigma` (float, optional): If provided, use Gaussian-weighted averaging.
- `mask` (np.ndarray, optional): Binary mask to restrict analysis.
- `bins` (int): Number of bins for histogram. Default is 50.

**Returns:**
- `hist` (np.ndarray): Histogram counts.
- `bin_edges` (np.ndarray): Bin edges for the histogram.
- `stats` (dict): Dictionary with statistical measures ('mean', 'std', 'median', 'min', 'max', 'global_vf').

---

### estimate_vf_slice_by_slice

```python
def estimate_vf_slice_by_slice(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, int]] = 50,
    gaussian_sigma: Optional[float] = None,
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]
```

Estimate fiber volume fraction for each slice along an axis.

**Args:**
- `segmentation` (np.ndarray): 3D segmented volume.
- `fiber_label` (int): Label value representing fiber regions.
- `window_size` (int or tuple): Size of the window for local Vf calculation.
- `gaussian_sigma` (float, optional): If provided, use Gaussian-weighted averaging.
- `axis` (int): Axis along which to compute slice-by-slice Vf. Default is 0 (z-axis).

**Returns:**
- `slice_indices` (np.ndarray): Array of slice indices.
- `vf_per_slice` (np.ndarray): Mean Vf for each slice.

---

### compute_vf_map_3d

```python
def compute_vf_map_3d(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, int, int]] = 50,
    gaussian_sigma: Optional[Union[float, Tuple[float, float, float]]] = None
) -> np.ndarray
```

Compute 3D local fiber volume fraction map.

**Args:**
- `segmentation` (np.ndarray): 3D segmented volume.
- `fiber_label` (int): Label value representing fiber regions.
- `window_size` (int or tuple): Size of the window for local Vf calculation.
- `gaussian_sigma` (float or tuple, optional): If provided, use Gaussian-weighted averaging.

**Returns:**
- `np.ndarray`: 3D local fiber volume fraction map.

---

### threshold_otsu

```python
def threshold_otsu(image: np.ndarray) -> Tuple[np.ndarray, float]
```

Apply Otsu's thresholding to segment fiber and matrix.

**Args:**
- `image` (np.ndarray): Grayscale image or volume.

**Returns:**
- `binary` (np.ndarray): Binary segmentation (1=fiber, 0=matrix).
- `threshold` (float): Computed Otsu threshold value.

**Example:**

```python
from vmm.segment import threshold_otsu

binary, threshold = threshold_otsu(volume)
print(f"Otsu threshold: {threshold}")
```

---

### threshold_percentile

```python
def threshold_percentile(
    image: np.ndarray,
    percentile: float = 50.0,
    invert: bool = False
) -> Tuple[np.ndarray, float]
```

Apply percentile-based thresholding to segment fiber and matrix.

**Args:**
- `image` (np.ndarray): Grayscale image or volume.
- `percentile` (float): Percentile value for threshold (0-100).
- `invert` (bool): If True, values below threshold are labeled as fiber.

**Returns:**
- `binary` (np.ndarray): Binary segmentation (1=fiber, 0=matrix).
- `threshold` (float): Computed threshold value.

---

### apply_morphological_cleaning

```python
def apply_morphological_cleaning(
    segmentation: np.ndarray,
    opening_size: int = 2,
    closing_size: int = 2
) -> np.ndarray
```

Apply morphological operations to clean up segmentation.

**Args:**
- `segmentation` (np.ndarray): Binary segmentation image/volume.
- `opening_size` (int): Size of structuring element for opening (removes small objects).
- `closing_size` (int): Size of structuring element for closing (fills small holes).

**Returns:**
- `np.ndarray`: Cleaned binary segmentation.
