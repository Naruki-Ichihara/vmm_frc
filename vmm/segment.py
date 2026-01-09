"""
Segmentation utilities for fiber volume fraction estimation.

This module provides methods to estimate local fiber volume fraction (Vf)
from segmented CT images where fibers and matrix regions are identified.
"""

import numpy as np
from scipy import ndimage
from typing import Optional, Tuple, Union
from vmm.logger import get_logger

logger = get_logger()


def estimate_local_vf(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    void_label: Optional[int] = None,
    window_size: Union[int, Tuple[int, ...]] = 50,
    gaussian_sigma: Optional[float] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Estimate local fiber volume fraction from segmented image.

    This function calculates the local fiber volume fraction (Vf) by computing
    the fraction of fiber pixels within a sliding window or using Gaussian
    weighted averaging. If void_label is provided, void regions are excluded
    from the calculation (Vf = fiber / (fiber + matrix), excluding void).

    Args:
        segmentation: Segmented image/volume where fiber regions are labeled.
                     Can be 2D (single slice) or 3D (volume).
        fiber_label: Label value representing fiber regions in the segmentation.
                    Default is 1 (assuming: 0=background, 1=fiber, 2=matrix, 3=void).
        void_label: Label value representing void regions. If provided, voids are
                   excluded from Vf calculation. Default is None (no void exclusion).
        window_size: Size of the window for local Vf calculation.
                    - If int: same size for all dimensions
                    - If tuple: specific size for each dimension
                    For box averaging without Gaussian smoothing.
        gaussian_sigma: If provided, use Gaussian-weighted averaging instead of
                       box averaging. The sigma value controls the smoothing scale.
                       Larger values result in smoother Vf distributions.
        normalize: If True, normalize output to [0, 1] range. Default is True.

    Returns:
        np.ndarray: Local fiber volume fraction map with same shape as input.
                   Values range from 0 (pure matrix) to 1 (pure fiber).

    Example:
        >>> # Segmentation: 0=background, 1=fiber, 2=matrix, 3=void
        >>> seg = np.array([[0, 0, 1, 1],
        ...                 [0, 1, 1, 1],
        ...                 [1, 1, 1, 3],
        ...                 [1, 1, 3, 3]])
        >>> vf = estimate_local_vf(seg, void_label=3, window_size=3)
        >>> print(vf.shape)
        (4, 4)
    """
    logger.info(f"Estimating local Vf: input shape={segmentation.shape}, fiber_label={fiber_label}, window_size={window_size}")
    if void_label is not None:
        logger.debug(f"Void exclusion enabled: void_label={void_label}")

    # Create binary fiber mask
    fiber_mask = (segmentation == fiber_label).astype(np.float64)

    if void_label is not None:
        # Create valid region mask (exclude void)
        valid_mask = (segmentation != void_label).astype(np.float64)
        # Also exclude background (label 0) from valid region if we want fiber/(fiber+matrix)
        # Actually, for Vf calculation, we want fiber / (all non-void pixels that are fiber or matrix)
        # Let's only exclude void from the denominator
    else:
        valid_mask = None

    if gaussian_sigma is not None:
        logger.debug(f"Using Gaussian averaging: sigma={gaussian_sigma}")
        # Gaussian-weighted averaging
        fiber_sum = ndimage.gaussian_filter(fiber_mask, sigma=gaussian_sigma)
        if valid_mask is not None:
            valid_sum = ndimage.gaussian_filter(valid_mask, sigma=gaussian_sigma)
            # Avoid division by zero
            vf_map = np.divide(fiber_sum, valid_sum, out=np.zeros_like(fiber_sum), where=valid_sum > 0.001)
        else:
            vf_map = fiber_sum
    else:
        logger.debug(f"Using box averaging: window_size={window_size}")
        # Box averaging using uniform filter
        if isinstance(window_size, int):
            size = window_size
        else:
            size = window_size

        fiber_sum = ndimage.uniform_filter(fiber_mask, size=size, mode='nearest')
        if valid_mask is not None:
            valid_sum = ndimage.uniform_filter(valid_mask, size=size, mode='nearest')
            # Avoid division by zero
            vf_map = np.divide(fiber_sum, valid_sum, out=np.zeros_like(fiber_sum), where=valid_sum > 0.001)
        else:
            vf_map = fiber_sum

    if normalize:
        # Ensure values are in [0, 1] range
        vf_map = np.clip(vf_map, 0.0, 1.0)

    logger.info(f"Local Vf computed: output shape={vf_map.shape}, mean={np.mean(vf_map):.3f}, range=[{np.min(vf_map):.3f}, {np.max(vf_map):.3f}]")

    return vf_map


def estimate_vf_distribution(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    void_label: Optional[int] = None,
    window_size: Union[int, Tuple[int, ...]] = 50,
    gaussian_sigma: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Estimate the distribution of local fiber volume fraction.

    Args:
        segmentation: Segmented image/volume where fiber regions are labeled.
        fiber_label: Label value representing fiber regions.
        void_label: Label value representing void regions. If provided, voids are
                   excluded from Vf calculation. Default is None.
        window_size: Size of the window for local Vf calculation.
        gaussian_sigma: If provided, use Gaussian-weighted averaging.
        mask: Optional binary mask to restrict analysis to specific region.
        bins: Number of bins for histogram.

    Returns:
        Tuple containing:
        - hist: Histogram counts
        - bin_edges: Bin edges for the histogram
        - stats: Dictionary with statistical measures:
            - 'mean': Mean Vf
            - 'std': Standard deviation
            - 'median': Median Vf
            - 'min': Minimum Vf
            - 'max': Maximum Vf
            - 'global_vf': Global fiber volume fraction
            - 'void_fraction': Void fraction (if void_label provided)
    """
    logger.info(f"Estimating Vf distribution: input shape={segmentation.shape}, bins={bins}")

    # Calculate local Vf map
    vf_map = estimate_local_vf(
        segmentation,
        fiber_label=fiber_label,
        void_label=void_label,
        window_size=window_size,
        gaussian_sigma=gaussian_sigma
    )

    # Apply mask if provided
    if mask is not None:
        vf_values = vf_map[mask > 0]
        fiber_mask = (segmentation == fiber_label)
        if void_label is not None:
            # Exclude void from global Vf calculation
            valid_mask = (segmentation != void_label) & (mask > 0)
            fiber_count = np.sum(fiber_mask[mask > 0])
            valid_count = np.sum(valid_mask)
            global_vf = fiber_count / valid_count if valid_count > 0 else 0
            # Calculate void fraction
            void_mask = (segmentation == void_label) & (mask > 0)
            void_fraction = np.sum(void_mask) / np.sum(mask > 0)
        else:
            global_vf = np.sum(fiber_mask[mask > 0]) / np.sum(mask > 0)
            void_fraction = 0
    else:
        vf_values = vf_map.ravel()
        fiber_mask = (segmentation == fiber_label)
        if void_label is not None:
            # Exclude void from global Vf calculation
            valid_mask = (segmentation != void_label)
            fiber_count = np.sum(fiber_mask)
            valid_count = np.sum(valid_mask)
            global_vf = fiber_count / valid_count if valid_count > 0 else 0
            # Calculate void fraction
            void_mask = (segmentation == void_label)
            void_fraction = np.mean(void_mask)
        else:
            global_vf = np.mean(fiber_mask)
            void_fraction = 0

    # Compute histogram
    hist, bin_edges = np.histogram(vf_values, bins=bins, range=(0, 1))

    # Compute statistics
    stats = {
        'mean': np.mean(vf_values),
        'std': np.std(vf_values),
        'median': np.median(vf_values),
        'min': np.min(vf_values),
        'max': np.max(vf_values),
        'global_vf': global_vf,
        'void_fraction': void_fraction
    }

    logger.info(f"Vf distribution computed: mean={stats['mean']:.3f}, std={stats['std']:.3f}, global_vf={global_vf:.3f}")
    if void_label is not None:
        logger.debug(f"Void fraction: {void_fraction:.3f}")

    return hist, bin_edges, stats


def estimate_vf_slice_by_slice(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, int]] = 50,
    gaussian_sigma: Optional[float] = None,
    axis: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate fiber volume fraction for each slice along an axis.

    This is useful for analyzing Vf variation through the thickness of a sample.

    Args:
        segmentation: 3D segmented volume.
        fiber_label: Label value representing fiber regions.
        window_size: Size of the window for local Vf calculation (2D).
        gaussian_sigma: If provided, use Gaussian-weighted averaging.
        axis: Axis along which to compute slice-by-slice Vf (default: 0 = z-axis).

    Returns:
        Tuple containing:
        - slice_indices: Array of slice indices
        - vf_per_slice: Mean Vf for each slice
    """
    if segmentation.ndim != 3:
        raise ValueError("Input must be a 3D volume")

    n_slices = segmentation.shape[axis]
    logger.info(f"Computing slice-by-slice Vf: n_slices={n_slices}, axis={axis}")

    slice_indices = np.arange(n_slices)
    vf_per_slice = np.zeros(n_slices)

    for i in range(n_slices):
        # Extract slice
        if axis == 0:
            slice_seg = segmentation[i, :, :]
        elif axis == 1:
            slice_seg = segmentation[:, i, :]
        else:
            slice_seg = segmentation[:, :, i]

        # Calculate local Vf for this slice
        vf_map = estimate_local_vf(
            slice_seg,
            fiber_label=fiber_label,
            window_size=window_size,
            gaussian_sigma=gaussian_sigma
        )

        vf_per_slice[i] = np.mean(vf_map)

    logger.info(f"Slice-by-slice Vf completed: mean_vf={np.mean(vf_per_slice):.3f}, range=[{np.min(vf_per_slice):.3f}, {np.max(vf_per_slice):.3f}]")

    return slice_indices, vf_per_slice


def compute_vf_map_3d(
    segmentation: np.ndarray,
    fiber_label: int = 1,
    window_size: Union[int, Tuple[int, int, int]] = 50,
    gaussian_sigma: Optional[Union[float, Tuple[float, float, float]]] = None
) -> np.ndarray:
    """
    Compute 3D local fiber volume fraction map.

    This function computes a full 3D Vf distribution map, which can be
    visualized as slices or rendered in 3D.

    Args:
        segmentation: 3D segmented volume.
        fiber_label: Label value representing fiber regions.
        window_size: Size of the window for local Vf calculation.
                    Can be int (isotropic) or tuple (anisotropic).
        gaussian_sigma: If provided, use Gaussian-weighted averaging.
                       Can be float (isotropic) or tuple (anisotropic).

    Returns:
        np.ndarray: 3D local fiber volume fraction map.
    """
    if segmentation.ndim != 3:
        raise ValueError("Input must be a 3D volume")

    logger.info(f"Computing 3D Vf map: input shape={segmentation.shape}, window_size={window_size}")

    result = estimate_local_vf(
        segmentation,
        fiber_label=fiber_label,
        window_size=window_size,
        gaussian_sigma=gaussian_sigma
    )

    logger.info(f"3D Vf map computed: output shape={result.shape}")
    return result


def threshold_otsu(image: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Apply Otsu's thresholding to segment fiber and matrix.

    Args:
        image: Grayscale image or volume.

    Returns:
        Tuple containing:
        - binary: Binary segmentation (1=fiber, 0=matrix)
        - threshold: Computed Otsu threshold value
    """
    from skimage.filters import threshold_otsu as skimage_otsu

    logger.debug(f"Computing Otsu threshold: input shape={image.shape}")
    threshold = skimage_otsu(image)
    binary = (image > threshold).astype(np.uint8)
    logger.info(f"Otsu threshold computed: threshold={threshold:.2f}, fiber_fraction={np.mean(binary):.3f}")

    return binary, threshold


def threshold_percentile(
    image: np.ndarray,
    percentile: float = 50.0,
    invert: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Apply percentile-based thresholding to segment fiber and matrix.

    Args:
        image: Grayscale image or volume.
        percentile: Percentile value for threshold (0-100).
        invert: If True, values below threshold are labeled as fiber.

    Returns:
        Tuple containing:
        - binary: Binary segmentation (1=fiber, 0=matrix)
        - threshold: Computed threshold value
    """
    logger.debug(f"Computing percentile threshold: percentile={percentile}, invert={invert}")
    threshold = np.percentile(image, percentile)

    if invert:
        binary = (image < threshold).astype(np.uint8)
    else:
        binary = (image > threshold).astype(np.uint8)

    logger.info(f"Percentile threshold computed: threshold={threshold:.2f}, fiber_fraction={np.mean(binary):.3f}")

    return binary, threshold


def apply_morphological_cleaning(
    segmentation: np.ndarray,
    opening_size: int = 2,
    closing_size: int = 2
) -> np.ndarray:
    """
    Apply morphological operations to clean up segmentation.

    Args:
        segmentation: Binary segmentation image/volume.
        opening_size: Size of structuring element for opening (removes small objects).
        closing_size: Size of structuring element for closing (fills small holes).

    Returns:
        np.ndarray: Cleaned binary segmentation.
    """
    from scipy.ndimage import binary_opening, binary_closing

    logger.info(f"Applying morphological cleaning: input shape={segmentation.shape}, opening_size={opening_size}, closing_size={closing_size}")

    # Create structuring element based on dimensionality
    if segmentation.ndim == 2:
        struct_open = np.ones((opening_size, opening_size))
        struct_close = np.ones((closing_size, closing_size))
    else:
        struct_open = np.ones((opening_size, opening_size, opening_size))
        struct_close = np.ones((closing_size, closing_size, closing_size))

    # Apply opening to remove small noise
    if opening_size > 0:
        cleaned = binary_opening(segmentation, structure=struct_open)
        logger.debug(f"Opening applied: removed {np.sum(segmentation) - np.sum(cleaned)} pixels")
    else:
        cleaned = segmentation.copy()

    # Apply closing to fill small holes
    if closing_size > 0:
        before_closing = np.sum(cleaned)
        cleaned = binary_closing(cleaned, structure=struct_close)
        logger.debug(f"Closing applied: added {np.sum(cleaned) - before_closing} pixels")

    logger.info(f"Morphological cleaning complete: final fiber fraction={np.mean(cleaned):.3f}")

    return cleaned.astype(np.uint8)
