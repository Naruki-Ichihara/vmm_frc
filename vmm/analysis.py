import numpy as np
import numba
from skimage.feature import structure_tensor
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from scipy import ndimage
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict
from vmm.logger import get_logger

logger = get_logger()

def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray:
    """
    Remove edge pixels from all sides of a 3D volume to eliminate boundary artifacts.
    
    This function crops the volume by removing 'width' pixels from each face,
    effectively reducing each dimension by 2*width pixels.

    Args:
        width: Number of pixels to remove from each edge (must be positive).
        volume: Input 3D array with shape (depth, height, width).

    Returns:
        Cropped 3D volume with shape (depth-2*width, height-2*width, width-2*width).
        
    Raises:
        ValueError: If width is larger than half of any volume dimension.
        
    Example:
        >>> vol = np.random.rand(100, 200, 200)
        >>> cropped = drop_edges_3D(10, vol)
        >>> print(cropped.shape)  # (80, 180, 180)
    """
    return volume[width:volume.shape[0]-width, width:volume.shape[1]-width, width:volume.shape[2]-width]

def compute_structure_tensor(volume: np.ndarray, noise_scale: int, mode: str = 'nearest') -> np.ndarray:
    """
    Compute 3D structure tensor for fiber orientation analysis using gradient-based methods.
    
    The structure tensor is a symmetric 3x3 matrix field that captures local orientation
    information in 3D volumes. Each voxel gets a tensor describing the predominant
    direction of intensity gradients in its neighborhood.

    Args:
        volume: Input 3D grayscale volume (depth, height, width).
        noise_scale: Gaussian smoothing sigma for noise reduction before gradient computation.
                    Larger values provide more smoothing but reduce spatial resolution.
        mode: Boundary condition for Gaussian filtering.
             Options: 'constant', 'edge', 'wrap', 'reflect', 'mirror', 'nearest'.
             See scikit-image documentation for details.

    Returns:
        Structure tensor array with shape (6, depth, height, width).
        Components represent the upper triangular part of the symmetric 3x3 tensor:
        [T_xx, T_xy, T_xz, T_yy, T_yz, T_zz] where T_ij = ∇I_i * ∇I_j.
    
    Raises:
        ValueError: If the input volume is not exactly 3D.
        
    Note:
        Uses float32 precision for memory efficiency in large volume processing.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D.")

    logger.info(f"Computing structure tensor: volume shape={volume.shape}, noise_scale={noise_scale}")

    tensor_list = structure_tensor(volume, sigma=noise_scale, mode=mode)
    tensors = np.empty((6, *tensor_list[0].shape), dtype=np.float32)
    for i, tensor in enumerate(tensor_list):
        tensors[i] = tensor

    logger.info(f"Structure tensor computed: output shape={tensors.shape}")
    return tensors

@numba.njit(parallel=True, cache=True)
def _orientation_function(structureTensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    phi = np.zeros(structureTensor.shape[1:], dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[2], selectedEigenVector[0]))
                phi[z, y, x] = np.rad2deg(np.arctan2(selectedEigenVector[1], selectedEigenVector[0]))

    return theta, phi

@numba.njit(parallel=True, cache=True)
def _orientation_function_reference(structureTensor: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:

    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    theta = np.zeros(structureTensor.shape[1:], dtype="<f4")
    axial_vec = np.array(reference_vector, dtype="<f4")

    for z in numba.prange(0, structureTensor.shape[1]):
        for y in range(0, structureTensor.shape[2]):
            for x in range(0, structureTensor.shape[3]):
                structureTensorLocal = np.empty((3, 3), dtype="<f4")
                for n, [i, j] in enumerate(symmetricComponents3d):
                    structureTensorLocal[i, j] = structureTensor[n, z, y, x]
                    if i != j:
                        structureTensorLocal[j, i] = structureTensor[n, z, y, x]

                w, v = np.linalg.eig(structureTensorLocal)
                m = np.argmin(w)

                selectedEigenVector = v[:, m]

                if selectedEigenVector[0] < 0:
                    selectedEigenVector *= -1

                theta[z, y, x] = np.rad2deg(np.arccos(np.dot(selectedEigenVector, axial_vec)))

    return theta

def compute_orientation(structure_tensor: np.ndarray, reference_vector: Optional[List[float]] = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Extract fiber orientation angles from structure tensor using eigenvalue decomposition.
    
    Computes orientation angles by finding the eigenvector corresponding to the smallest
    eigenvalue of each structure tensor, which represents the direction of minimal
    intensity variation (i.e., the fiber direction).

    Args:
        structure_tensor: 4D array with shape (6, depth, height, width) containing
                         the symmetric structure tensor components.
        reference_vector: Optional 3D reference direction vector [x, y, z].
                         If provided, returns only the angle relative to this reference.
                         If None, returns both theta and phi spherical angles.

    Returns:
        If reference_vector is None:
            Tuple of (theta, phi) arrays with shape (depth, height, width).
            theta: Azimuthal angle in degrees (-180 to 180).
            phi: Elevation angle in degrees (-90 to 90).
        If reference_vector is provided:
            Single array of angles in degrees (0 to 180) relative to reference.
            
    Note:
        Uses JIT compilation for performance. Progress messages printed during computation.
        Eigenvector signs are normalized for consistency (x-component made positive).
    """
    if reference_vector is None:
        logger.info("Computing orientation function without reference vector")
        logger.debug(f"Structure tensor shape: {structure_tensor.shape}")
        theta, phi = _orientation_function(structure_tensor)
        logger.info(f"Orientation computed: theta shape={theta.shape}, phi shape={phi.shape}")
        logger.debug(f"Theta range: [{np.min(theta):.2f}, {np.max(theta):.2f}] degrees")
        logger.debug(f"Phi range: [{np.min(phi):.2f}, {np.max(phi):.2f}] degrees")
        return theta, phi
    else:
        logger.info(f"Computing orientation function with reference vector: {reference_vector}")
        logger.debug(f"Structure tensor shape: {structure_tensor.shape}")
        theta = _orientation_function_reference(structure_tensor, reference_vector)
        logger.info(f"Orientation computed: theta shape={theta.shape}")
        logger.debug(f"Theta range: [{np.min(theta):.2f}, {np.max(theta):.2f}] degrees")
        return theta


# =============================================================================
# Void Analysis Functions
# =============================================================================

def segment_voids_otsu(
    volume: np.ndarray,
    invert: bool = True,
    min_size: int = 0,
    closing_size: int = 0
) -> Tuple[np.ndarray, float]:
    """
    Segment voids in a 3D volume using Otsu's thresholding method.

    Voids typically appear as dark regions in CT images. This function
    uses Otsu's method to automatically determine the threshold separating
    void regions from the material (fiber + matrix).

    Args:
        volume: Input 3D grayscale volume (depth, height, width).
        invert: If True (default), assumes voids are darker than material.
               Set to False if voids appear brighter.
        min_size: Minimum void size in voxels. Smaller regions are removed.
                 Set to 0 to keep all detected voids.
        closing_size: Morphological closing kernel size for filling small holes.
                     Set to 0 to disable closing.

    Returns:
        Tuple of:
        - binary_mask: Boolean 3D array where True indicates void regions.
        - threshold: The computed Otsu threshold value.

    Example:
        >>> void_mask, thresh = segment_voids_otsu(ct_volume)
        >>> void_fraction = np.mean(void_mask)
        >>> print(f"Void fraction: {void_fraction*100:.2f}%")
    """
    # Compute Otsu threshold
    threshold = threshold_otsu(volume)

    # Create binary mask
    if invert:
        # Voids are dark (below threshold)
        binary_mask = volume < threshold
    else:
        # Voids are bright (above threshold)
        binary_mask = volume > threshold

    # Apply morphological closing to fill small holes
    if closing_size > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        binary_mask = ndimage.binary_closing(binary_mask, structure=struct, iterations=closing_size)

    # Remove small objects
    if min_size > 0:
        labeled = label(binary_mask)
        for region in regionprops(labeled):
            if region.area < min_size:
                binary_mask[labeled == region.label] = False

    return binary_mask, threshold


def segment_voids_from_insegt(
    segmentation: np.ndarray,
    void_label: int = 3
) -> np.ndarray:
    """
    Extract void regions from InSegt segmentation results.

    InSegt segmentation uses:
    - Label 1: Fiber (Cyan)
    - Label 2: Matrix (Magenta)
    - Label 3: Void (Yellow)

    This function extracts the void regions (label 3) from the segmentation.

    Args:
        segmentation: 3D array of InSegt segmentation labels.
        void_label: Label value representing void regions (default: 3).

    Returns:
        Boolean 3D array where True indicates void regions.

    Example:
        >>> void_mask = segment_voids_from_insegt(insegt_result)
        >>> void_fraction = np.mean(void_mask)
    """
    return segmentation == void_label


def compute_void_statistics(
    void_mask: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_void_size: int = 1
) -> Dict:
    """
    Compute comprehensive statistics for void regions.

    Analyzes the void distribution including size, shape, and spatial
    distribution metrics useful for material characterization.

    Args:
        void_mask: Boolean 3D array where True indicates void regions.
        voxel_size: Physical size of each voxel as (z, y, x) in consistent units.
                   Used to convert voxel counts to physical volumes.
        min_void_size: Minimum void size in voxels to include in statistics.

    Returns:
        Dictionary containing:
        - 'void_fraction': Total void volume fraction (0-1).
        - 'void_fraction_percent': Total void volume fraction (%).
        - 'total_void_volume': Total void volume in physical units^3.
        - 'num_voids': Number of distinct void regions.
        - 'void_sizes': Array of individual void volumes in physical units^3.
        - 'mean_void_size': Mean void volume.
        - 'std_void_size': Standard deviation of void volumes.
        - 'max_void_size': Maximum void volume.
        - 'min_void_size': Minimum void volume (above threshold).
        - 'void_sphericity': Array of sphericity values for each void (0-1).
        - 'mean_sphericity': Mean sphericity of voids.
        - 'void_centroids': Array of void centroid coordinates.
        - 'slice_void_fractions': Void fraction for each slice along z-axis.

    Example:
        >>> stats = compute_void_statistics(void_mask, voxel_size=(0.5, 0.5, 0.5))
        >>> print(f"Void fraction: {stats['void_fraction_percent']:.2f}%")
        >>> print(f"Number of voids: {stats['num_voids']}")
    """
    voxel_volume = np.prod(voxel_size)
    total_voxels = void_mask.size
    void_voxels = np.sum(void_mask)

    # Basic statistics
    void_fraction = void_voxels / total_voxels
    total_void_volume = void_voxels * voxel_volume

    # Label connected components
    labeled_voids = label(void_mask)

    # Analyze each void region
    void_sizes = []
    void_sphericities = []
    void_centroids = []

    for region in regionprops(labeled_voids):
        if region.area >= min_void_size:
            # Volume in physical units
            volume = region.area * voxel_volume
            void_sizes.append(volume)

            # Sphericity approximation using major/minor axis ratio
            if region.area > 0:
                try:
                    aspect_ratio = region.major_axis_length / max(region.minor_axis_length, 1)
                    sphericity = 1.0 / max(aspect_ratio, 1.0)
                except Exception:
                    sphericity = 1.0
                void_sphericities.append(min(sphericity, 1.0))
            else:
                void_sphericities.append(0.0)

            # Centroid
            void_centroids.append(region.centroid)

    void_sizes = np.array(void_sizes) if void_sizes else np.array([])
    void_sphericities = np.array(void_sphericities) if void_sphericities else np.array([])
    void_centroids = np.array(void_centroids) if void_centroids else np.array([]).reshape(0, 3)

    # Slice-by-slice void fraction
    n_slices = void_mask.shape[0]
    slice_void_fractions = np.array([
        np.mean(void_mask[z]) for z in range(n_slices)
    ])

    # Compile statistics
    stats = {
        'void_fraction': void_fraction,
        'void_fraction_percent': void_fraction * 100,
        'total_void_volume': total_void_volume,
        'num_voids': len(void_sizes),
        'void_sizes': void_sizes,
        'mean_void_size': np.mean(void_sizes) if len(void_sizes) > 0 else 0,
        'std_void_size': np.std(void_sizes) if len(void_sizes) > 0 else 0,
        'max_void_size': np.max(void_sizes) if len(void_sizes) > 0 else 0,
        'min_void_size': np.min(void_sizes) if len(void_sizes) > 0 else 0,
        'void_sphericity': void_sphericities,
        'mean_sphericity': np.mean(void_sphericities) if len(void_sphericities) > 0 else 0,
        'void_centroids': void_centroids,
        'slice_void_fractions': slice_void_fractions,
        'voxel_size': voxel_size,
    }

    return stats


def compute_local_void_fraction(
    void_mask: np.ndarray,
    window_size: int = 50,
    gaussian_sigma: Optional[float] = None
) -> np.ndarray:
    """
    Compute local void fraction map using sliding window or Gaussian averaging.

    Creates a 3D map showing the spatial distribution of void concentration,
    useful for identifying regions with higher void content.

    Args:
        void_mask: Boolean 3D array where True indicates void regions.
        window_size: Size of the averaging window in voxels.
        gaussian_sigma: If provided, use Gaussian-weighted averaging instead
                       of box averaging. Larger values = smoother result.

    Returns:
        3D array of local void fractions (0-1) with same shape as input.

    Example:
        >>> local_vf = compute_local_void_fraction(void_mask, window_size=30)
        >>> hot_spots = local_vf > 0.1  # Regions with >10% local void content
    """
    void_float = void_mask.astype(np.float64)

    if gaussian_sigma is not None:
        local_vf = ndimage.gaussian_filter(void_float, sigma=gaussian_sigma)
    else:
        local_vf = ndimage.uniform_filter(void_float, size=window_size, mode='nearest')

    return np.clip(local_vf, 0.0, 1.0)


def analyze_void_distribution(
    void_mask: np.ndarray,
    bins: int = 50,
    log_scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Analyze the size distribution of voids.

    Computes histogram and statistics of void sizes, useful for
    understanding the void population characteristics.

    Args:
        void_mask: Boolean 3D array where True indicates void regions.
        bins: Number of histogram bins.
        log_scale: If True, use logarithmic binning for size distribution.

    Returns:
        Tuple of:
        - hist: Histogram counts.
        - bin_edges: Bin edges for the histogram.
        - stats: Dictionary with distribution statistics.

    Example:
        >>> hist, edges, stats = analyze_void_distribution(void_mask)
        >>> print(f"Median void size: {stats['median_size']:.1f} voxels")
    """
    # Label connected components
    labeled_voids = label(void_mask)

    # Get sizes
    sizes = []
    for region in regionprops(labeled_voids):
        if region.area > 0:
            sizes.append(region.area)

    sizes = np.array(sizes) if sizes else np.array([1])

    # Compute histogram
    if log_scale and len(sizes) > 0 and np.min(sizes) > 0:
        log_sizes = np.log10(sizes)
        hist, bin_edges = np.histogram(log_sizes, bins=bins)
        bin_edges = 10 ** bin_edges  # Convert back to linear
    else:
        hist, bin_edges = np.histogram(sizes, bins=bins)

    # Statistics
    stats = {
        'count': len(sizes),
        'mean_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'median_size': np.median(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'total_volume': np.sum(sizes),
    }

    return hist, bin_edges, stats


# =============================================================================
# Orientation Masking Functions
# =============================================================================

def mask_orientation_with_voids(
    theta: np.ndarray,
    void_mask: np.ndarray,
    dilation_pixels: int = 0,
    phi: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Mask orientation data using void regions.

    Sets orientation values to NaN where voids are present, optionally
    dilating the void mask to exclude regions near voids.

    Args:
        theta: 3D array of theta orientation angles.
        void_mask: Boolean 3D array where True indicates void regions.
        dilation_pixels: Number of pixels to dilate the void mask.
                        This excludes regions near voids where orientation
                        may be unreliable. Set to 0 to use exact void mask.
        phi: Optional 3D array of phi orientation angles.

    Returns:
        If phi is None: Masked theta array with NaN at void locations.
        If phi is provided: Tuple of (masked_theta, masked_phi).

    Example:
        >>> masked_theta = mask_orientation_with_voids(theta, void_mask, dilation_pixels=5)
        >>> # Now theta values at and near voids are NaN
    """
    # Dilate void mask if requested
    if dilation_pixels > 0:
        struct = ndimage.generate_binary_structure(3, 1)
        dilated_mask = ndimage.binary_dilation(
            void_mask,
            structure=struct,
            iterations=dilation_pixels
        )
    else:
        dilated_mask = void_mask

    # Create masked theta
    masked_theta = theta.copy().astype(np.float64)
    masked_theta[dilated_mask] = np.nan

    if phi is not None:
        masked_phi = phi.copy().astype(np.float64)
        masked_phi[dilated_mask] = np.nan
        return masked_theta, masked_phi

    return masked_theta


def crop_orientation_to_roi(
    theta: np.ndarray,
    roi_bounds: List[int],
    void_mask: Optional[np.ndarray] = None,
    void_roi_bounds: Optional[List[int]] = None,
    dilation_pixels: int = 0,
    phi: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Crop orientation data to ROI and optionally mask with voids.

    Args:
        theta: 3D array of theta orientation angles.
        roi_bounds: ROI bounds as [z_min, z_max, y_min, y_max, x_min, x_max].
        void_mask: Optional boolean 3D array of void regions.
        void_roi_bounds: ROI bounds for void mask if different from theta ROI.
        dilation_pixels: Number of pixels to dilate void mask.
        phi: Optional 3D array of phi orientation angles.

    Returns:
        Cropped (and optionally masked) orientation data.

    Example:
        >>> cropped_theta = crop_orientation_to_roi(
        ...     theta, [0, 100, 0, 200, 0, 200],
        ...     void_mask=void_mask, dilation_pixels=3
        ... )
    """
    z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds

    # Crop theta
    cropped_theta = theta[z_min:z_max, y_min:y_max, x_min:x_max].copy()

    # Crop phi if provided
    cropped_phi = None
    if phi is not None:
        cropped_phi = phi[z_min:z_max, y_min:y_max, x_min:x_max].copy()

    # Apply void mask if provided
    if void_mask is not None:
        # Handle void mask alignment
        if void_roi_bounds is not None:
            vz_min, vz_max, vy_min, vy_max, vx_min, vx_max = void_roi_bounds

            # Calculate overlap between theta ROI and void ROI
            oz_min = max(z_min, vz_min)
            oz_max = min(z_max, vz_max)
            oy_min = max(y_min, vy_min)
            oy_max = min(y_max, vy_max)
            ox_min = max(x_min, vx_min)
            ox_max = min(x_max, vx_max)

            if oz_min < oz_max and oy_min < oy_max and ox_min < ox_max:
                # Create aligned void mask for cropped theta
                aligned_void_mask = np.zeros(cropped_theta.shape, dtype=bool)

                # Calculate indices in cropped theta space
                ct_z_start = oz_min - z_min
                ct_z_end = oz_max - z_min
                ct_y_start = oy_min - y_min
                ct_y_end = oy_max - y_min
                ct_x_start = ox_min - x_min
                ct_x_end = ox_max - x_min

                # Calculate indices in void mask space
                vm_z_start = oz_min - vz_min
                vm_z_end = oz_max - vz_min
                vm_y_start = oy_min - vy_min
                vm_y_end = oy_max - vy_min
                vm_x_start = ox_min - vx_min
                vm_x_end = ox_max - vx_min

                aligned_void_mask[ct_z_start:ct_z_end, ct_y_start:ct_y_end, ct_x_start:ct_x_end] = \
                    void_mask[vm_z_start:vm_z_end, vm_y_start:vm_y_end, vm_x_start:vm_x_end]

                void_mask_to_use = aligned_void_mask
            else:
                # No overlap, no masking needed
                void_mask_to_use = None
        else:
            # Assume void mask is same size as cropped theta
            void_mask_to_use = void_mask

        if void_mask_to_use is not None:
            if cropped_phi is not None:
                cropped_theta, cropped_phi = mask_orientation_with_voids(
                    cropped_theta, void_mask_to_use, dilation_pixels, cropped_phi
                )
            else:
                cropped_theta = mask_orientation_with_voids(
                    cropped_theta, void_mask_to_use, dilation_pixels
                )

    if cropped_phi is not None:
        return cropped_theta, cropped_phi
    return cropped_theta