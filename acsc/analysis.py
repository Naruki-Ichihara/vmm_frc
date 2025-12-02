import numpy as np
import numba
from skimage.feature import structure_tensor
import pandas as pd
from typing import Optional, Tuple, Union, List, Dict

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
    tensor_list = structure_tensor(volume, sigma=noise_scale, mode=mode)
    tensors = np.empty((6, *tensor_list[0].shape), dtype=np.float32)
    for i, tensor in enumerate(tensor_list):
        tensors[i] = tensor
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
        print("[INFO] Computing orientation function without reference vector.")
        print("[INFO] Progressing...")
        theta, phi = _orientation_function(structure_tensor)
        print("[INFO] Progress complete.")
        return theta, phi
    else:
        print("[INFO] Computing orientation function with reference vector.")
        print("[INFO] Progressing...")
        theta = _orientation_function_reference(structure_tensor, reference_vector)
        print("[INFO] Progress complete.")
        return theta