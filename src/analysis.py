import numpy as np
import numba
from skimage.feature import structure_tensor
import pandas as pd

def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray:
    """Drop edges of 3D volume.

    Args:
        width (int): Width of edges.
        volume (np.ndarray): 3D volume.

    Returns:
        np.ndarray: 3D volume without edges.

    """
    return volume[width:volume.shape[0]-width, width:volume.shape[1]-width, width:volume.shape[2]-width]

def compute_structure_tensor(volume: np.ndarray, noise_scale: int, mode: str = 'nearest') -> np.ndarray:
    """
    Computes the structure tensor of a 3D volume. Based on the skimage feature structure_tensor function.

    Args:
        volume (np.ndarray): The input 3D volume.
        noise_scale (int): The scale for the Gaussian filter.
        mode (str): The mode for the Gaussian filter. Default is 'nearest'. 
                    See [skimage doc.](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.structure_tensor)

    Returns:
        np.ndarray: The structure tensor of the input volume.
    
    Raises:
        ValueError: If the input volume is not 3D.
    """
    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D.")
    tensor_list = structure_tensor(volume, sigma=noise_scale, mode=mode)
    tensors = np.empty((6, *tensor_list[0].shape), dtype=np.float32)
    for i, tensor in enumerate(tensor_list):
        tensors[i] = tensor
    return tensors

@numba.njit(parallel=True, cache=True)
def _orientation_function(structureTensor):
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
def _orientation_function_reference(structureTensor, reference_vector):

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

def compute_orientation(structure_tensor: np.ndarray, reference_vector=None) -> tuple:
    """ Compute orientation function.

    Args:
        structureTensor (np.ndarray): Structure tensor.
        reference_vector (list, optional): Reference vector. If None, the function computes the orientation without a reference vector.

    Returns:
        tuple: Orientation angles.
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
    
def compute_static_data(theta: np.ndarray, phi: np.ndarray, varphi: np.ndarray, drop=10) -> pd.DataFrame:
    """ Compute static data.

    Args:
        theta (np.ndarray): Theta angles.
        phi (np.ndarray): Phi angles.
        varphi (np.ndarray): Varphi angles.
        drop (int): Number of pixels to drop from edges.

    Returns:
        pd.DataFrame: Static data.
    """
    theta = drop_edges_3D(drop, theta)
    phi = drop_edges_3D(drop, phi)
    varphi = drop_edges_3D(drop, varphi)

    # Flatten the arrays
    flatten_theta = theta.ravel()
    flatten_phi = phi.ravel()
    flatten_varphi = varphi.ravel()

    # Histogram
    hist_theta, bins_theta = np.histogram(flatten_theta, bins=1000, density=True)
    hist_phi, bins_phi = np.histogram(flatten_phi, bins=1000, density=True)
    hist_varphi, bins_varphi = np.histogram(flatten_varphi, bins=1000, density=True)

    # Pandas Series
    hist_theta_series = pd.Series(hist_theta, name="Histgram")
    bin_theta_series = pd.Series(bins_theta[1:], name="Bin")
    hist_phi_series = pd.Series(hist_phi, name="Histgram")
    bin_phi_series = pd.Series(bins_phi[1:], name="Bin")
    hist_varphi_series = pd.Series(hist_varphi, name="Histgram")
    bin_varphi_series = pd.Series(bins_varphi[1:], name="Bin")

    # Pandas DF
    static_df_theta = pd.DataFrame([bin_theta_series, hist_theta_series], index=["Bin (theta)", "Histgram (theta)"]).transpose()
    static_df_phi = pd.DataFrame([bin_phi_series, hist_phi_series], index=["Bin (phi)", "Histgram (phi)"]).transpose()
    setatic_df_varphi = pd.DataFrame([bin_varphi_series, hist_varphi_series], index=["Bin (varphi)", "Histgram (varphi)"]).transpose()

    # Combine dataframes
    static_df = pd.concat([static_df_theta, static_df_phi, setatic_df_varphi], axis=1)

    return static_df