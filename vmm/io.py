import numpy as np
import cv2 as cv
import pydicom as dicom
from typing import Optional, Callable, Union
from vmm.logger import get_logger

logger = get_logger()

def drop_edges_3D(width: int, volume: np.ndarray) -> np.ndarray:
    """Drop edges of 3D volume.

    Args:
        width (int): Width of edges.
        volume (np.ndarray): 3D volume.

    Returns:
        np.ndarray: 3D volume without edges.

    """
    return volume[width:volume.shape[0]-width, width:volume.shape[1]-width, width:volume.shape[2]-width]

def trim_image(start: tuple[int, int], end: tuple[int, int], image: np.ndarray) -> np.ndarray:
    """Crop image.

    Args:
        start (tuple[int, int]): Start coordinate.
        end (tuple[int, int]): End coordinate.
        image (np.ndarray): Image.

    Returns:
        np.ndarray: Cropped image.

    """
    return image[start[0]:end[0], start[1]:end[1]]

def import_image(path: str) -> np.ndarray:
    """Import image from path and automatically convert to grayscale uint8.

    Supports various image formats including:
    - 8-bit grayscale/color images (PNG, JPEG, BMP, etc.)
    - 16-bit images (TIFF, PNG)
    - 32-bit float images (TIFF)
    - Multi-channel images (automatically converted to grayscale)

    Args:
        path (str): Path to the image.

    Returns:
        np.ndarray: Imported image as 2D grayscale uint8 array.

    Raises:
        FileNotFoundError: If the image file cannot be loaded.
    """
    logger.debug(f"Importing image: {path}")

    # Read image with IMREAD_UNCHANGED to preserve original format
    image = cv.imread(path, cv.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {path}")

    original_dtype = image.dtype
    original_shape = image.shape
    logger.debug(f"Original image: dtype={original_dtype}, shape={original_shape}")

    # Step 1: Handle different bit depths - convert to float for normalization
    if image.dtype == np.uint8:
        # 8-bit: already in 0-255 range
        image_normalized = image.astype(np.float32)
    elif image.dtype == np.uint16:
        # 16-bit: normalize from 0-65535 to 0-255
        image_normalized = (image.astype(np.float32) / 65535.0 * 255.0)
        logger.debug("Converted 16-bit image to 8-bit range")
    elif image.dtype == np.float32 or image.dtype == np.float64:
        # Float: normalize based on actual min/max values
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image_normalized = ((image - image_min) / (image_max - image_min) * 255.0).astype(np.float32)
        else:
            image_normalized = np.zeros_like(image, dtype=np.float32)
        logger.debug(f"Converted float image: min={image_min}, max={image_max}")
    else:
        # Other types: try direct conversion
        image_normalized = image.astype(np.float32)
        logger.warning(f"Unknown dtype {original_dtype}, attempting direct conversion")

    # Step 2: Convert to grayscale if needed
    if len(image_normalized.shape) == 3:
        if image_normalized.shape[2] == 3:
            # BGR to grayscale
            image_gray = cv.cvtColor(image_normalized.astype(np.uint8), cv.COLOR_BGR2GRAY).astype(np.float32)
        elif image_normalized.shape[2] == 4:
            # BGRA to grayscale
            image_gray = cv.cvtColor(image_normalized.astype(np.uint8), cv.COLOR_BGRA2GRAY).astype(np.float32)
        else:
            # Unknown channel count, take first channel
            image_gray = image_normalized[:, :, 0]
            logger.warning(f"Unknown channel count {image_normalized.shape[2]}, using first channel")
    else:
        # Already grayscale
        image_gray = image_normalized

    # Step 3: Convert to uint8
    image_uint8 = np.clip(image_gray, 0, 255).astype(np.uint8)

    logger.debug(f"Image imported: shape={image_uint8.shape}, dtype={image_uint8.dtype}")
    return image_uint8

def import_dicom(path: str) -> np.ndarray:
    """Import DICOM image from path and convert to grayscale uint8.

    Args:
        path (str): Path to the DICOM file.

    Returns:
        np.ndarray: Imported image as 2D grayscale uint8 array.
    """
    logger.debug(f"Importing DICOM: {path}")
    dicom_image = dicom.dcmread(path)
    image = dicom_image.pixel_array
    original_dtype = image.dtype
    logger.debug(f"Original DICOM: dtype={original_dtype}, shape={image.shape}")

    # Normalize to 0-255 range
    image_float = image.astype(np.float32)
    image_min = image_float.min()
    image_max = image_float.max()
    if image_max > image_min:
        image_normalized = ((image_float - image_min) / (image_max - image_min) * 255.0)
    else:
        image_normalized = np.zeros_like(image_float)
    logger.debug(f"DICOM normalized: min={image_min}, max={image_max}")

    # Convert to uint8
    image_uint8 = np.clip(image_normalized, 0, 255).astype(np.uint8)
    logger.debug(f"DICOM imported: shape={image_uint8.shape}, dtype={image_uint8.dtype}")
    return image_uint8

def normalize_intensity(volume: np.ndarray, method: str = 'mean') -> np.ndarray:
    """Normalize intensity across slices to correct for inter-slice brightness variations.

    CT scan slices often have slight brightness differences due to X-ray source
    fluctuations or detector variations. This function corrects these differences
    by normalizing each slice to a common reference.

    Args:
        volume: 3D volume array with shape (z, y, x).
        method: Normalization method. Options:
            - 'mean': Adjust each slice to have the same mean intensity as the volume mean.
            - 'histogram': Match histogram of each slice to the first slice (CLAHE-like).
            - 'minmax': Normalize each slice to use full 0-255 range based on global min/max.

    Returns:
        Normalized 3D volume as uint8 array.
    """
    logger.info(f"Normalizing intensity across {volume.shape[0]} slices using method='{method}'")

    volume_float = volume.astype(np.float32)

    if method == 'mean':
        # Calculate global mean intensity
        global_mean = volume_float.mean()
        logger.debug(f"Global mean intensity: {global_mean:.2f}")

        # Normalize each slice to have the same mean
        normalized = np.zeros_like(volume_float)
        for i in range(volume.shape[0]):
            slice_mean = volume_float[i].mean()
            if slice_mean > 0:
                # Scale slice to match global mean
                scale_factor = global_mean / slice_mean
                normalized[i] = volume_float[i] * scale_factor
            else:
                normalized[i] = volume_float[i]

        # Clip and convert to uint8
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    elif method == 'histogram':
        # Use the first slice as reference for histogram matching
        reference_slice = volume_float[0]
        normalized = np.zeros_like(volume_float)
        normalized[0] = reference_slice

        for i in range(1, volume.shape[0]):
            # Simple histogram matching using cumulative distribution
            source = volume_float[i]

            # Compute histograms
            ref_hist, bins = np.histogram(reference_slice.flatten(), bins=256, range=(0, 255))
            src_hist, _ = np.histogram(source.flatten(), bins=256, range=(0, 255))

            # Compute CDFs
            ref_cdf = np.cumsum(ref_hist).astype(np.float64)
            ref_cdf = ref_cdf / ref_cdf[-1]  # Normalize

            src_cdf = np.cumsum(src_hist).astype(np.float64)
            src_cdf = src_cdf / src_cdf[-1]  # Normalize

            # Create lookup table
            lookup = np.zeros(256)
            for j in range(256):
                diff = np.abs(ref_cdf - src_cdf[j])
                lookup[j] = np.argmin(diff)

            # Apply lookup table
            normalized[i] = lookup[np.clip(source, 0, 255).astype(np.uint8)]

        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    elif method == 'minmax':
        # Normalize using global min/max
        global_min = volume_float.min()
        global_max = volume_float.max()
        logger.debug(f"Global intensity range: [{global_min:.2f}, {global_max:.2f}]")

        if global_max > global_min:
            normalized = ((volume_float - global_min) / (global_max - global_min) * 255.0)
        else:
            normalized = np.zeros_like(volume_float)

        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    else:
        logger.warning(f"Unknown normalization method '{method}', returning original volume")
        return volume

    logger.info(f"Intensity normalization complete")
    return normalized


def get_image_path(path_template: str,
                   index_of_image: int,
                   number_of_digit: int,
                   format: str) -> str:
    """Get paths of images.

    Args:
        path_template (str): Path template of image.
        index_of_image (int): Index of the image.
        number_of_digit (int): Number of digits.
        format (str): Format of image.

    Returns:
        str: Path of image.

    """
    digit = f'{index_of_image}'.zfill(number_of_digit)
    path_of_image = path_template + digit + '.' + format
    return path_of_image

def import_image_sequence(path_template: str,
                          number_of_images: int,
                          number_of_digits: int,
                          format: str,
                          initial_number: Optional[int] = 0,
                          path_for_save: Optional[str] = None,
                          process: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                          normalize: Optional[str] = None) -> np.ndarray:
    """Import image sequence as volume.

    Array convention: (axial, d1, d0)
        - axial: Fiber axis direction (image sequence/slice index)
        - d1: Cross-section first direction (row, increases downward in image)
        - d0: Cross-section second direction (column, increases rightward in image)

    All images are automatically converted to grayscale uint8 format.

    Args:
        path_template (str): Path template of image sequence.
        number_of_images (int): Number of images to import.
        number_of_digits (int): Number of digits in the file numbering.
        format (str): Format of image (e.g., 'tif', 'png', 'dcm').
        initial_number (Optional[int]): Starting index for image sequence. Defaults to 0.
        path_for_save (Optional[str]): Path to save the imported volume as .npy file.
        process (Optional[Callable[[np.ndarray], np.ndarray]]): Processing function applied to each image.
        normalize (Optional[str]): Intensity normalization method to correct inter-slice
            brightness variations. Options:
            - None: No normalization (default).
            - 'mean': Adjust each slice to have the same mean intensity.
            - 'histogram': Match histogram of each slice to the first slice.
            - 'minmax': Normalize using global min/max values.

    Returns:
        np.ndarray: Imported image sequence as 3D grayscale volume (uint8).

    """
    logger.info(f"Importing image sequence: path_template={path_template}, n_images={number_of_images}, format={format}")
    logger.debug(f"Sequence range: [{initial_number}, {initial_number + number_of_images})")

    if process is None:
        process = lambda image: image

    # Calculate end index: start from initial_number and import number_of_images
    end_number = initial_number + number_of_images

    if format == "dcm":
        volume = np.stack(
        [process(import_dicom(get_image_path(path_template, i, number_of_digits, format)))
         for i in range(initial_number, end_number)], axis=0)
    else:
        volume = np.stack(
        [process(import_image(get_image_path(path_template, i, number_of_digits, format)))
         for i in range(initial_number, end_number)], axis=0)

    logger.info(f"Image sequence loaded: volume shape={volume.shape}, dtype={volume.dtype}")

    # Apply intensity normalization if requested
    if normalize is not None:
        volume = normalize_intensity(volume, method=normalize)

    if path_for_save is not None:
        logger.debug(f"Saving volume to: {path_for_save}")
        np.save(path_for_save, volume)

    return volume


def export_orientation_to_vtk(filepath: str, tilt_d0: np.ndarray, tilt_d1: np.ndarray = None,
                              origin_offset: int = 0) -> None:
    """Export orientation data to VTK format for ParaView visualization.

    Internal coordinate system (right-hand, Y-down):
        - X (d0): column, increases rightward
        - Y (d1): row, increases downward
        - Z (axial): fiber direction, into screen

    VTK coordinate system (right-hand, Y-up):
        - VTK X = d0 (column, rightward)
        - VTK Y = d1 (row index, but VTK displays Y-up)
        - VTK Z = axial (fiber direction)

    Note: VTK standard is Y-up, so the visualization in ParaView will show
    the Y-axis pointing upward. To match the GUI display (Y-down), rotate
    the view in ParaView by 180° around the X-axis.

    Args:
        filepath: Output file path (.vtk or .vti extension).
        tilt_d0: 3D array with shape (axial, d1, d0) containing tilt angles in d0-axial plane.
        tilt_d1: Optional 3D array with shape (axial, d1, d0) containing tilt angles in d1-axial plane.
        origin_offset: Offset in pixels to align with CT volume (e.g., noise_scale from
                       structure tensor computation). Applied equally to all axes.

    """
    import pyvista as pv

    logger.info(f"Exporting orientation to VTK: {filepath}")
    logger.debug(f"Input shape: {tilt_d0.shape}, origin_offset: {origin_offset}")

    # Transpose from internal (axial, d1, d0) to VTK (d0, d1, axial) = (X, Y, Z)
    tilt_d0_vtk = np.transpose(tilt_d0, (2, 1, 0))

    # Create VTK ImageData (uniform grid)
    grid = pv.ImageData()
    grid.dimensions = np.array(tilt_d0_vtk.shape) + 1  # VTK needs n+1 for cell data
    grid.spacing = (1, 1, 1)
    # Set origin offset to align with CT volume center
    # Structure tensor computation cuts edges by noise_scale pixels on each side
    grid.origin = (origin_offset, origin_offset, origin_offset)

    # Add tilt_d0 as cell data (Fortran order for VTK)
    grid.cell_data['Tilt_d0'] = tilt_d0_vtk.flatten(order='F')

    # Add tilt_d1 if provided
    if tilt_d1 is not None:
        tilt_d1_vtk = np.transpose(tilt_d1, (2, 1, 0))
        grid.cell_data['Tilt_d1'] = tilt_d1_vtk.flatten(order='F')

    grid.save(filepath)
    logger.info(f"Orientation exported: shape={tilt_d0.shape} -> VTK dimensions={grid.dimensions}, origin={grid.origin}")


def export_void_to_vtk(filepath: str, void_mask: np.ndarray,
                       roi_bounds: list = None, polygon_mask: np.ndarray = None) -> dict:
    """Export void regions as closed surface mesh to VTK format.

    Uses marching cubes algorithm to generate an isosurface mesh from the binary void mask.

    Internal coordinate system (right-hand, Y-down):
        - X (d0): column, increases rightward
        - Y (d1): row, increases downward
        - Z (axial): fiber direction, into screen

    VTK coordinate system (right-hand, Y-up):
        - VTK X = d0 (column, rightward)
        - VTK Y = d1 (row index, but VTK displays Y-up)
        - VTK Z = axial (fiber direction)

    Args:
        filepath: Output file path (.vtk or .vtp extension).
        void_mask: 3D boolean array with shape (axial, d1, d0) containing void regions (True = void).
        roi_bounds: Optional [z_min, z_max, y_min, y_max, x_min, x_max] to offset mesh origin.
        polygon_mask: Optional 3D boolean mask to apply to void_mask before export.

    Returns:
        dict: Statistics about the exported mesh (n_vertices, n_faces, volume, surface_area).
    """
    import pyvista as pv
    from skimage import measure

    logger.info(f"Exporting void surface to VTK: {filepath}")
    logger.debug(f"Void mask shape: {void_mask.shape}")

    # Apply polygon mask if provided
    if polygon_mask is not None:
        void_mask = void_mask & polygon_mask
        logger.debug("Applied polygon mask to void mask")

    # Check if there are any voids
    if not np.any(void_mask):
        logger.warning("No void regions found in mask")
        return {'n_vertices': 0, 'n_faces': 0, 'volume': 0, 'surface_area': 0}

    # Convert boolean mask to float for marching cubes
    # Pad with zeros to ensure closed surface at boundaries
    void_padded = np.pad(void_mask.astype(np.float32), pad_width=1, mode='constant', constant_values=0)

    # Run marching cubes to generate surface mesh
    # Input shape is (axial, d1, d0), marching cubes returns vertices as (z, y, x)
    verts, faces, normals, values = measure.marching_cubes(void_padded, level=0.5)

    # Adjust vertices for padding offset (-1 on each axis)
    verts = verts - 1.0

    # Convert from (z, y, x) to VTK (x, y, z) coordinate order
    # verts[:, 0] = z (axial), verts[:, 1] = y (d1), verts[:, 2] = x (d0)
    verts_vtk = np.column_stack([verts[:, 2], verts[:, 1], verts[:, 0]])

    # Apply ROI offset if provided
    if roi_bounds is not None:
        z_min, z_max, y_min, y_max, x_min, x_max = roi_bounds
        verts_vtk[:, 0] += x_min  # X offset
        verts_vtk[:, 1] += y_min  # Y offset
        verts_vtk[:, 2] += z_min  # Z offset
        logger.debug(f"Applied ROI offset: x+{x_min}, y+{y_min}, z+{z_min}")

    # Create PyVista PolyData mesh
    # faces need to be converted to VTK format: [n_verts, v0, v1, v2, ...]
    n_faces = len(faces)
    faces_vtk = np.column_stack([np.full(n_faces, 3), faces]).flatten()

    mesh = pv.PolyData(verts_vtk, faces_vtk)

    # Compute mesh statistics
    mesh_volume = mesh.volume if mesh.n_cells > 0 else 0
    mesh_area = mesh.area if mesh.n_cells > 0 else 0

    # Save mesh
    mesh.save(filepath)

    stats = {
        'n_vertices': mesh.n_points,
        'n_faces': mesh.n_cells,
        'volume': mesh_volume,
        'surface_area': mesh_area
    }

    logger.info(f"Void surface exported: {stats['n_vertices']} vertices, {stats['n_faces']} faces, "
                f"volume={stats['volume']:.1f} voxels³, area={stats['surface_area']:.1f} voxels²")

    return stats