"""
Fiber trajectory generation module.

This module provides functionality to generate fiber trajectories based on
orientation tensor data from CT scan analysis.
"""

import numpy as np
from scipy.stats.qmc import PoissonDisk
from scipy.spatial import KDTree
from scipy.ndimage import map_coordinates, distance_transform_edt, gaussian_filter1d
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
import numba
import pyvista as pv


class FiberTrajectory:
    """
    Class to generate and manage fiber trajectories in a 3D domain.

    Fiber initial positions are generated using Poisson disk sampling to ensure
    non-overlapping distribution. Trajectories are computed by propagating fibers
    along the direction specified by the reference vector.

    The propagation axis is determined by the reference vector:
    - [1, 0, 0] or similar -> propagate along X axis (sample in YZ plane)
    - [0, 1, 0] or similar -> propagate along Y axis (sample in XZ plane)
    - [0, 0, 1] or similar -> propagate along Z axis (sample in XY plane, default)
    """

    def __init__(self, status_callback=None):
        self.points = None
        self.bounds = None
        self.fiber_diameter = None
        self.fiber_volume_fraction = None
        self.trajectories = []  # List of (axis_position, points) tuples
        self.angles = []  # List of misalignment angles (tilt from axis) at each step
        self.azimuths = []  # List of azimuthal angles (direction in cross-section plane) at each step
        # Per-fiber trajectory data (for variable-length trajectories)
        self.fiber_trajectories = []  # List of per-fiber trajectories
        self.fiber_angles = []  # List of per-fiber angles
        self.fiber_azimuths = []  # List of per-fiber azimuthal angles
        self.active_fibers = None  # Boolean mask of active fibers
        # Propagation axis: 0=Z (default), 1=Y, 2=X
        self.propagation_axis = 0
        self.reference_vector = np.array([0.0, 0.0, 1.0])
        # Status callback for GUI status updates
        self.status_callback = status_callback

    def _log(self, message):
        """Log a message to status callback or print."""
        if self.status_callback:
            self.status_callback(message)
        else:
            print(message)

    def initialize(
        self,
        shape: tuple,
        fiber_diameter: float,
        fiber_volume_fraction: float,
        scale: float = 1.0,
        seed: int = 42,
        reference_vector: list = None
    ) -> np.ndarray:
        """
        Initialize fiber positions using Poisson disk sampling.

        Args:
            shape: Shape of the domain (z, y, x).
            fiber_diameter: Diameter of the fibers in pixels.
            fiber_volume_fraction: Target volume fraction of fibers (0-1).
            scale: Scale factor for minimum distance between fibers.
            seed: Random seed for reproducibility.
            reference_vector: Reference direction [x, y, z] for fiber axis.
                             Determines the propagation axis and sampling plane.
                             Default is [0, 0, 1] (Z-axis, sample in XY plane).

        Returns:
            Initial fiber center positions as (N, 2) array.
        """
        self.bounds = shape
        self.fiber_diameter = fiber_diameter
        self.fiber_volume_fraction = fiber_volume_fraction

        # Determine propagation axis from reference vector
        if reference_vector is None:
            reference_vector = [0.0, 0.0, 1.0]
        self.reference_vector = np.array(reference_vector, dtype=np.float32)
        ref_norm = np.linalg.norm(self.reference_vector)
        if ref_norm > 0:
            self.reference_vector = self.reference_vector / ref_norm

        # Determine which axis is the primary fiber direction
        # Find the component with largest absolute value
        abs_ref = np.abs(self.reference_vector)
        primary_axis = np.argmax(abs_ref)  # 0=x, 1=y, 2=z

        # Map primary axis to propagation axis
        # primary_axis: 0=X -> propagation along X (sample in YZ plane)
        #               1=Y -> propagation along Y (sample in XZ plane)
        #               2=Z -> propagation along Z (sample in XY plane)
        self.propagation_axis = primary_axis

        # Determine cross-section dimensions based on propagation axis
        # shape is (z, y, x) -> indices (0, 1, 2)
        if self.propagation_axis == 0:  # X-axis propagation, sample in YZ plane
            cross_section_dims = (shape[0], shape[1])  # (z, y)
            n_slices = shape[2]  # x
            self._log(f"Propagation axis: X, sampling in YZ plane ({cross_section_dims[0]}x{cross_section_dims[1]})")
        elif self.propagation_axis == 1:  # Y-axis propagation, sample in XZ plane
            cross_section_dims = (shape[0], shape[2])  # (z, x)
            n_slices = shape[1]  # y
            self._log(f"Propagation axis: Y, sampling in XZ plane ({cross_section_dims[0]}x{cross_section_dims[1]})")
        else:  # Z-axis propagation (default), sample in XY plane
            cross_section_dims = (shape[1], shape[2])  # (y, x)
            n_slices = shape[0]  # z
            self._log(f"Propagation axis: Z, sampling in XY plane ({cross_section_dims[0]}x{cross_section_dims[1]})")

        # Calculate number of fibers based on volume fraction
        cross_section_area = cross_section_dims[0] * cross_section_dims[1]
        fiber_area = np.pi / 4 * fiber_diameter ** 2
        num_fibers = int(cross_section_area * fiber_volume_fraction / fiber_area)
        self._log(f"Target fibers: {num_fibers} (Vf={fiber_volume_fraction}, area={cross_section_area})")

        # Poisson disk sampling in actual cross-section dimensions
        # Use the actual dimensions for proper scaling
        dim0 = cross_section_dims[0]  # First dimension (y for Z-axis, z for X/Y-axis)
        dim1 = cross_section_dims[1]  # Second dimension (x for Z-axis)

        # The minimum distance between fiber centers is fiber_diameter * scale
        min_distance = fiber_diameter * scale

        # Normalize radius relative to the maximum dimension for PoissonDisk
        max_dim = max(dim0, dim1)
        normalized_radius = min_distance / max_dim

        # PoissonDisk samples in [0, 1]^d, then we scale to actual dimensions
        # Request more samples than needed because some will be filtered out
        n_samples_requested = max(num_fibers * 4, 100000)  # Request at least 4x or 100000

        sampler = PoissonDisk(d=2, radius=normalized_radius, seed=seed)
        normalized_points = sampler.random(n_samples_requested)

        # Scale to actual cross-section dimensions
        # normalized_points[:, 0] -> scale to dim1 (x direction for Z-axis)
        # normalized_points[:, 1] -> scale to dim0 (y direction for Z-axis)
        all_points = np.zeros_like(normalized_points)
        all_points[:, 0] = normalized_points[:, 0] * max_dim  # x coordinate
        all_points[:, 1] = normalized_points[:, 1] * max_dim  # y coordinate

        # Filter points to be within actual cross-section bounds
        # points[:, 0] is x (should be < dim1), points[:, 1] is y (should be < dim0)
        valid_mask = (all_points[:, 0] >= 0) & (all_points[:, 0] < dim1) & \
                     (all_points[:, 1] >= 0) & (all_points[:, 1] < dim0)
        points = all_points[valid_mask]

        self._log(f"Generated {len(points)} valid fiber positions")

        # Limit to target number of fibers
        if len(points) > num_fibers:
            points = points[:num_fibers]

        self._log(f"Final number of fibers: {len(points)}")

        self.points = points
        self.trajectories = [(0, points.copy())]
        self.angles = [np.zeros(len(points))]
        self.azimuths = [np.zeros(len(points))]

        # Initialize per-fiber trajectory data
        n_fibers = len(points)
        self.fiber_trajectories = [[(0, points[i].copy())] for i in range(n_fibers)]
        self.fiber_angles = [[0.0] for _ in range(n_fibers)]
        self.fiber_azimuths = [[0.0] for _ in range(n_fibers)]
        self.active_fibers = np.ones(n_fibers, dtype=bool)

        return points

    def initialize_from_image(
        self,
        image: np.ndarray,
        shape: tuple,
        min_diameter: float = 5.0,
        max_diameter: float = 20.0,
        min_distance: int = 5,
        reference_vector: list = None,
        exclude_boundary: bool = True,
        boundary_margin: float = None
    ) -> np.ndarray:
        """
        Initialize fiber positions by detecting fiber centers from an image.

        Args:
            image: 2D grayscale image of fiber cross-section.
            shape: Shape of the 3D domain (z, y, x).
            min_diameter: Minimum fiber diameter in pixels to accept.
            max_diameter: Maximum fiber diameter in pixels to accept.
            min_distance: Minimum distance between detected peaks.
            reference_vector: Reference direction [x, y, z] for fiber axis.
            exclude_boundary: If True, exclude fibers near the boundary from tracking.
            boundary_margin: Margin from boundary to exclude fibers. If None, uses fiber_diameter/2.

        Returns:
            Detected fiber center positions as (N, 2) array.
        """
        self.bounds = shape

        # Determine propagation axis from reference vector
        if reference_vector is None:
            reference_vector = [0.0, 0.0, 1.0]
        self.reference_vector = np.array(reference_vector, dtype=np.float32)
        ref_norm = np.linalg.norm(self.reference_vector)
        if ref_norm > 0:
            self.reference_vector = self.reference_vector / ref_norm

        abs_ref = np.abs(self.reference_vector)
        self.propagation_axis = np.argmax(abs_ref)

        # Detect fiber centers from image
        centers, diameters = detect_fiber_centers(
            image,
            min_diameter=min_diameter,
            max_diameter=max_diameter,
            min_distance=min_distance
        )

        if len(centers) == 0:
            raise ValueError("No fibers detected in the image")

        # Store mean diameter
        self.fiber_diameter = np.mean(diameters)
        self._log(f"Detected {len(centers)} fibers, mean diameter: {self.fiber_diameter:.2f} px")

        # Points are (x, y) from detection, need to convert to (dim0, dim1) format
        # For Z-axis propagation: dim0=x, dim1=y (same as detection output)
        points = centers.copy()

        self.points = points
        self.trajectories = [(0, points.copy())]
        self.angles = [np.zeros(len(points))]
        self.azimuths = [np.zeros(len(points))]

        # Initialize per-fiber trajectory data
        n_fibers = len(points)
        self.fiber_trajectories = [[(0, points[i].copy())] for i in range(n_fibers)]
        self.fiber_angles = [[0.0] for _ in range(n_fibers)]
        self.fiber_azimuths = [[0.0] for _ in range(n_fibers)]
        self.active_fibers = np.ones(n_fibers, dtype=bool)

        # Exclude fibers near the boundary from tracking
        if exclude_boundary:
            # Determine boundary margin
            if boundary_margin is None:
                boundary_margin = self.fiber_diameter / 2.0

            # Get domain size based on propagation axis
            if self.propagation_axis == 2:  # Z-axis
                dim0_max = shape[1]  # y
                dim1_max = shape[2]  # x
            elif self.propagation_axis == 1:  # Y-axis
                dim0_max = shape[0]  # z
                dim1_max = shape[2]  # x
            else:  # X-axis
                dim0_max = shape[0]  # z
                dim1_max = shape[1]  # y

            # Mark fibers near boundary as inactive
            near_boundary = (
                (points[:, 0] < boundary_margin) |
                (points[:, 0] > dim1_max - boundary_margin) |
                (points[:, 1] < boundary_margin) |
                (points[:, 1] > dim0_max - boundary_margin)
            )
            self.active_fibers = ~near_boundary
            n_excluded = np.sum(near_boundary)
            if n_excluded > 0:
                self._log(f"Excluded {n_excluded} fibers near boundary (margin={boundary_margin:.1f}px)")

        return points

    def get_num_fibers(self) -> int:
        """Return the number of fibers."""
        return len(self.points) if self.points is not None else 0

    def get_initial_points(self) -> np.ndarray:
        """Return the initial fiber positions."""
        return self.points.copy() if self.points is not None else None

    def get_trajectories(self) -> list:
        """Return all trajectory data (slice-based format for backward compatibility)."""
        return self.trajectories.copy()

    def get_fiber_trajectories(self) -> list:
        """Return per-fiber trajectory data (variable length trajectories)."""
        return self.fiber_trajectories.copy()

    def get_fiber_angles(self) -> list:
        """Return per-fiber angle data (variable length)."""
        return self.fiber_angles.copy()

    def get_angles(self) -> list:
        """Return misalignment angles (tilt from axis) at each trajectory step."""
        return self.angles.copy()

    def get_azimuths(self) -> list:
        """Return azimuthal angles (direction in cross-section plane, 0-360°) at each trajectory step."""
        return self.azimuths.copy()

    def propagate(
        self,
        structure_tensor: np.ndarray,
        relax: bool = True,
        relax_iterations: int = 100,
        stop_at_boundary: bool = True,
        boundary_margin: float = 0.5,
        resample_interval: int = 0,
        resample_seed: int = None
    ) -> None:
        """
        Propagate fibers through all slices based on structure tensor orientation.

        The propagation direction is determined by the reference vector set during
        initialization. For each slice along the propagation axis, computes the
        local fiber direction from the structure tensor eigenvector and moves
        fiber centers accordingly.

        Args:
            structure_tensor: 4D array with shape (6, z, y, x) containing
                             the symmetric structure tensor components.
            relax: Whether to apply relaxation to avoid fiber overlaps.
            relax_iterations: Number of iterations for relaxation.
            stop_at_boundary: If True, stop tracking fibers that exit the domain.
            boundary_margin: Margin from boundary to consider as "out of bounds" (in pixels).
            resample_interval: If > 0, resample new fibers in empty regions every N slices.
            resample_seed: Random seed for resampling (uses slice-based seed if None).
        """
        if self.points is None:
            raise ValueError("Fibers not initialized. Call initialize() first.")

        # Determine number of slices and domain bounds based on propagation axis
        # structure_tensor shape is (6, z, y, x)
        if self.propagation_axis == 0:  # X-axis propagation
            n_slices = structure_tensor.shape[3]  # x dimension
            # Cross-section is YZ plane, points are (dim0=y or z, dim1=z or y)
            # We use (z, y) order, so dim0_max = z, dim1_max = y
            dim0_max = structure_tensor.shape[1] - 1  # z
            dim1_max = structure_tensor.shape[2] - 1  # y
            axis_name = "X"
        elif self.propagation_axis == 1:  # Y-axis propagation
            n_slices = structure_tensor.shape[2]  # y dimension
            # Cross-section is XZ plane, points are (x, z)
            dim0_max = structure_tensor.shape[3] - 1  # x
            dim1_max = structure_tensor.shape[1] - 1  # z
            axis_name = "Y"
        else:  # Z-axis propagation (default)
            n_slices = structure_tensor.shape[1]  # z dimension
            # Cross-section is XY plane, points are (x, y)
            dim0_max = structure_tensor.shape[3] - 1  # x
            dim1_max = structure_tensor.shape[2] - 1  # y
            axis_name = "Z"

        n_fibers = len(self.points)
        print(f"[INFO] Propagating {n_fibers} fibers along {axis_name}-axis through {n_slices} slices...")
        if resample_interval > 0:
            print(f"[INFO] Resampling enabled every {resample_interval} slices")

        # Compute eigenvectors for all slices
        eigenvectors = _compute_eigenvectors(structure_tensor, self.reference_vector)

        current_points = self.points.copy()

        # Track active fibers (those still within bounds)
        active = self.active_fibers.copy()
        stopped_count = 0
        resampled_count = 0

        for s in range(1, n_slices):
            if s % 50 == 0:
                active_count = np.sum(active)
                print(f"[INFO] Processing slice {s}/{n_slices} (active fibers: {active_count}, total: {n_fibers})")

            # Get direction field at current slice
            # eigenvectors shape: (4, n_slices, dim1, dim0) where:
            # - [0]: displacement in dim0 direction per unit step
            # - [1]: displacement in dim1 direction per unit step
            # - [2]: component along propagation axis (for angle calculation)
            # - [3]: anisotropy (confidence)
            ev_d0 = eigenvectors[0, s]  # displacement in dim0 direction
            ev_d1 = eigenvectors[1, s]  # displacement in dim1 direction
            ev_axial = eigenvectors[2, s]  # component along propagation axis
            ev_aniso = eigenvectors[3, s]  # anisotropy (confidence)

            # Interpolate direction at each fiber position
            # points are (dim0, dim1)
            d0_coords = current_points[:, 0]
            d1_coords = current_points[:, 1]

            # Clamp coordinates to valid range for interpolation
            d0_coords_clamped = np.clip(d0_coords, 0, ev_d0.shape[1] - 1)
            d1_coords_clamped = np.clip(d1_coords, 0, ev_d0.shape[0] - 1)

            coords = np.array([d1_coords_clamped, d0_coords_clamped])

            v_d0 = map_coordinates(ev_d0, coords, order=1, mode='nearest')
            v_d1 = map_coordinates(ev_d1, coords, order=1, mode='nearest')
            v_axial = map_coordinates(ev_axial, coords, order=1, mode='nearest')
            aniso = map_coordinates(ev_aniso, coords, order=1, mode='nearest')

            # Compute displacement per unit step along propagation axis
            # d(dim0)/d(axis) = v_d0/v_axial, d(dim1)/d(axis) = v_d1/v_axial
            v_axial_safe = np.where(np.abs(v_axial) > 1e-6, v_axial, 1e-6)
            d_d0 = v_d0 / v_axial_safe
            d_d1 = v_d1 / v_axial_safe

            # Clamp displacement to physically reasonable range
            max_displacement = 1.0  # Maximum pixels per slice
            d_d0 = np.clip(d_d0, -max_displacement, max_displacement)
            d_d1 = np.clip(d_d1, -max_displacement, max_displacement)

            # Compute misalignment angle (angle from propagation axis)
            angles = np.rad2deg(np.arctan2(np.sqrt(v_d0**2 + v_d1**2), np.abs(v_axial)))

            # Compute azimuthal angle (direction in cross-section plane, 0-360 degrees)
            # atan2 returns -180 to 180, convert to 0-360
            azimuths = np.rad2deg(np.arctan2(v_d1, v_d0))
            azimuths = np.mod(azimuths, 360)  # Convert to 0-360 range

            # Move points based on direction
            new_points = current_points.copy()
            new_points[:, 0] += d_d0
            new_points[:, 1] += d_d1

            # Check for boundary crossing (before clamping)
            # Stop tracking when fiber center approaches boundary by fiber radius
            if stop_at_boundary:
                fiber_radius = self.fiber_diameter / 2.0 if self.fiber_diameter else boundary_margin
                effective_margin = max(fiber_radius, boundary_margin)

                out_of_bounds = (
                    (new_points[:, 0] < effective_margin) |
                    (new_points[:, 0] > dim0_max - effective_margin) |
                    (new_points[:, 1] < effective_margin) |
                    (new_points[:, 1] > dim1_max - effective_margin)
                )
                # Mark fibers that just went out of bounds as inactive
                newly_stopped = active & out_of_bounds
                stopped_count += np.sum(newly_stopped)
                active = active & ~out_of_bounds

            # Resample fibers in empty regions at specified intervals
            if resample_interval > 0 and s % resample_interval == 0:
                new_fiber_points, new_fiber_count = self._resample_empty_regions(
                    new_points, active, dim0_max + 1, dim1_max + 1,
                    seed=resample_seed if resample_seed else s
                )
                if new_fiber_count > 0:
                    # Add new fibers
                    new_points = np.vstack([new_points, new_fiber_points])
                    active = np.concatenate([active, np.ones(new_fiber_count, dtype=bool)])
                    # Extend trajectory storage for new fibers
                    for i in range(new_fiber_count):
                        self.fiber_trajectories.append([(s, new_fiber_points[i].copy())])
                        self.fiber_angles.append([0.0])
                        self.fiber_azimuths.append([0.0])
                    # Update angles/azimuths arrays for new fibers
                    angles = np.concatenate([angles, np.zeros(new_fiber_count)])
                    azimuths = np.concatenate([azimuths, np.zeros(new_fiber_count)])
                    resampled_count += new_fiber_count
                    n_fibers = len(new_points)

            # Apply relaxation to avoid overlaps (only for active fibers)
            if relax and np.sum(active) > 1:
                # Only relax active fibers
                active_indices = np.where(active)[0]
                if len(active_indices) > 1:
                    active_points = new_points[active_indices]
                    relaxed_points = self._relax_points(active_points, self.fiber_diameter, relax_iterations)
                    new_points[active_indices] = relaxed_points

            # Store per-fiber trajectory data
            for i in range(n_fibers):
                if i < len(self.fiber_trajectories):
                    if active[i] or not stop_at_boundary:
                        # Check if this fiber already has data for this slice (from resampling)
                        if len(self.fiber_trajectories[i]) == 0 or self.fiber_trajectories[i][-1][0] != s:
                            self.fiber_trajectories[i].append((s, new_points[i].copy()))
                            self.fiber_angles[i].append(angles[i])
                            if i < len(self.fiber_azimuths):
                                self.fiber_azimuths[i].append(azimuths[i])

            # Store trajectory (for backward compatibility, store all points)
            self.trajectories.append((s, new_points.copy()))
            self.angles.append(angles)
            self.azimuths.append(azimuths)

            current_points = new_points

            # Early exit if all fibers have stopped
            if stop_at_boundary and np.sum(active) == 0:
                print(f"[INFO] All fibers have exited the domain at slice {s}")
                break

        self.active_fibers = active
        msg = f"[INFO] Propagation complete. {stopped_count} fibers stopped at boundary."
        if resampled_count > 0:
            msg += f" {resampled_count} new fibers resampled."
        print(msg)

    def propagate_rk4(
        self,
        structure_tensor: np.ndarray,
        relax: bool = True,
        relax_iterations: int = 100,
        stop_at_boundary: bool = True,
        boundary_margin: float = 0.5,
        resample_interval: int = 0,
        resample_seed: int = None
    ) -> None:
        """
        Propagate fibers using 4th-order Runge-Kutta integration (RK4).

        This method provides higher accuracy than the standard Euler method,
        especially for curved fiber trajectories. The RK4 method has 4th-order
        accuracy O(h^4) compared to Euler's 1st-order accuracy O(h).

        Args:
            structure_tensor: 4D array with shape (6, z, y, x) containing
                             the symmetric structure tensor components.
            relax: Whether to apply relaxation to avoid fiber overlaps.
            relax_iterations: Number of iterations for relaxation.
            stop_at_boundary: If True, stop tracking fibers that exit the domain.
            boundary_margin: Margin from boundary to consider as "out of bounds" (in pixels).
            resample_interval: If > 0, resample new fibers in empty regions every N slices.
            resample_seed: Random seed for resampling (uses slice-based seed if None).
        """
        if self.points is None:
            raise ValueError("Fibers not initialized. Call initialize() first.")

        # Determine number of slices and domain bounds based on propagation axis
        if self.propagation_axis == 0:  # X-axis propagation
            n_slices = structure_tensor.shape[3]
            dim0_max = structure_tensor.shape[1] - 1
            dim1_max = structure_tensor.shape[2] - 1
            axis_name = "X"
        elif self.propagation_axis == 1:  # Y-axis propagation
            n_slices = structure_tensor.shape[2]
            dim0_max = structure_tensor.shape[3] - 1
            dim1_max = structure_tensor.shape[1] - 1
            axis_name = "Y"
        else:  # Z-axis propagation (default)
            n_slices = structure_tensor.shape[1]
            dim0_max = structure_tensor.shape[3] - 1
            dim1_max = structure_tensor.shape[2] - 1
            axis_name = "Z"

        n_fibers = len(self.points)
        print(f"[INFO] RK4 Propagating {n_fibers} fibers along {axis_name}-axis through {n_slices} slices...")
        if resample_interval > 0:
            print(f"[INFO] Resampling enabled every {resample_interval} slices")

        # Compute eigenvectors for all slices
        eigenvectors = _compute_eigenvectors(structure_tensor, self.reference_vector)

        current_points = self.points.copy()
        active = self.active_fibers.copy()
        stopped_count = 0
        resampled_count = 0

        def get_displacement(points, slice_idx):
            """Get displacement field at given points and slice index (with fractional support)."""
            # Handle fractional slice indices by interpolation
            s_floor = int(np.floor(slice_idx))
            s_ceil = int(np.ceil(slice_idx))
            s_frac = slice_idx - s_floor

            # Clamp to valid range
            s_floor = max(0, min(s_floor, n_slices - 1))
            s_ceil = max(0, min(s_ceil, n_slices - 1))

            def sample_at_slice(s):
                ev_d0 = eigenvectors[0, s]
                ev_d1 = eigenvectors[1, s]
                ev_axial = eigenvectors[2, s]

                d0_coords = np.clip(points[:, 0], 0, ev_d0.shape[1] - 1)
                d1_coords = np.clip(points[:, 1], 0, ev_d0.shape[0] - 1)
                coords = np.array([d1_coords, d0_coords])

                v_d0 = map_coordinates(ev_d0, coords, order=1, mode='nearest')
                v_d1 = map_coordinates(ev_d1, coords, order=1, mode='nearest')
                v_axial = map_coordinates(ev_axial, coords, order=1, mode='nearest')

                # Compute displacement per unit step
                v_axial_safe = np.where(np.abs(v_axial) > 1e-6, v_axial, 1e-6)
                d_d0 = v_d0 / v_axial_safe
                d_d1 = v_d1 / v_axial_safe

                # Clamp displacement
                max_displacement = 1.0
                d_d0 = np.clip(d_d0, -max_displacement, max_displacement)
                d_d1 = np.clip(d_d1, -max_displacement, max_displacement)

                return np.stack([d_d0, d_d1], axis=1), v_d0, v_d1, v_axial

            if s_floor == s_ceil:
                return sample_at_slice(s_floor)
            else:
                # Linear interpolation between slices
                disp_floor, v_d0_f, v_d1_f, v_ax_f = sample_at_slice(s_floor)
                disp_ceil, v_d0_c, v_d1_c, v_ax_c = sample_at_slice(s_ceil)
                disp = (1 - s_frac) * disp_floor + s_frac * disp_ceil
                v_d0 = (1 - s_frac) * v_d0_f + s_frac * v_d0_c
                v_d1 = (1 - s_frac) * v_d1_f + s_frac * v_d1_c
                v_axial = (1 - s_frac) * v_ax_f + s_frac * v_ax_c
                return disp, v_d0, v_d1, v_axial

        for s in range(1, n_slices):
            if s % 50 == 0:
                active_count = np.sum(active)
                print(f"[INFO] RK4 Processing slice {s}/{n_slices} (active fibers: {active_count}, total: {n_fibers})")

            # RK4 integration: compute k1, k2, k3, k4
            h = 1.0  # Step size (1 slice)

            # k1 = f(s-1, y_n)
            k1, _, _, _ = get_displacement(current_points, s - 1)

            # k2 = f(s-0.5, y_n + h/2 * k1)
            points_k2 = current_points + 0.5 * h * k1
            k2, _, _, _ = get_displacement(points_k2, s - 0.5)

            # k3 = f(s-0.5, y_n + h/2 * k2)
            points_k3 = current_points + 0.5 * h * k2
            k3, _, _, _ = get_displacement(points_k3, s - 0.5)

            # k4 = f(s, y_n + h * k3)
            points_k4 = current_points + h * k3
            k4, v_d0, v_d1, v_axial = get_displacement(points_k4, s)

            # RK4 update: y_{n+1} = y_n + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            new_points = current_points + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Compute angles at final position
            angles = np.rad2deg(np.arctan2(np.sqrt(v_d0**2 + v_d1**2), np.abs(v_axial)))
            azimuths = np.rad2deg(np.arctan2(v_d1, v_d0))
            azimuths = np.mod(azimuths, 360)

            # Check for boundary crossing
            # Stop tracking when fiber center approaches boundary by fiber radius
            if stop_at_boundary:
                fiber_radius = self.fiber_diameter / 2.0 if self.fiber_diameter else boundary_margin
                effective_margin = max(fiber_radius, boundary_margin)

                out_of_bounds = (
                    (new_points[:, 0] < effective_margin) |
                    (new_points[:, 0] > dim0_max - effective_margin) |
                    (new_points[:, 1] < effective_margin) |
                    (new_points[:, 1] > dim1_max - effective_margin)
                )
                newly_stopped = active & out_of_bounds
                stopped_count += np.sum(newly_stopped)
                active = active & ~out_of_bounds

            # Resample fibers in empty regions
            if resample_interval > 0 and s % resample_interval == 0:
                new_fiber_points, new_fiber_count = self._resample_empty_regions(
                    new_points, active, dim0_max + 1, dim1_max + 1,
                    seed=resample_seed if resample_seed else s
                )
                if new_fiber_count > 0:
                    new_points = np.vstack([new_points, new_fiber_points])
                    active = np.concatenate([active, np.ones(new_fiber_count, dtype=bool)])
                    for i in range(new_fiber_count):
                        self.fiber_trajectories.append([(s, new_fiber_points[i].copy())])
                        self.fiber_angles.append([0.0])
                        self.fiber_azimuths.append([0.0])
                    angles = np.concatenate([angles, np.zeros(new_fiber_count)])
                    azimuths = np.concatenate([azimuths, np.zeros(new_fiber_count)])
                    resampled_count += new_fiber_count
                    n_fibers = len(new_points)

            # Apply relaxation
            if relax and np.sum(active) > 1:
                active_indices = np.where(active)[0]
                if len(active_indices) > 1:
                    active_points = new_points[active_indices]
                    relaxed_points = self._relax_points(active_points, self.fiber_diameter, relax_iterations)
                    new_points[active_indices] = relaxed_points

            # Store per-fiber trajectory data
            for i in range(n_fibers):
                if i < len(self.fiber_trajectories):
                    if active[i] or not stop_at_boundary:
                        if len(self.fiber_trajectories[i]) == 0 or self.fiber_trajectories[i][-1][0] != s:
                            self.fiber_trajectories[i].append((s, new_points[i].copy()))
                            self.fiber_angles[i].append(angles[i])
                            if i < len(self.fiber_azimuths):
                                self.fiber_azimuths[i].append(azimuths[i])

            self.trajectories.append((s, new_points.copy()))
            self.angles.append(angles)
            self.azimuths.append(azimuths)

            current_points = new_points

            if stop_at_boundary and np.sum(active) == 0:
                print(f"[INFO] All fibers have exited the domain at slice {s}")
                break

        self.active_fibers = active
        msg = f"[INFO] RK4 Propagation complete. {stopped_count} fibers stopped at boundary."
        if resampled_count > 0:
            msg += f" {resampled_count} new fibers resampled."
        print(msg)

    def propagate_with_detection(
        self,
        volume: np.ndarray,
        structure_tensor: np.ndarray,
        detection_interval: int = 5,
        max_matching_distance: float = None,
        min_diameter: float = 5.0,
        max_diameter: float = 20.0,
        min_peak_distance: int = 5,
        relax: bool = True,
        relax_iterations: int = 100,
        stop_at_boundary: bool = True,
        boundary_margin: float = 0.5,
        add_new_fibers: bool = False,
        new_fiber_interval: int = 10
    ) -> None:
        """
        Propagate fibers using RK4 with periodic image-based center detection and matching.

        This method combines RK4 trajectory prediction with image-based fiber center
        detection. Every `detection_interval` slices, it detects fiber centers from
        the actual CT image and matches them to the predicted positions using nearest
        neighbor search. This corrects drift and improves tracking accuracy.

        Args:
            volume: 3D volume array (z, y, x) of CT data.
            structure_tensor: 4D array with shape (6, z, y, x) containing
                             the symmetric structure tensor components.
            detection_interval: Number of slices between detection updates (1-20).
            max_matching_distance: Maximum distance for matching detected centers
                                  to predicted positions. Defaults to fiber_diameter.
            min_diameter: Minimum fiber diameter for detection.
            max_diameter: Maximum fiber diameter for detection.
            min_peak_distance: Minimum distance between peaks for detection.
            relax: Whether to apply relaxation to avoid fiber overlaps.
            relax_iterations: Number of iterations for relaxation.
            stop_at_boundary: If True, stop tracking fibers that exit the domain.
            boundary_margin: Margin from boundary to consider as "out of bounds".
            add_new_fibers: If True, add unmatched detected centers as new fibers.
            new_fiber_interval: Interval (slices) at which to check for new fibers.
        """
        if self.points is None:
            raise ValueError("Fibers not initialized. Call initialize_from_image() first.")

        # Validate detection interval
        detection_interval = max(1, min(20, detection_interval))

        # Set default max matching distance
        if max_matching_distance is None:
            max_matching_distance = self.fiber_diameter if self.fiber_diameter else 10.0

        # Determine number of slices and domain bounds
        # structure_tensor shape is (6, z, y, x)
        # Points from detect_fiber_centers are (x, y) format
        print(f"[DEBUG] structure_tensor type = {type(structure_tensor)}, shape = {structure_tensor.shape if hasattr(structure_tensor, 'shape') else 'N/A'}")
        print(f"[DEBUG] volume type = {type(volume)}, shape = {volume.shape if hasattr(volume, 'shape') else 'N/A'}")

        if self.propagation_axis == 0:  # X-axis propagation
            n_slices = structure_tensor.shape[3]  # x
            dim0_max = structure_tensor.shape[1] - 1  # z
            dim1_max = structure_tensor.shape[2] - 1  # y
            axis_name = "X"
        elif self.propagation_axis == 1:  # Y-axis propagation
            n_slices = structure_tensor.shape[2]  # y
            dim0_max = structure_tensor.shape[3] - 1  # x
            dim1_max = structure_tensor.shape[1] - 1  # z
            axis_name = "Y"
        else:  # Z-axis propagation (default)
            n_slices = structure_tensor.shape[1]  # z
            # Cross-section is XY plane
            # For Z-axis propagation: points[:, 0]=x, points[:, 1]=y
            # structure_tensor shape (6, z, y, x)
            dim0_max = structure_tensor.shape[3] - 1  # x dimension
            dim1_max = structure_tensor.shape[2] - 1  # y dimension
            axis_name = "Z"

        n_fibers = len(self.points)
        print(f"[INFO] Propagating {n_fibers} fibers along {axis_name}-axis through {n_slices} slices")
        print(f"[INFO] Domain bounds: x=[0, {dim0_max}], y=[0, {dim1_max}]")
        print(f"[INFO] Points range: x=[{self.points[:, 0].min():.1f}, {self.points[:, 0].max():.1f}], y=[{self.points[:, 1].min():.1f}, {self.points[:, 1].max():.1f}]")
        print(f"[INFO] Detection every {detection_interval} slices...")

        # Compute eigenvectors for all slices
        eigenvectors = _compute_eigenvectors(structure_tensor, self.reference_vector)

        current_points = self.points.copy()
        active = self.active_fibers.copy()
        stopped_count = 0
        detection_count = 0
        matched_count = 0
        new_fiber_count = 0

        # Validate new fiber interval
        new_fiber_interval = max(1, new_fiber_interval)

        def get_displacement(points, slice_idx):
            """Get displacement field at given points and slice index."""
            s_floor = int(np.floor(slice_idx))
            s_ceil = int(np.ceil(slice_idx))
            s_frac = slice_idx - s_floor
            s_floor = max(0, min(s_floor, n_slices - 1))
            s_ceil = max(0, min(s_ceil, n_slices - 1))

            def sample_at_slice(s):
                ev_d0 = eigenvectors[0, s]
                ev_d1 = eigenvectors[1, s]
                ev_axial = eigenvectors[2, s]

                d0_coords = np.clip(points[:, 0], 0, ev_d0.shape[1] - 1)
                d1_coords = np.clip(points[:, 1], 0, ev_d0.shape[0] - 1)
                coords = np.array([d1_coords, d0_coords])

                v_d0 = map_coordinates(ev_d0, coords, order=1, mode='nearest')
                v_d1 = map_coordinates(ev_d1, coords, order=1, mode='nearest')
                v_axial = map_coordinates(ev_axial, coords, order=1, mode='nearest')

                v_axial_safe = np.where(np.abs(v_axial) > 1e-6, v_axial, 1e-6)
                d_d0 = np.clip(v_d0 / v_axial_safe, -1.0, 1.0)
                d_d1 = np.clip(v_d1 / v_axial_safe, -1.0, 1.0)

                return np.stack([d_d0, d_d1], axis=1), v_d0, v_d1, v_axial

            if s_floor == s_ceil:
                return sample_at_slice(s_floor)
            else:
                disp_floor, v_d0_f, v_d1_f, v_ax_f = sample_at_slice(s_floor)
                disp_ceil, v_d0_c, v_d1_c, v_ax_c = sample_at_slice(s_ceil)
                disp = (1 - s_frac) * disp_floor + s_frac * disp_ceil
                v_d0 = (1 - s_frac) * v_d0_f + s_frac * v_d0_c
                v_d1 = (1 - s_frac) * v_d1_f + s_frac * v_d1_c
                v_axial = (1 - s_frac) * v_ax_f + s_frac * v_ax_c
                return disp, v_d0, v_d1, v_axial

        for s in range(1, n_slices):
            if s % 50 == 0:
                active_count = np.sum(active)
                print(f"[INFO] Processing slice {s}/{n_slices} (active: {active_count}, matched: {matched_count})")

            # RK4 integration
            h = 1.0
            k1, _, _, _ = get_displacement(current_points, s - 1)
            k2, _, _, _ = get_displacement(current_points + 0.5 * h * k1, s - 0.5)
            k3, _, _, _ = get_displacement(current_points + 0.5 * h * k2, s - 0.5)
            k4, v_d0, v_d1, v_axial = get_displacement(current_points + h * k3, s)

            predicted_points = current_points + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Clamp predicted points to valid range and handle NaN/Inf
            predicted_points = np.clip(predicted_points, 0, max(dim0_max, dim1_max))
            predicted_points = np.nan_to_num(predicted_points, nan=0.0, posinf=dim0_max, neginf=0.0)

            # Perform detection and matching at specified intervals
            if s % detection_interval == 0:
                detection_count += 1

                # Get the slice image
                if self.propagation_axis == 0:
                    slice_image = volume[:, :, s]
                elif self.propagation_axis == 1:
                    slice_image = volume[:, s, :]
                else:
                    slice_image = volume[s, :, :]

                # Detect fiber centers in this slice
                detected_centers, _ = detect_fiber_centers(
                    slice_image,
                    min_diameter=min_diameter,
                    max_diameter=max_diameter,
                    min_distance=min_peak_distance
                )

                if len(detected_centers) > 0:
                    # Build KD-tree of detected centers
                    detected_tree = KDTree(detected_centers)

                    # Track which detected centers are matched
                    matched_detected_indices = set()

                    # Match each active predicted point to nearest detected center
                    new_points = predicted_points.copy()
                    for i in range(len(predicted_points)):
                        if active[i]:
                            pt = predicted_points[i]
                            # Check for valid (finite) coordinates
                            if np.isfinite(pt[0]) and np.isfinite(pt[1]):
                                dist, idx = detected_tree.query(pt)
                                if dist < max_matching_distance:
                                    # Update to detected position
                                    new_points[i] = detected_centers[idx]
                                    matched_detected_indices.add(idx)
                                    matched_count += 1

                    predicted_points = new_points

                    # Add unmatched detected centers as new fibers
                    if add_new_fibers and s % new_fiber_interval == 0:
                        unmatched_indices = [i for i in range(len(detected_centers))
                                            if i not in matched_detected_indices]

                        if unmatched_indices:
                            # Filter unmatched centers that are too close to existing active fibers
                            active_positions = predicted_points[active]
                            if len(active_positions) > 0:
                                active_tree = KDTree(active_positions)

                            new_fibers_added = 0
                            for idx in unmatched_indices:
                                new_center = detected_centers[idx]

                                # Check if far enough from existing fibers
                                if len(active_positions) > 0:
                                    dist_to_existing, _ = active_tree.query(new_center)
                                    if dist_to_existing < self.fiber_diameter * 0.8:
                                        continue  # Too close to existing fiber

                                # Check boundary margin
                                fiber_radius = self.fiber_diameter / 2.0 if self.fiber_diameter else boundary_margin
                                effective_margin = max(fiber_radius, boundary_margin)
                                if (new_center[0] < effective_margin or
                                    new_center[0] > dim0_max - effective_margin or
                                    new_center[1] < effective_margin or
                                    new_center[1] > dim1_max - effective_margin):
                                    continue  # Too close to boundary

                                # Add new fiber
                                new_fiber_idx = len(current_points)
                                current_points = np.vstack([current_points, new_center])
                                predicted_points = np.vstack([predicted_points, new_center])
                                active = np.append(active, True)

                                # Initialize new fiber trajectory data
                                self.fiber_trajectories.append([(s, new_center.copy())])
                                self.fiber_angles.append([0.0])  # Will be updated
                                self.fiber_azimuths.append([0.0])  # Will be updated

                                new_fibers_added += 1
                                new_fiber_count += 1

                            if new_fibers_added > 0:
                                # Rebuild active_tree for relaxation
                                n_fibers = len(current_points)
                                # Extend angles/azimuths arrays for this slice
                                angles = np.zeros(n_fibers)
                                azimuths = np.zeros(n_fibers)

            # Get current number of fibers (may have increased due to new fibers)
            n_fibers_current = len(predicted_points)

            # Compute angles - need to handle case where new fibers were added
            if len(v_d0) < n_fibers_current:
                # New fibers were added, need to extend angle arrays
                # Recompute displacement for all current points
                disp_all, v_d0_all, v_d1_all, v_axial_all = get_displacement(predicted_points, s)
                angles = np.rad2deg(np.arctan2(np.sqrt(v_d0_all**2 + v_d1_all**2), np.abs(v_axial_all)))
                azimuths = np.mod(np.rad2deg(np.arctan2(v_d1_all, v_d0_all)), 360)
            else:
                angles = np.rad2deg(np.arctan2(np.sqrt(v_d0**2 + v_d1**2), np.abs(v_axial)))
                azimuths = np.mod(np.rad2deg(np.arctan2(v_d1, v_d0)), 360)

            # Check boundaries - stop tracking when fiber center approaches boundary by fiber radius
            if stop_at_boundary:
                # Use fiber radius as boundary margin (fiber center should stay at least radius away from edge)
                fiber_radius = self.fiber_diameter / 2.0 if self.fiber_diameter else boundary_margin
                effective_margin = max(fiber_radius, boundary_margin)

                out_of_bounds = (
                    (predicted_points[:, 0] < effective_margin) |
                    (predicted_points[:, 0] > dim0_max - effective_margin) |
                    (predicted_points[:, 1] < effective_margin) |
                    (predicted_points[:, 1] > dim1_max - effective_margin)
                )
                newly_stopped = active & out_of_bounds
                stopped_count += np.sum(newly_stopped)
                active = active & ~out_of_bounds

            # Apply relaxation
            if relax and np.sum(active) > 1:
                active_indices = np.where(active)[0]
                if len(active_indices) > 1:
                    active_pts = predicted_points[active_indices]
                    relaxed_pts = self._relax_points(active_pts, self.fiber_diameter, relax_iterations)
                    predicted_points[active_indices] = relaxed_pts

            # Store per-fiber trajectory data
            for i in range(n_fibers_current):
                if i < len(self.fiber_trajectories):
                    if active[i] or not stop_at_boundary:
                        if len(self.fiber_trajectories[i]) == 0 or self.fiber_trajectories[i][-1][0] != s:
                            self.fiber_trajectories[i].append((s, predicted_points[i].copy()))
                            if i < len(angles):
                                self.fiber_angles[i].append(angles[i])
                            if i < len(self.fiber_azimuths) and i < len(azimuths):
                                self.fiber_azimuths[i].append(azimuths[i])

            self.trajectories.append((s, predicted_points.copy()))
            self.angles.append(angles)
            self.azimuths.append(azimuths)

            current_points = predicted_points

            if stop_at_boundary and np.sum(active) == 0:
                print(f"[INFO] All fibers have exited the domain at slice {s}")
                break

        self.active_fibers = active
        self.points = current_points  # Update points to include new fibers
        print(f"[INFO] Propagation with detection complete.")
        print(f"[INFO] Detections: {detection_count}, Total matches: {matched_count}, New fibers: {new_fiber_count}, Stopped: {stopped_count}")

    def _resample_empty_regions(
        self,
        current_points: np.ndarray,
        active: np.ndarray,
        x_size: float,
        y_size: float,
        seed: int = 42
    ) -> tuple:
        """
        Find empty regions and resample new fibers there.

        Args:
            current_points: Current fiber positions (N, 2).
            active: Boolean mask of active fibers.
            x_size: Domain size in x direction.
            y_size: Domain size in y direction.
            seed: Random seed for sampling.

        Returns:
            Tuple of (new_points, count) where new_points is array of new fiber
            positions and count is the number of new fibers.
        """
        # Get only active fiber positions
        active_points = current_points[active]

        if len(active_points) == 0:
            # All fibers stopped - resample entire domain
            rng = np.random.default_rng(seed)
            # Estimate number of fibers based on original volume fraction
            target_fibers = int(x_size * y_size * self.fiber_volume_fraction / (np.pi / 4 * self.fiber_diameter ** 2))
            new_points = rng.random((target_fibers, 2)) * np.array([x_size, y_size])
            # Filter to maintain minimum distance
            new_points = self._filter_by_distance(new_points, self.fiber_diameter)
            return new_points, len(new_points)

        # Build KD-tree of active fibers
        tree = KDTree(active_points)

        # Generate candidate points on a grid
        grid_spacing = self.fiber_diameter * 1.5
        x_grid = np.arange(self.fiber_diameter, x_size - self.fiber_diameter, grid_spacing)
        y_grid = np.arange(self.fiber_diameter, y_size - self.fiber_diameter, grid_spacing)

        candidates = []
        for x in x_grid:
            for y in y_grid:
                # Check if there's no fiber nearby
                dist, _ = tree.query([x, y])
                if dist > self.fiber_diameter * 1.2:  # Empty region
                    candidates.append([x, y])

        if not candidates:
            return np.array([]).reshape(0, 2), 0

        candidates = np.array(candidates)

        # Randomly select subset of candidates
        rng = np.random.default_rng(seed)
        max_new = min(len(candidates), max(1, len(candidates) // 4))
        indices = rng.choice(len(candidates), size=max_new, replace=False)
        new_points = candidates[indices]

        # Filter to maintain minimum distance among new points
        new_points = self._filter_by_distance(new_points, self.fiber_diameter)

        return new_points, len(new_points)

    def _filter_by_distance(self, points: np.ndarray, min_distance: float) -> np.ndarray:
        """Filter points to maintain minimum distance between them."""
        if len(points) <= 1:
            return points

        filtered = [points[0]]
        for pt in points[1:]:
            # Check distance to all already selected points
            dists = np.linalg.norm(np.array(filtered) - pt, axis=1)
            if np.all(dists >= min_distance):
                filtered.append(pt)

        return np.array(filtered)

    def _relax_points(
        self,
        points: np.ndarray,
        min_distance: float,
        iterations: int = 100
    ) -> np.ndarray:
        """
        Relax points to avoid overlaps using iterative repulsion.

        Args:
            points: Array of 2D points with shape (N, 2).
            min_distance: Minimum distance between points.
            iterations: Maximum number of relaxation iterations.

        Returns:
            Relaxed points array.
        """
        points = np.array(points, dtype=float)
        for _ in range(iterations):
            tree = KDTree(points)
            pairs = tree.query_pairs(r=min_distance)
            if not pairs:
                break

            moved = np.zeros_like(points)
            for i, j in pairs:
                delta = points[j] - points[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-5:
                    continue
                overlap = min_distance - dist
                shift = (delta / dist) * (overlap / 2)
                moved[i] -= shift
                moved[j] += shift

            points += moved
            if np.all(np.linalg.norm(moved, axis=1) < 1e-3):
                break

        return points

    def smooth_trajectories(
        self,
        method: str = 'gaussian',
        window_size: int = 5,
        sigma: float = 1.0
    ) -> None:
        """
        Smooth fiber trajectories to reduce oscillation.

        This method applies smoothing to the (x, y) coordinates of each fiber
        trajectory. The z-coordinate (slice index) is preserved unchanged.

        Args:
            method: Smoothing method, either 'gaussian' or 'moving_average'.
            window_size: Window size for moving average (must be odd, >=3).
            sigma: Standard deviation for Gaussian smoothing.
        """
        if not self.fiber_trajectories:
            self._log("No trajectories to smooth")
            return

        if window_size < 3:
            window_size = 3
        if window_size % 2 == 0:
            window_size += 1  # Make odd

        smoothed_count = 0
        for fiber_idx, traj in enumerate(self.fiber_trajectories):
            if len(traj) < 3:
                # Too short to smooth
                continue

            # Extract coordinates
            slices = [pt[0] for pt in traj]
            x_coords = np.array([pt[1][0] for pt in traj])
            y_coords = np.array([pt[1][1] for pt in traj])

            # Apply smoothing
            if method == 'gaussian':
                x_smooth = gaussian_filter1d(x_coords, sigma=sigma, mode='nearest')
                y_smooth = gaussian_filter1d(y_coords, sigma=sigma, mode='nearest')
            else:  # moving_average
                # Pad for edge handling
                pad_size = window_size // 2
                x_padded = np.pad(x_coords, pad_size, mode='edge')
                y_padded = np.pad(y_coords, pad_size, mode='edge')

                # Compute moving average using convolution
                kernel = np.ones(window_size) / window_size
                x_smooth = np.convolve(x_padded, kernel, mode='valid')
                y_smooth = np.convolve(y_padded, kernel, mode='valid')

            # Rebuild trajectory with smoothed coordinates
            smoothed_traj = []
            for i, s in enumerate(slices):
                smoothed_traj.append((s, np.array([x_smooth[i], y_smooth[i]])))

            self.fiber_trajectories[fiber_idx] = smoothed_traj
            smoothed_count += 1

        # Also update the slice-based trajectories for backward compatibility
        if self.trajectories and smoothed_count > 0:
            self._rebuild_trajectories_from_fiber_data()

        self._log(f"Smoothed {smoothed_count} fiber trajectories (method={method})")

    def _rebuild_trajectories_from_fiber_data(self) -> None:
        """
        Rebuild slice-based trajectories from per-fiber trajectory data.

        This is used after smoothing to update the backward-compatible
        trajectories list. The trajectories list is indexed by slice order
        (0, 1, 2, ...), not by actual slice number.
        """
        if not self.fiber_trajectories:
            return

        # Collect all unique slice indices
        all_slices = set()
        for traj in self.fiber_trajectories:
            for s, _ in traj:
                all_slices.add(s)

        sorted_slices = sorted(all_slices)
        n_fibers = len(self.fiber_trajectories)

        # Build a lookup for each fiber's position at each slice
        fiber_positions = {}
        for fiber_idx, traj in enumerate(self.fiber_trajectories):
            for s, pos in traj:
                if s not in fiber_positions:
                    fiber_positions[s] = {}
                fiber_positions[s][fiber_idx] = pos

        # Build a lookup for angles at each slice (from existing angles data)
        # angles[slice_idx] contains angles for all fibers at that slice
        old_angles = self.angles if self.angles else []
        old_azimuths = self.azimuths if self.azimuths else []

        # Rebuild trajectories - index by slice order (0, 1, 2, ...)
        new_trajectories = []
        new_angles = []
        new_azimuths = []

        for slice_idx, s in enumerate(sorted_slices):
            if s not in fiber_positions:
                continue

            # Get positions for all fibers at this slice
            points = np.zeros((n_fibers, 2))
            for fiber_idx in range(n_fibers):
                if fiber_idx in fiber_positions[s]:
                    points[fiber_idx] = fiber_positions[s][fiber_idx]
                else:
                    # Fiber not present at this slice - use last known position
                    # (or zeros if never seen)
                    points[fiber_idx] = np.zeros(2)

            new_trajectories.append((slice_idx, points))

            # Copy angles from old data if available
            if slice_idx < len(old_angles):
                new_angles.append(old_angles[slice_idx])
            else:
                new_angles.append(np.zeros(n_fibers))

            if slice_idx < len(old_azimuths):
                new_azimuths.append(old_azimuths[slice_idx])
            else:
                new_azimuths.append(np.zeros(n_fibers))

        self.trajectories = new_trajectories
        self.angles = new_angles
        self.azimuths = new_azimuths


def _compute_eigenvectors(structure_tensor: np.ndarray, reference_vector: np.ndarray) -> np.ndarray:
    """
    Compute eigenvectors (minimum eigenvalue) from structure tensor with reference vector support.

    Args:
        structure_tensor: 4D array with shape (6, z, y, x).
        reference_vector: Reference direction [x, y, z] for fiber axis (normalized).

    Returns:
        Array with shape (4, z, y, x):
        - [0]: component perpendicular to propagation axis (first)
        - [1]: component perpendicular to propagation axis (second)
        - [2]: component along propagation axis
        - [3]: anisotropy (confidence) value
    """
    # Determine propagation axis from reference vector
    abs_ref = np.abs(reference_vector)
    propagation_axis = np.argmax(abs_ref)  # 0=x, 1=y, 2=z

    return _compute_eigenvectors_numba(structure_tensor, propagation_axis)


@numba.njit(parallel=True, cache=True)
def _compute_eigenvectors_numba(structure_tensor: np.ndarray, propagation_axis: int) -> np.ndarray:
    """
    Numba-accelerated eigenvector computation.

    Args:
        structure_tensor: 4D array with shape (6, z, y, x).
        propagation_axis: 0=X, 1=Y, 2=Z (which axis fibers are aligned with).

    Returns:
        Array with shape (4, z, y, x):
        - For Z-axis propagation: [x, y, z, anisotropy]
        - For Y-axis propagation: [x, z, y, anisotropy]
        - For X-axis propagation: [z, y, x, anisotropy]
    """
    symmetricComponents3d = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]

    z_size = structure_tensor.shape[1]
    y_size = structure_tensor.shape[2]
    x_size = structure_tensor.shape[3]

    # Store eigenvector components and anisotropy
    eigenvectors = np.zeros((4, z_size, y_size, x_size), dtype=np.float32)

    for z in numba.prange(z_size):
        for y in range(y_size):
            for x in range(x_size):
                # Build 3x3 symmetric tensor
                tensor = np.empty((3, 3), dtype=np.float32)
                for n, (i, j) in enumerate(symmetricComponents3d):
                    tensor[i, j] = structure_tensor[n, z, y, x]
                    if i != j:
                        tensor[j, i] = structure_tensor[n, z, y, x]

                # Eigenvalue decomposition
                w, v = np.linalg.eig(tensor)

                # Sort eigenvalues
                sorted_idx = np.argsort(w)
                min_idx = sorted_idx[0]
                max_idx = sorted_idx[2]

                eigenvec = v[:, min_idx]

                # Compute anisotropy: (lambda_max - lambda_min) / (lambda_max + lambda_min + eps)
                lambda_min = w[min_idx]
                lambda_max = w[max_idx]
                anisotropy = (lambda_max - lambda_min) / (lambda_max + lambda_min + 1e-10)

                # In skimage structure_tensor with (z,y,x) input, component order is (z,y,x)
                # eigenvec[0]=z, eigenvec[1]=y, eigenvec[2]=x
                ev_z = eigenvec[0]
                ev_y = eigenvec[1]
                ev_x = eigenvec[2]

                # Normalize sign based on propagation axis
                # Make the component along propagation axis positive
                if propagation_axis == 0:  # X-axis propagation
                    if ev_x < 0:
                        ev_x = -ev_x
                        ev_y = -ev_y
                        ev_z = -ev_z
                    # Output: [z, y, x, anisotropy] for YZ plane sampling
                    # dim0=z, dim1=y in cross-section
                    eigenvectors[0, z, y, x] = ev_z  # displacement in z (dim0)
                    eigenvectors[1, z, y, x] = ev_y  # displacement in y (dim1)
                    eigenvectors[2, z, y, x] = ev_x  # component along X (propagation axis)
                elif propagation_axis == 1:  # Y-axis propagation
                    if ev_y < 0:
                        ev_x = -ev_x
                        ev_y = -ev_y
                        ev_z = -ev_z
                    # Output: [x, z, y, anisotropy] for XZ plane sampling
                    # dim0=x, dim1=z in cross-section
                    eigenvectors[0, z, y, x] = ev_x  # displacement in x (dim0)
                    eigenvectors[1, z, y, x] = ev_z  # displacement in z (dim1)
                    eigenvectors[2, z, y, x] = ev_y  # component along Y (propagation axis)
                else:  # Z-axis propagation (default)
                    if ev_z < 0:
                        ev_x = -ev_x
                        ev_y = -ev_y
                        ev_z = -ev_z
                    # Output: [x, y, z, anisotropy] for XY plane sampling
                    # dim0=x, dim1=y in cross-section
                    eigenvectors[0, z, y, x] = ev_x  # displacement in x (dim0)
                    eigenvectors[1, z, y, x] = ev_y  # displacement in y (dim1)
                    eigenvectors[2, z, y, x] = ev_z  # component along Z (propagation axis)

                eigenvectors[3, z, y, x] = anisotropy

    return eigenvectors


def compute_orientation_volume(
    structure_tensor: np.ndarray,
    reference_vector: np.ndarray = None,
    status_callback=None
) -> tuple:
    """
    Compute tilt and azimuth volumes from structure tensor using the same method as fiber trajectories.

    This function uses the eigenvector decomposition method consistent with fiber trajectory
    propagation, ensuring the orientation colors match between trajectory and volume views.

    Args:
        structure_tensor: 4D array with shape (6, z, y, x) containing structure tensor components.
        reference_vector: Reference direction [x, y, z] for fiber axis. Defaults to [0, 0, 1] (Z-axis).
        status_callback: Optional callback function for status updates.

    Returns:
        Tuple of (tilt_volume, azimuth_volume):
        - tilt_volume: 3D array (z, y, x) of tilt angles in degrees (0-90, angle from propagation axis)
        - azimuth_volume: 3D array (z, y, x) of azimuth angles in degrees (0-360, direction in cross-section plane)
    """
    def _log(msg):
        if status_callback:
            status_callback(msg)
        else:
            print(msg)

    if reference_vector is None:
        reference_vector = np.array([0.0, 0.0, 1.0])

    _log("Computing orientation volume...")

    # Use the same eigenvector computation as fiber trajectory propagation
    eigenvectors = _compute_eigenvectors(structure_tensor, reference_vector)

    # For Z-axis propagation (default):
    # eigenvectors[0] = ev_x (x direction displacement)
    # eigenvectors[1] = ev_y (y direction displacement)
    # eigenvectors[2] = ev_z (z direction, along propagation axis)
    # eigenvectors[3] = anisotropy

    ev_d0 = eigenvectors[0]  # First cross-section direction (x for Z-axis)
    ev_d1 = eigenvectors[1]  # Second cross-section direction (y for Z-axis)
    ev_axial = eigenvectors[2]  # Along propagation axis (z for Z-axis)

    # Compute tilt angle (same as trajectory: angle from propagation axis)
    # angles = arctan2(sqrt(v_d0^2 + v_d1^2), abs(v_axial))
    tilt_volume = np.rad2deg(np.arctan2(
        np.sqrt(ev_d0**2 + ev_d1**2),
        np.abs(ev_axial)
    ))

    # Compute azimuth angle (same as trajectory: direction in cross-section plane)
    # azimuths = arctan2(v_d1, v_d0), converted to 0-360 range
    azimuth_volume = np.rad2deg(np.arctan2(ev_d1, ev_d0))
    azimuth_volume = np.mod(azimuth_volume, 360)  # Convert to 0-360 range

    _log(f"Orientation volume computed: tilt {tilt_volume.min():.1f}°-{tilt_volume.max():.1f}°")

    return tilt_volume, azimuth_volume


def detect_fiber_centers(
    image: np.ndarray,
    min_diameter: float = 5.0,
    max_diameter: float = 20.0,
    min_distance: int = 5,
    valid_mask: np.ndarray = None,
    return_labels: bool = False,
    threshold_percentile: float = None
) -> tuple:
    """
    Detect fiber centers from a 2D cross-section image using watershed segmentation.

    This function uses Otsu thresholding, distance transform, and watershed
    segmentation to detect individual fiber centers in CT images where fibers
    appear brighter than the surrounding matrix.

    Args:
        image: 2D grayscale image (fiber cross-section).
        min_diameter: Minimum fiber diameter in pixels to accept.
        max_diameter: Maximum fiber diameter in pixels to accept.
        min_distance: Minimum distance between detected peaks for watershed markers.
        valid_mask: Optional mask for valid image region. If None, uses image > 0.
        return_labels: If True, also return watershed labels array.
        threshold_percentile: If provided, use percentile-based thresholding instead of Otsu.
                              For example, 70.0 means threshold at 70th percentile.

    Returns:
        Tuple of (centers, diameters) or (centers, diameters, labels) if return_labels=True:
        - centers: (N, 2) array of fiber center coordinates (x, y)
        - diameters: (N,) array of estimated fiber diameters
        - labels: 2D array of watershed labels (only if return_labels=True)
    """
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = image > 0

    # Get valid pixel values for threshold computation
    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    # Thresholding: use percentile-based or Otsu
    if threshold_percentile is not None:
        threshold = np.percentile(valid_pixels, threshold_percentile)
    else:
        threshold = threshold_otsu(valid_pixels)

    # Binary mask of fibers
    binary = (image > threshold) & valid_mask

    # Distance transform
    distance = distance_transform_edt(binary)

    # Find local maxima (potential fiber centers)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
        exclude_border=False
    )

    if len(coords) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    # Create markers for watershed
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary)

    # Get region properties
    props = regionprops(labels)

    # Filter by diameter and collect results
    centers = []
    diameters = []
    valid_labels = []  # Track which labels are valid

    for prop in props:
        area = prop.area
        diameter = 2 * np.sqrt(area / np.pi)

        if min_diameter < diameter < max_diameter:
            y, x = prop.centroid
            centers.append([x, y])
            diameters.append(diameter)
            valid_labels.append(prop.label)

    if len(centers) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    # Create filtered labels array (only keep valid fibers)
    if return_labels:
        filtered_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(valid_labels, start=1):
            filtered_labels[labels == old_label] = new_label
        return np.array(centers), np.array(diameters), filtered_labels

    return np.array(centers), np.array(diameters)


def detect_fiber_centers_insegt(
    image: np.ndarray,
    min_diameter: float = 5.0,
    max_diameter: float = 20.0,
    valid_mask: np.ndarray = None,
    return_labels: bool = False,
    patch_size: int = 9,
    branching_factor: int = 5,
    number_layers: int = 5,
    training_patches: int = 30000,
    sigmas: list = None
) -> tuple:
    """
    Detect fiber centers using InSegt (Interactive Segmentation) with KM-tree.

    This function uses Gaussian derivative features and KM-tree dictionary learning
    for fiber segmentation, followed by centroid detection.

    Args:
        image: 2D grayscale image (fiber cross-section).
        min_diameter: Minimum fiber diameter in pixels to accept.
        max_diameter: Maximum fiber diameter in pixels to accept.
        valid_mask: Optional mask for valid image region. If None, uses image > 0.
        return_labels: If True, also return segmentation labels array.
        patch_size: Patch size for KM-tree (must be odd).
        branching_factor: Branching factor for KM-tree.
        number_layers: Number of layers in KM-tree.
        training_patches: Number of training patches for KM-tree.
        sigmas: List of sigma values for Gaussian features. Default [1, 2, 4].

    Returns:
        Tuple of (centers, diameters) or (centers, diameters, labels) if return_labels=True:
        - centers: (N, 2) array of fiber center coordinates (x, y)
        - diameters: (N,) array of estimated fiber diameters
        - labels: 2D array of segmentation labels (only if return_labels=True)
    """
    try:
        from acsc.insegt import KMTree, GaussFeatureExtractor, DictionaryPropagator
        import acsc.insegt.models.utils as insegt_utils
    except ImportError as e:
        raise ImportError(f"InSegt module not available: {e}")

    if sigmas is None:
        sigmas = [1, 2, 4]

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = image > 0

    # Normalize image to float [0, 1]
    if image.dtype != np.float64:
        img_float = insegt_utils.normalize_to_float(image.astype(np.uint8))
    else:
        img_float = image

    # Extract Gaussian features (GaussFeatureExtractor is callable)
    gauss = GaussFeatureExtractor(sigmas=sigmas)
    features = gauss(img_float, update_normalization=True, normalize=True)  # (channels, rows, cols)

    # Build KM-tree
    kmtree = KMTree(
        patch_size=patch_size,
        branching_factor=branching_factor,
        number_layers=number_layers,
        normalization=False
    )
    kmtree.build(features, training_patches)

    # Search for assignments
    assignment = kmtree.search(features)

    # Use Otsu thresholding on original image to create initial labels
    valid_pixels = image[valid_mask]
    if len(valid_pixels) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    threshold = threshold_otsu(valid_pixels)
    initial_labels = np.zeros_like(image, dtype=np.uint8)
    initial_labels[(image > threshold) & valid_mask] = 1  # Fiber
    initial_labels[(image <= threshold) & valid_mask] = 2  # Background

    # Convert labels to one-hot encoding
    labels_onehot = insegt_utils.labels_to_onehot(initial_labels)

    # Create dictionary propagator and propagate labels
    dict_prop = DictionaryPropagator(
        dictionary_size=kmtree.tree.shape[0],
        patch_size=patch_size
    )
    dict_prop.improb_to_dictprob(assignment, labels_onehot)
    probs = dict_prop.dictprob_to_improb(assignment)

    # Get segmentation from probabilities
    segmentation = insegt_utils.segment_probabilities(probs)

    # Iterate to refine segmentation (2 iterations)
    for _ in range(2):
        labels_onehot = insegt_utils.labels_to_onehot(segmentation)
        dict_prop.improb_to_dictprob(assignment, labels_onehot)
        probs = dict_prop.dictprob_to_improb(assignment)
        segmentation = insegt_utils.segment_probabilities(probs)

    # Create binary mask of fibers (class 1)
    binary = (segmentation == 1) & valid_mask

    # Distance transform
    distance = distance_transform_edt(binary)

    # Find local maxima (fiber centers)
    min_distance = int(min_diameter / 2)
    coords = peak_local_max(
        distance,
        min_distance=max(min_distance, 3),
        labels=binary,
        exclude_border=False
    )

    if len(coords) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    # Create markers for watershed
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (y, x) in enumerate(coords):
        markers[y, x] = i + 1

    # Watershed segmentation
    labels = watershed(-distance, markers, mask=binary)

    # Get region properties
    props = regionprops(labels)

    # Filter by diameter and collect results
    centers = []
    diameters = []
    valid_labels_list = []

    for prop in props:
        area = prop.area
        diameter = 2 * np.sqrt(area / np.pi)

        if min_diameter < diameter < max_diameter:
            y, x = prop.centroid
            centers.append([x, y])
            diameters.append(diameter)
            valid_labels_list.append(prop.label)

    if len(centers) == 0:
        if return_labels:
            return np.array([]).reshape(0, 2), np.array([]), np.zeros_like(image, dtype=np.int32)
        return np.array([]).reshape(0, 2), np.array([])

    # Create filtered labels array
    if return_labels:
        filtered_labels = np.zeros_like(labels)
        for new_label, old_label in enumerate(valid_labels_list, start=1):
            filtered_labels[labels == old_label] = new_label
        return np.array(centers), np.array(diameters), filtered_labels

    return np.array(centers), np.array(diameters)


def create_fiber_distribution(
    shape: tuple,
    fiber_diameter: float,
    fiber_volume_fraction: float,
    scale: float = 1.0,
    seed: int = 42,
    reference_vector: list = None,
    status_callback=None
) -> FiberTrajectory:
    """
    Convenience function to create a fiber distribution.

    Args:
        shape: Shape of the domain (z, y, x).
        fiber_diameter: Diameter of the fibers in pixels.
        fiber_volume_fraction: Target volume fraction of fibers (0-1).
        scale: Scale factor for minimum distance between fibers.
        seed: Random seed for reproducibility.
        reference_vector: Reference direction [x, y, z] for fiber axis.
                         Determines the propagation axis and sampling plane.
                         Default is [0, 0, 1] (Z-axis propagation).
        status_callback: Optional callback function for status updates.

    Returns:
        FiberTrajectory object with initialized fiber positions.
    """
    fibers = FiberTrajectory(status_callback=status_callback)
    fibers.initialize(shape, fiber_diameter, fiber_volume_fraction, scale, seed, reference_vector)
    return fibers


def visualize_fiber_points(
    fiber_trajectory: FiberTrajectory,
    point_size: float = 10.0,
    show_bounds: bool = True
) -> pv.Plotter:
    """
    Visualize fiber center points using PyVista.

    Args:
        fiber_trajectory: FiberTrajectory object with initialized points.
        point_size: Size of the points in the visualization.
        show_bounds: Whether to show domain boundary.

    Returns:
        PyVista Plotter object.
    """
    if fiber_trajectory.points is None:
        raise ValueError("No points to visualize. Call initialize() first.")

    points = fiber_trajectory.points
    bounds = fiber_trajectory.bounds
    diameter = fiber_trajectory.fiber_diameter

    # Create 3D points (x, y, z=0)
    points_3d = np.zeros((len(points), 3))
    points_3d[:, 0] = points[:, 0]  # x
    points_3d[:, 1] = points[:, 1]  # y
    points_3d[:, 2] = 0  # z = 0 for initial slice

    # Create point cloud
    point_cloud = pv.PolyData(points_3d)

    # Create plotter
    plotter = pv.Plotter()
    plotter.add_mesh(
        point_cloud,
        color='blue',
        point_size=point_size,
        render_points_as_spheres=True
    )

    # Add circles to show fiber diameter
    for pt in points_3d:
        circle = pv.Circle(radius=diameter / 2, resolution=32)
        circle.translate(pt, inplace=True)
        plotter.add_mesh(circle, color='lightblue', opacity=0.5)

    # Show domain boundary
    if show_bounds and bounds is not None:
        box = pv.Box(bounds=(0, bounds[2], 0, bounds[1], -1, 1))
        plotter.add_mesh(box, style='wireframe', color='gray', line_width=2)

    plotter.add_axes()
    plotter.view_xy()
    plotter.camera.zoom(1.2)

    return plotter


def visualize_fiber_trajectories(
    fiber_trajectory: FiberTrajectory,
    color_by_angle: bool = True,
    cmap: str = 'coolwarm',
    line_width: float = 2.0,
    show_bounds: bool = True,
    angle_range: tuple = (0, 20)
) -> pv.Plotter:
    """
    Visualize fiber trajectories as 3D lines colored by misalignment angle.

    Args:
        fiber_trajectory: FiberTrajectory object with computed trajectories.
        color_by_angle: If True, color lines by misalignment angle.
        cmap: Colormap name for angle coloring.
        line_width: Width of the trajectory lines.
        show_bounds: Whether to show domain boundary.
        angle_range: (min, max) angle range for colormap scaling.

    Returns:
        PyVista Plotter object.
    """
    trajectories = fiber_trajectory.trajectories
    angles = fiber_trajectory.angles
    bounds = fiber_trajectory.bounds

    if len(trajectories) < 2:
        raise ValueError("Not enough trajectory data. Call propagate() first.")

    n_fibers = len(trajectories[0][1])
    n_slices = len(trajectories)

    plotter = pv.Plotter()

    # Build lines for each fiber
    for fiber_idx in range(n_fibers):
        # Collect points along this fiber's trajectory
        points = []
        fiber_angles = []

        for slice_idx, (z, slice_points) in enumerate(trajectories):
            x = slice_points[fiber_idx, 0]
            y = slice_points[fiber_idx, 1]
            points.append([x, y, z])

            if slice_idx < len(angles):
                fiber_angles.append(angles[slice_idx][fiber_idx])

        points = np.array(points)

        # Create polyline
        n_points = len(points)
        lines = np.zeros(n_points + 1, dtype=np.int_)
        lines[0] = n_points
        lines[1:] = np.arange(n_points)

        poly = pv.PolyData(points, lines=lines)

        if color_by_angle and len(fiber_angles) > 0:
            # Assign angle values to points
            poly['angle'] = np.array(fiber_angles)
            plotter.add_mesh(
                poly,
                scalars='angle',
                cmap=cmap,
                clim=angle_range,
                line_width=line_width,
                render_lines_as_tubes=True
            )
        else:
            plotter.add_mesh(
                poly,
                color='blue',
                line_width=line_width,
                render_lines_as_tubes=True
            )

    # Add scalar bar once
    if color_by_angle:
        plotter.add_scalar_bar(
            title='Misalignment Angle (deg)',
            n_labels=5,
            position_x=0.85,
            width=0.1
        )

    # Show domain boundary
    if show_bounds and bounds is not None:
        box = pv.Box(bounds=(0, bounds[2], 0, bounds[1], 0, bounds[0]))
        plotter.add_mesh(box, style='wireframe', color='gray', line_width=1)

    plotter.add_axes()
    plotter.camera_position = 'iso'

    return plotter
