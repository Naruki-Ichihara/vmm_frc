import numpy as np
from scipy.stats.qmc import PoissonDisk
from scipy.ndimage import map_coordinates
from scipy.spatial import KDTree
import pyvista as pv

class Fibers:
    """
    Class to generate and manage fibers in a 3D domain.
    The initial positions of fibers are generated using a Poisson disk sampling method to ensure that they do not overlap.
    The fibers can be moved in the x and y directions based on provided direction vectors, and they can be relaxed to avoid overlaps.
    """
    def __init__(self):
        self.points = None
        self.bounds = None
        self.fiber_diameter = None
        self.fiber_volume_fraction = None
        self.fiber = []  # [z position, points]
        self.trajectory = []

    def initialize(self, shape: tuple, fiber_diameter: float, fiber_volume_fraction: float, scale=1.0, seed=42) -> None:
        """
        Initialize the Fibers class with a given shape, fiber diameter, and volume fraction.
        The centers of the fibers are generated using a Poisson disk sampling method.

        Args:
            shape (tuple): Shape of the domain (z, y, x).
            fiber_diameter (float): Diameter of the fibers.
            fiber_volume_fraction (float): Volume fraction of the fibers.
            scale (float): Scale factor for the fiber diameter.
            seed (int): Random seed for reproducibility.
        """
        if not shape[1] == shape[2]:
            raise ValueError("[ERROR] Shape must be square")
        self.bounds = shape
        self.fiber_diameter = fiber_diameter
        self.fiber_volume_fraction = fiber_volume_fraction

        total_area = shape[1] * shape[2]
        fiber_area = np.pi / 4 * fiber_diameter**2
        num_fibers = int(total_area * fiber_volume_fraction / fiber_area)

        max_dim = shape[1]
        normalized_radius = fiber_diameter*scale / max_dim
        print("[INFO] Sampling in progress...")
        sampler = PoissonDisk(d=2, radius=normalized_radius, seed=seed)
        points = sampler.random(num_fibers) * max_dim
        self.points = points
        self.fiber.append([0, self.points])
        self.trajectory.append(self.points.copy())
        print("[INFO] Sampling completed.")

    def update_fiber(self, position: float, points: np.ndarray) -> None:
        """
        Update the fiber data with a new position and points.
        This is used to add new layers of fibers at different z positions.
    
        Args:
            position (float): The z position of the new layer.
            points (np.ndarray): The points of the new layer.

        """
        self.fiber.append([position, points])
        self.trajectory.append(points.copy())

    def move_points(self, directions_x: np.ndarray, directions_y: np.ndarray, update=True, relax=True) -> np.ndarray:
        """
        Move the points in the x and y directions based on the provided direction arrays.
        The points are then relaxed to avoid overlaps.

        Args:
            directions_x (np.ndarray): Array of x direction vectors.
            directions_y (np.ndarray): Array of y direction vectors.
            update (bool): Whether to update the points in the object.
            relax (bool): Whether to relax the points after moving.

        Returns:
            np.ndarray: The new positions of the points after moving and relaxing.
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        coords = np.array([y, x])
        dir_x = map_coordinates(directions_x, coords, order=2, mode='nearest')
        dir_y = map_coordinates(directions_y, coords, order=2, mode='nearest')
        directions = np.stack([dir_x, dir_y], axis=1)
        new_points = self.points + directions
        if relax:
            relaxed_points = self._relax_points(new_points, self.fiber_diameter)
        else:
            relaxed_points = new_points
        if update:
            self.points = relaxed_points
        return relaxed_points
    
    def _relax_points(self, points, d, iterations=100):
        points = np.array(points, dtype=float)
        for _ in range(iterations):
            tree = KDTree(points)
            pairs = tree.query_pairs(r=d)
            moved = np.zeros_like(points)
            for i, j in pairs:
                delta = points[j] - points[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-5:
                    continue
                overlap = d - dist
                shift = (delta / dist) * (overlap / 2)
                moved[i] -= shift
                moved[j] += shift
            points += moved
            if np.all(np.linalg.norm(moved, axis=1) < 1e-3):
                break
        return points
    
def generate_fiber_stl(fibers: Fibers) -> None:
    """
    Generate an STL file for the fibers with y-axis mirrored 
    to convert from right-handed to left-handed coordinate system.

    Args:
        fibers (Fibers): The Fibers object containing fiber data.
    """
    if fibers.points is None:
        raise ValueError("No points to generate STL")

    diameter = fibers.fiber_diameter
    n_fibers = fibers.fiber[0][1].shape[0]
    radius = diameter / 2
    tubes = []

    print("[INFO] Generating STL...")

    # Collect all y values to compute center_y for mirroring
    all_y_values = []
    for _, points in fibers.fiber:
        all_y_values.extend([pt[1] for pt in points])
    center_y = (max(all_y_values) + min(all_y_values)) / 2

    for i in range(n_fibers):
        fiber_path = []
        for z_index, (z, points) in enumerate(fibers.fiber):
            x, y = points[i]
            mirrored_y = 2 * center_y - y  # Reflect across center_y
            fiber_path.append([x, mirrored_y, z])
        fiber_path = np.array(fiber_path)

        poly = pv.Spline(fiber_path, n_points=len(fiber_path))
        tube = poly.tube(radius=radius, n_sides=10, capping=True)
        tubes.append(tube)

    full_mesh = tubes[0]
    for tube in tubes[1:]:
        full_mesh += tube

    print("[INFO] STL generation completed.")
    return full_mesh