---
sidebar_position: 6
title: vmm.fiber_trajectory
---

# vmm.fiber_trajectory

Fiber trajectory generation module for virtual microstructure modeling.

## Classes

### FiberTrajectory

```python
class FiberTrajectory:
    def __init__(self, status_callback=None)
```

Class to generate and manage fiber trajectories in a 3D domain.

Fiber initial positions are generated using Poisson disk sampling to ensure non-overlapping distribution. Trajectories are computed by propagating fibers along the direction specified by the reference vector.

**Attributes:**
- `points`: Initial fiber positions.
- `bounds`: Shape of the domain.
- `fiber_diameter`: Diameter of fibers in pixels.
- `fiber_volume_fraction`: Target volume fraction.
- `trajectories`: List of (axis_position, points) tuples.
- `angles`: List of misalignment angles at each step.
- `azimuths`: List of azimuthal angles at each step.
- `propagation_axis`: 0=Z, 1=Y, 2=X.
- `reference_vector`: Reference direction for fiber axis.

---

## Methods

### initialize

```python
def initialize(
    self,
    shape: tuple,
    fiber_diameter: float,
    fiber_volume_fraction: float,
    scale: float = 1.0,
    seed: int = 42,
    reference_vector: list = None,
    vf_map: np.ndarray = None,
    vf_roi_bounds: tuple = None
) -> np.ndarray
```

Initialize fiber positions using Poisson disk sampling.

**Args:**
- `shape` (tuple): Shape of the domain (z, y, x).
- `fiber_diameter` (float): Diameter of the fibers in pixels.
- `fiber_volume_fraction` (float): Target volume fraction of fibers (0-1).
- `scale` (float): Scale factor for minimum distance between fibers.
- `seed` (int): Random seed for reproducibility.
- `reference_vector` (list): Reference direction [x, y, z] for fiber axis.
- `vf_map` (np.ndarray): Optional 3D volume fraction map for weighted sampling.
- `vf_roi_bounds` (tuple): Bounds of Vf map.

**Returns:**
- `np.ndarray`: Initial fiber center positions as (N, 2) array.

---

### initialize_from_image

```python
def initialize_from_image(
    self,
    segmentation: np.ndarray,
    fiber_diameter: float,
    reference_vector: list = None
) -> np.ndarray
```

Initialize fiber positions from segmented image using centroid detection.

**Args:**
- `segmentation` (np.ndarray): Binary or labeled segmentation image.
- `fiber_diameter` (float): Expected fiber diameter in pixels.
- `reference_vector` (list): Reference direction for fiber axis.

**Returns:**
- `np.ndarray`: Detected fiber center positions.

---

### propagate

```python
def propagate(
    self,
    orientation: np.ndarray,
    azimuth: np.ndarray = None,
    steps: int = 100,
    step_size: float = 1.0
) -> None
```

Propagate fibers through the volume using orientation field.

**Args:**
- `orientation` (np.ndarray): 3D array of fiber tilt angles in degrees.
- `azimuth` (np.ndarray): 3D array of azimuthal angles in degrees.
- `steps` (int): Number of propagation steps.
- `step_size` (float): Size of each propagation step in pixels.

---

### to_pyvista

```python
def to_pyvista(self) -> pv.PolyData
```

Convert fiber trajectories to PyVista PolyData for visualization.

**Returns:**
- `pv.PolyData`: PyVista polydata with fiber trajectories as lines.

---

## Example

```python
from vmm.fiber_trajectory import FiberTrajectory
from vmm.analysis import compute_structure_tensor, compute_orientation

# Create trajectory generator
trajectory = FiberTrajectory()

# Initialize fibers
trajectory.initialize(
    shape=volume.shape,
    fiber_diameter=7.0,
    fiber_volume_fraction=0.5,
    reference_vector=[0, 0, 1]
)

# Compute orientation
tensor = compute_structure_tensor(volume, noise_scale=2)
orientation = compute_orientation(tensor, reference_vector=[0, 0, 1])

# Propagate through volume
trajectory.propagate(orientation, steps=100)

# Visualize
polydata = trajectory.to_pyvista()
polydata.plot()
```
