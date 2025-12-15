---
sidebar_position: 5
title: Fiber Trajectory Generation
---

# Fiber Trajectory Generation

This document explains the fiber trajectory generation algorithm used to model individual fiber paths through a 3D volume based on orientation data.

## Overview

Fiber trajectory generation creates virtual fiber centerlines that follow the local orientation field computed from CT scan data. The process consists of two main stages:

1. **Initialization**: Generate initial fiber positions in a cross-section plane
2. **Propagation**: Trace fiber paths through the volume following the orientation field

## Initial Fiber Positioning

### Poisson Disk Sampling

Initial fiber center positions are generated using **Poisson disk sampling**, which ensures a minimum distance between any two points. This creates a more realistic fiber distribution compared to random sampling.

#### Algorithm

The Poisson disk sampling algorithm generates points such that:
- No two points are closer than a specified minimum distance $r$
- Points are approximately uniformly distributed

For fiber composites, the minimum distance $r$ is set based on the fiber diameter $d$ and a scale factor $s$:

$$
r = d \times s
$$

where typically $s \geq 1.0$ to prevent fiber overlap.

#### Number of Fibers

The target number of fibers $N$ is computed from the fiber volume fraction $V_f$ and fiber diameter $d$:

$$
N = \frac{A_{cross} \times V_f}{A_{fiber}} = \frac{A_{cross} \times V_f}{\pi d^2 / 4}
$$

where $A_{cross}$ is the cross-sectional area of the sampling plane.

### Variable-Spacing Poisson Sampling with Local Vf

When a local volume fraction map is available, the fiber spacing can be varied spatially to match the measured Vf distribution. This creates a more realistic representation of non-uniform fiber distributions.

#### Spacing from Volume Fraction

For square packing geometry, the relationship between fiber spacing $s$ and volume fraction $V_f$ is:

$$
V_f = \frac{\pi d^2 / 4}{s^2}
$$

Solving for spacing:

$$
s = d \sqrt{\frac{\pi}{4 V_f}} = \frac{d \cdot 0.886}{\sqrt{V_f}}
$$

where $d$ is the fiber diameter.

#### Algorithm (Fast Variable Density Poisson-Disc Sampling)

The implementation uses the fast variable density Poisson-disc sampling algorithm (Dwork et al., 2021). The key insight is to use a spatial grid with cell size based on the minimum spacing $r_{min}$, and store point indices in all cells that could be affected by each point.

1. **Calculate local spacing map**: Convert local Vf values to spacing requirements using the first slice of the Vf map:
   $$
   s(x, y) = \frac{d \cdot 0.886}{\sqrt{V_f(x, y)}}
   $$

2. **Grid initialization**: Create a spatial grid with cell size equal to $r_{min}$ (the minimum spacing across all locations). Each cell stores a list of point indices.

3. **Point registration**: When adding a new point $\mathbf{p}$ with local spacing $r(\mathbf{p})$:
   - Calculate how many cells the point's influence extends: $n_{cells} = \lceil r(\mathbf{p}) / r_{min} \rceil$
   - Register the point's index in all cells within this radius
   - This ensures any future candidate in those cells will check against this point

4. **Validity check**: To check if a candidate point is valid:
   - Look up only the cell containing the candidate
   - Check distance against all points registered in that cell
   - A candidate is valid if: $|\mathbf{p}_{new} - \mathbf{p}_{existing}| > \min(r_{new}, r_{existing})$

5. **Iterative expansion**: Standard Bridson-style expansion using an active list:
   - Generate candidates in an annulus $[r, 2r]$ around active points
   - Add valid candidates to the point set and active list
   - Remove points from active list when no valid candidates are found

6. **Gap filling**: Add random seeds to fill isolated regions not reached from the initial seed

#### Vf-Based Relaxation During Propagation

During fiber propagation, the relaxation step can use the local Vf map at each slice to adjust fiber spacing dynamically. This ensures that as fibers move through regions with varying Vf, their spacing adapts accordingly:

- In high-Vf regions: fibers are pushed closer together
- In low-Vf regions: fibers spread apart

This uses the same spacing formula applied at each propagation step.

#### Implementation

```python
# Initialize with local Vf consideration
ft.initialize(
    shape=volume.shape,
    fiber_diameter=7.0,
    fiber_volume_fraction=0.5,  # Fallback value
    vf_map=local_vf_map,        # 3D local Vf array
    vf_roi_bounds=(z_min, z_max, y_min, y_max, x_min, x_max)
)
```

This method produces fiber distributions where:
- High Vf regions have densely packed fibers
- Low Vf regions have sparse fiber placement
- Fiber spacing adapts dynamically during propagation

#### Validation Results

The following figure shows the variable-density Poisson disk sampling results for various synthetic Vf maps (400×400 pixels, fiber diameter = 7 pixels):

![Vf-based Poisson Sampling Results](/img/vf_poisson_sampling_results.png)

| Test Case | Description | Result |
|-----------|-------------|--------|
| Uniform Vf | Constant Vf = 0.5 | 1963 fibers, uniform spacing |
| Gradient | Left-to-right (0.2 → 0.8) | 1348 fibers, denser on right |
| Circular | Center=0.8, edge=0.2 | 1253 fibers, denser at center |
| Extreme Gradient | 0.15 → 0.85 | 1352 fibers, strong density variation |
| Two Regions | Left=0.2, Right=0.8 | Right/Left ratio = 3.63 (expected 4.0) |
| Checkerboard | Alternating 0.25/0.75 blocks | Local density matches Vf pattern |

The algorithm successfully adapts fiber density to match the local volume fraction distribution.

### Image-Based Initialization

Alternatively, fiber positions can be initialized from actual CT image data using fiber center detection (see [Segmentation](./segmentation.md)). This approach uses:

1. Otsu thresholding to segment fibers
2. Distance transform to find fiber centers
3. Watershed segmentation to separate touching fibers
4. Centroid computation for each detected fiber

## Trajectory Propagation

### Euler Method

The basic propagation method uses first-order Euler integration:

$$
\mathbf{p}_{s+1} = \mathbf{p}_s + \mathbf{v}(\mathbf{p}_s, s)
$$

where:
- $\mathbf{p}_s$ is the fiber position at slice $s$
- $\mathbf{v}(\mathbf{p}_s, s)$ is the local fiber direction from the orientation field

The direction vector is computed from the structure tensor eigenvector corresponding to the fiber axis direction.

### Runge-Kutta 4th Order (RK4)

For improved accuracy, the RK4 method provides 4th-order convergence:

$$
\mathbf{p}_{s+1} = \mathbf{p}_s + \frac{h}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)
$$

where:

$$
\begin{aligned}
\mathbf{k}_1 &= \mathbf{v}(\mathbf{p}_s, s) \\
\mathbf{k}_2 &= \mathbf{v}(\mathbf{p}_s + \frac{h}{2}\mathbf{k}_1, s + \frac{1}{2}) \\
\mathbf{k}_3 &= \mathbf{v}(\mathbf{p}_s + \frac{h}{2}\mathbf{k}_2, s + \frac{1}{2}) \\
\mathbf{k}_4 &= \mathbf{v}(\mathbf{p}_s + h\mathbf{k}_3, s + 1)
\end{aligned}
$$

RK4 is more accurate for curved fiber paths and reduces numerical drift.

### Detection-Based Propagation

This hybrid method combines RK4 prediction with periodic image-based correction:

1. Predict next position using RK4
2. Every $n$ slices, detect actual fiber centers from CT image
3. Match predicted positions to detected centers using nearest-neighbor search
4. Correct positions when a match is found within tolerance

This approach compensates for accumulated numerical errors and improves tracking accuracy.

## Direction Field Computation

The local fiber direction is extracted from the structure tensor eigenvector. For a propagation along the Z-axis:

$$
\frac{dx}{dz} = \frac{v_x}{v_z}, \quad \frac{dy}{dz} = \frac{v_y}{v_z}
$$

where $(v_x, v_y, v_z)$ is the eigenvector corresponding to the fiber direction.

### Misalignment Angle

The fiber misalignment angle (tilt from the propagation axis) is computed as:

$$
\theta = \arctan\left(\frac{\sqrt{v_x^2 + v_y^2}}{|v_z|}\right)
$$

### Azimuthal Angle

The in-plane direction (azimuth) is:

$$
\phi = \arctan2(v_y, v_x)
$$

## Fiber Relaxation

To prevent fiber overlap during propagation, a relaxation step can be applied. Fibers that come too close are pushed apart:

$$
\mathbf{p}_i' = \mathbf{p}_i + \sum_{j \neq i} \mathbf{f}_{ij}
$$

where the repulsion force is:

$$
\mathbf{f}_{ij} = \begin{cases}
\alpha \left(1 - \frac{|\mathbf{r}_{ij}|}{d}\right) \hat{\mathbf{r}}_{ij} & \text{if } |\mathbf{r}_{ij}| < d \\
0 & \text{otherwise}
\end{cases}
$$

Here $\mathbf{r}_{ij} = \mathbf{p}_j - \mathbf{p}_i$ and $\alpha$ is a relaxation strength parameter.

## Boundary Handling

Fibers that approach the domain boundary can be handled in two ways:

1. **Stop at boundary**: Fiber tracking terminates when the center reaches a margin from the edge
2. **Continue tracking**: Fiber continues but may partially exit the domain

The boundary margin is typically set to the fiber radius to ensure the entire fiber cross-section remains within the domain.

## Implementation

```python
from vmm.fiber_trajectory import FiberTrajectory

# Create trajectory generator
ft = FiberTrajectory()

# Initialize with Poisson disk sampling
ft.initialize(
    shape=volume.shape,
    fiber_diameter=7.0,
    fiber_volume_fraction=0.5,
    scale=1.2,  # 20% extra spacing
    seed=42
)

# Or initialize from image detection
ft.initialize_from_image(
    image=volume[0],  # First slice
    min_diameter=5.0,
    max_diameter=15.0,
    shape=volume.shape
)

# Propagate using RK4
ft.propagate_rk4(
    structure_tensor=tensor,
    relax=True,
    stop_at_boundary=True
)

# Access results
trajectories = ft.fiber_trajectories  # Per-fiber trajectory data
angles = ft.fiber_angles              # Misalignment angles
```

## Parameters

### Initialization Parameters

| Parameter | Description |
|-----------|-------------|
| `fiber_diameter` | Fiber diameter in pixels |
| `fiber_volume_fraction` | Target Vf (0-1) for Poisson sampling |
| `scale` | Multiplier for minimum fiber spacing |
| `seed` | Random seed for reproducibility |

### Propagation Parameters

| Parameter | Description |
|-----------|-------------|
| `relax` | Enable fiber relaxation to prevent overlap |
| `relax_iterations` | Number of relaxation iterations per slice |
| `stop_at_boundary` | Stop tracking at domain edges |
| `boundary_margin` | Distance from edge to stop (pixels) |
| `detection_interval` | Slices between image-based corrections |
