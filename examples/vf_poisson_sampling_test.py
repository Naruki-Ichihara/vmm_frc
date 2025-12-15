"""
Example: Variable-density Poisson disk sampling based on local Vf map.

This example demonstrates the Dwork et al. (2021) fast variable density
Poisson-disc sampling algorithm using a synthetic Vf map.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from vmm.fiber_trajectory import FiberTrajectory


def create_gradient_vf_map(shape, vf_low=0.3, vf_high=0.7):
    """Create a left-to-right gradient Vf map."""
    z, y, x = shape
    vf_map = np.zeros(shape, dtype=np.float32)
    for i in range(x):
        vf_map[:, :, i] = vf_low + (vf_high - vf_low) * (i / (x - 1))
    return vf_map


def create_circular_vf_map(shape, vf_center=0.7, vf_edge=0.3):
    """Create a circular Vf map with high Vf at center."""
    z, y, x = shape
    cy, cx = y // 2, x // 2
    max_dist = np.sqrt(cy**2 + cx**2)

    vf_map = np.zeros(shape, dtype=np.float32)
    for j in range(y):
        for i in range(x):
            dist = np.sqrt((j - cy)**2 + (i - cx)**2)
            t = dist / max_dist
            vf_map[:, j, i] = vf_center + (vf_edge - vf_center) * t
    return vf_map


def visualize_sampling(points, vf_slice, fiber_diameter, title, ax):
    """Visualize sampling result with fiber circles."""
    # Show Vf map as background
    im = ax.imshow(vf_slice, cmap='viridis', origin='lower',
                   extent=[0, vf_slice.shape[1], 0, vf_slice.shape[0]],
                   vmin=0.1, vmax=0.9)

    # Draw fiber circles
    radius = fiber_diameter / 2
    for pt in points:
        circle = Circle((pt[0], pt[1]), radius, fill=False,
                        edgecolor='red', linewidth=0.5, alpha=0.8)
        ax.add_patch(circle)

    ax.scatter(points[:, 0], points[:, 1], c='red', s=2, zorder=5)
    ax.set_title(f"{title}\n({len(points)} fibers)")
    ax.set_aspect('equal')
    ax.set_xlim(0, vf_slice.shape[1])
    ax.set_ylim(0, vf_slice.shape[0])

    return im


def main():
    # Parameters - larger area for more fibers
    shape = (10, 400, 400)  # z, y, x
    fiber_diameter = 7.0
    global_vf = 0.5

    print("=" * 60)
    print("Variable-Density Poisson Disk Sampling Test")
    print("=" * 60)
    print(f"Volume shape: {shape}")
    print(f"Fiber diameter: {fiber_diameter} pixels")
    print()

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Test 1: Uniform Vf (no vf_map)
    print("Test 1: Uniform Vf (standard Poisson sampling)")
    ft1 = FiberTrajectory()
    ft1.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42
    )
    print(f"  Generated {len(ft1.points)} fibers")

    # Visualize with uniform background
    uniform_vf = np.full(shape[1:], global_vf)
    visualize_sampling(ft1.points, uniform_vf, fiber_diameter,
                       "Uniform Vf = 0.5", axes[0, 0])

    # Test 2: Gradient Vf map (left=0.2, right=0.8) - stronger contrast
    print("\nTest 2: Gradient Vf map (left=0.2, right=0.8)")
    vf_gradient = create_gradient_vf_map(shape, vf_low=0.2, vf_high=0.8)

    ft2 = FiberTrajectory()
    ft2.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42,
        vf_map=vf_gradient,
        vf_roi_bounds=(0, shape[0], 0, shape[1], 0, shape[2])
    )
    print(f"  Generated {len(ft2.points)} fibers")

    im = visualize_sampling(ft2.points, vf_gradient[0], fiber_diameter,
                            "Gradient Vf (0.2 → 0.8)", axes[0, 1])

    # Test 3: Circular Vf map (center=0.8, edge=0.2) - stronger contrast
    print("\nTest 3: Circular Vf map (center=0.8, edge=0.2)")
    vf_circular = create_circular_vf_map(shape, vf_center=0.8, vf_edge=0.2)

    ft3 = FiberTrajectory()
    ft3.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42,
        vf_map=vf_circular,
        vf_roi_bounds=(0, shape[0], 0, shape[1], 0, shape[2])
    )
    print(f"  Generated {len(ft3.points)} fibers")

    visualize_sampling(ft3.points, vf_circular[0], fiber_diameter,
                       "Circular Vf (center=0.8, edge=0.2)", axes[0, 2])

    # Test 4: Very extreme gradient (0.15 to 0.85)
    print("\nTest 4: Very extreme gradient Vf map (0.15 → 0.85)")
    vf_extreme = create_gradient_vf_map(shape, vf_low=0.15, vf_high=0.85)

    ft4 = FiberTrajectory()
    ft4.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42,
        vf_map=vf_extreme,
        vf_roi_bounds=(0, shape[0], 0, shape[1], 0, shape[2])
    )
    print(f"  Generated {len(ft4.points)} fibers")

    visualize_sampling(ft4.points, vf_extreme[0], fiber_diameter,
                       "Extreme Gradient (0.15 → 0.85)", axes[1, 0])

    # Test 5: Two-region Vf map with stronger contrast
    print("\nTest 5: Two-region Vf map (left=0.2, right=0.8)")
    vf_two_region = np.zeros(shape, dtype=np.float32)
    vf_two_region[:, :, :shape[2]//2] = 0.2
    vf_two_region[:, :, shape[2]//2:] = 0.8

    ft5 = FiberTrajectory()
    ft5.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42,
        vf_map=vf_two_region,
        vf_roi_bounds=(0, shape[0], 0, shape[1], 0, shape[2])
    )
    print(f"  Generated {len(ft5.points)} fibers")

    visualize_sampling(ft5.points, vf_two_region[0], fiber_diameter,
                       "Two Regions (0.2 | 0.8)", axes[1, 1])

    # Analysis: Count fibers in each region for Test 5
    left_count = np.sum(ft5.points[:, 0] < shape[2] // 2)
    right_count = np.sum(ft5.points[:, 0] >= shape[2] // 2)
    print(f"  Left region (Vf=0.2): {left_count} fibers")
    print(f"  Right region (Vf=0.8): {right_count} fibers")
    print(f"  Ratio (right/left): {right_count/left_count:.2f} (expected ~{0.8/0.2:.2f})")

    # Test 6: Checkerboard pattern with stronger contrast
    print("\nTest 6: Checkerboard Vf pattern (0.25/0.75)")
    vf_checker = np.zeros(shape, dtype=np.float32)
    block_size = 100
    for j in range(0, shape[1], block_size):
        for i in range(0, shape[2], block_size):
            checker = ((j // block_size) + (i // block_size)) % 2
            vf_val = 0.75 if checker == 0 else 0.25
            vf_checker[:, j:j+block_size, i:i+block_size] = vf_val

    ft6 = FiberTrajectory()
    ft6.initialize(
        shape=shape,
        fiber_diameter=fiber_diameter,
        fiber_volume_fraction=global_vf,
        scale=1.0,
        seed=42,
        vf_map=vf_checker,
        vf_roi_bounds=(0, shape[0], 0, shape[1], 0, shape[2])
    )
    print(f"  Generated {len(ft6.points)} fibers")

    visualize_sampling(ft6.points, vf_checker[0], fiber_diameter,
                       "Checkerboard (0.25/0.75)", axes[1, 2])

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Volume Fraction (Vf)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save figure
    output_path = 'vf_poisson_sampling_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")

    plt.show()


if __name__ == '__main__':
    main()
