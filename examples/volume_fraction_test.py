"""
Example: Volume fraction estimation from segmented images.

This example demonstrates global and local volume fraction estimation
using synthetic fiber data with known Vf distribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from vmm.segment import threshold_otsu, estimate_local_vf


def create_synthetic_binary_volume(shape, vf_pattern='uniform', target_vf=0.5,
                                    fiber_radius=3.5, seed=42):
    """
    Create a synthetic 3D binary volume with fibers.

    Parameters:
        shape: (depth, height, width) of the volume
        vf_pattern: 'uniform', 'gradient', or 'circular'
        target_vf: target volume fraction (for uniform) or mean Vf
        fiber_radius: fiber radius in pixels
        seed: random seed

    Returns:
        binary: 3D binary array
        true_vf_map: 3D array of true local Vf values
    """
    np.random.seed(seed)
    depth, height, width = shape
    binary = np.zeros(shape, dtype=np.uint8)

    # Create target Vf map based on pattern
    if vf_pattern == 'uniform':
        true_vf_map = np.full(shape, target_vf, dtype=np.float32)
    elif vf_pattern == 'gradient':
        # Left-to-right gradient: 0.3 to 0.7
        true_vf_map = np.zeros(shape, dtype=np.float32)
        for i in range(width):
            true_vf_map[:, :, i] = 0.3 + 0.4 * (i / (width - 1))
    elif vf_pattern == 'circular':
        # Center high (0.6), edge low (0.3)
        true_vf_map = np.zeros(shape, dtype=np.float32)
        cy, cx = height // 2, width // 2
        max_dist = np.sqrt(cy**2 + cx**2)
        for j in range(height):
            for i in range(width):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                t = dist / max_dist
                true_vf_map[:, j, i] = 0.6 - 0.3 * t
    else:
        raise ValueError(f"Unknown pattern: {vf_pattern}")

    # Generate fibers based on local Vf
    # Use spacing formula: s = d * 0.886 / sqrt(Vf)
    fiber_diameter = fiber_radius * 2

    # For each XY position, decide if there's a fiber based on local Vf
    y_coords, x_coords = np.ogrid[0:height, 0:width]

    # Generate fiber centers based on local probability
    fiber_area = np.pi * fiber_radius**2
    cell_size = fiber_diameter * 1.5

    for j in range(0, height, int(cell_size)):
        for i in range(0, width, int(cell_size)):
            # Get local target Vf
            local_vf = true_vf_map[0, min(j, height-1), min(i, width-1)]

            # Probability of having a fiber in this cell
            cell_area = cell_size ** 2
            expected_fiber_area = local_vf * cell_area
            prob = expected_fiber_area / fiber_area

            if np.random.random() < prob:
                # Add fiber with random offset within cell
                cx = i + np.random.uniform(0, cell_size)
                cy = j + np.random.uniform(0, cell_size)

                if 0 <= cx < width and 0 <= cy < height:
                    # Draw fiber in all slices
                    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                    fiber_mask = dist <= fiber_radius
                    binary[:, fiber_mask] = 1

    return binary, true_vf_map


def main():
    print("=" * 60)
    print("Volume Fraction Estimation Test")
    print("=" * 60)

    # Create synthetic volumes with different Vf patterns
    shape = (50, 300, 300)
    fiber_radius = 4.0

    patterns = ['uniform', 'gradient', 'circular']
    results = {}

    for pattern in patterns:
        print(f"\n--- Pattern: {pattern.upper()} ---")

        binary, true_vf_map = create_synthetic_binary_volume(
            shape, vf_pattern=pattern, fiber_radius=fiber_radius, seed=42
        )

        # Global Vf
        global_vf = binary.mean()
        true_global_vf = true_vf_map.mean()
        print(f"  Global Vf: {global_vf:.3f} (target: {true_global_vf:.3f})")

        # Local Vf estimation
        window_size = 50
        vf_map = estimate_local_vf(binary, window_size=window_size)

        # Statistics
        print(f"  Local Vf range: [{vf_map.min():.3f}, {vf_map.max():.3f}]")
        print(f"  Local Vf mean: {vf_map.mean():.3f}")
        print(f"  Local Vf std: {vf_map.std():.3f}")

        results[pattern] = {
            'binary': binary,
            'true_vf_map': true_vf_map,
            'estimated_vf_map': vf_map,
            'global_vf': global_vf
        }

    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, pattern in enumerate(patterns):
        data = results[pattern]
        mid_z = shape[0] // 2

        # Binary slice
        axes[row, 0].imshow(data['binary'][mid_z], cmap='gray')
        axes[row, 0].set_title(f'{pattern.capitalize()}: Binary\n(Global Vf={data["global_vf"]:.3f})')
        axes[row, 0].set_ylabel(f'{pattern.upper()}')

        # True Vf map
        im1 = axes[row, 1].imshow(data['true_vf_map'][mid_z], cmap='viridis',
                                   vmin=0.2, vmax=0.7)
        axes[row, 1].set_title('Target Vf Map')
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Estimated Vf map
        im2 = axes[row, 2].imshow(data['estimated_vf_map'][mid_z], cmap='viridis',
                                   vmin=0.2, vmax=0.7)
        axes[row, 2].set_title('Estimated Vf Map')
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Vf profile (horizontal line through center)
        mid_y = shape[1] // 2
        x_coords = np.arange(shape[2])
        true_profile = data['true_vf_map'][mid_z, mid_y, :]
        est_profile = data['estimated_vf_map'][mid_z, mid_y, :]

        axes[row, 3].plot(x_coords, true_profile, 'g-', linewidth=2, label='Target')
        axes[row, 3].plot(x_coords, est_profile, 'b-', linewidth=1, alpha=0.8, label='Estimated')
        axes[row, 3].set_xlabel('X Position')
        axes[row, 3].set_ylabel('Volume Fraction')
        axes[row, 3].set_title('Vf Profile (horizontal)')
        axes[row, 3].set_ylim(0.1, 0.8)
        axes[row, 3].legend()
        axes[row, 3].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = 'volume_fraction_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")

    # Additional: Slice-by-slice analysis for gradient pattern
    print("\n--- Slice-by-Slice Analysis (Gradient Pattern) ---")
    gradient_data = results['gradient']
    slice_vf = [gradient_data['binary'][z].mean() for z in range(shape[0])]

    fig2, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(shape[0]), slice_vf, 'b-', linewidth=2)
    ax.axhline(np.mean(slice_vf), color='red', linestyle='--',
               label=f'Mean = {np.mean(slice_vf):.3f}')
    ax.fill_between(range(shape[0]), slice_vf, alpha=0.3)
    ax.set_xlabel('Slice Index (Z)')
    ax.set_ylabel('Volume Fraction')
    ax.set_title('Vf Through Thickness (Gradient Pattern)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path2 = 'volume_fraction_slice_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Slice analysis saved to: {output_path2}")

    plt.show()


if __name__ == '__main__':
    main()
