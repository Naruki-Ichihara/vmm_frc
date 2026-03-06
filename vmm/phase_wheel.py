"""
Create a phase wheel visualization for azimuth angle definition.

Azimuth angle definition:
    azimuth = arctan2(v_d1, v_d0)

    where:
        v_d0 = eigenvector component in d0 direction (column, rightward)
        v_d1 = eigenvector component in d1 direction (row, downward)

    Range: -180° to 180°

Right-hand coordinate system (Z into screen):
    X-axis (d0): rightward ->
    Y-axis (d1): downward
    Z-axis (axial): into screen

This matches the GUI display with origin='upper' where:
    - Row index 0 is at the top
    - Row index increases downward (+d1 = down)
    - Column index increases rightward (+d0 = right)

HSV Colormap mapping:
    Hue = (azimuth + 180) / 360
    Saturation = |Tilt| / saturation_max (0 at center, 1 at edge)
    This maps -180 to 180 -> 0.0 to 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from typing import Optional


def create_phase_wheel(output_path: str = "azimuth_phase_wheel.svg",
                       saturation_max: float = 45.0,
                       saturation_min: float = 0.0,
                       show: bool = False,
                       dpi: int = 150,
                       figsize: tuple = (6, 6)) -> str:
    """
    Create a phase wheel showing azimuth angle definition and HSV colormap.

    Shows the right-hand coordinate system with Y-axis pointing down,
    matching GUI display with origin='upper'.

    The radial axis represents tilt angle (saturation), with:
    - Center (white): tilt = saturation_min (fibers aligned with axial)
    - Edge (full color): tilt = saturation_max (fibers tilted)

    Args:
        output_path: Path to save the image (supports .svg, .png, .pdf)
        saturation_max: Maximum tilt angle for full saturation (degrees)
        saturation_min: Minimum tilt angle for zero saturation (degrees)
        show: Whether to display the figure
        dpi: Resolution of saved image (for raster formats)
        figsize: Figure size in inches

    Returns:
        Path to the saved file
    """
    fig = plt.figure(figsize=figsize)

    # Single panel: Phase wheel with HSV colors and radial saturation gradient
    ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

    # Create color wheel using wedges with radial gradient (multiple rings)
    n_angles = 360
    n_rings = 50  # Number of radial rings for smooth gradient
    outer_radius = 1.0

    patches = []
    colors = []

    for ring in range(n_rings):
        # Radial position (0 = center, 1 = edge)
        r_inner = ring / n_rings * outer_radius
        r_outer = (ring + 1) / n_rings * outer_radius

        # Saturation increases from center (0) to edge (1)
        saturation = (ring + 0.5) / n_rings

        for i in range(n_angles):
            # Angle in degrees (starting from 0 at +X direction)
            angle_deg = i - 180  # Range: -180 to 179

            # Create wedge: theta1 and theta2 are in degrees, measured counter-clockwise from +X
            # Since Y points DOWN (not up), we negate the angle for display
            theta1 = -angle_deg - 0.5
            theta2 = -angle_deg + 0.5

            wedge = Wedge((0, 0), r_outer, theta1, theta2, width=r_outer - r_inner)
            patches.append(wedge)

            # HSV color: Hue = (azimuth + 180) / 360, Saturation varies with radius
            hue = (angle_deg + 180) / 360.0
            rgb = hsv_to_rgb([hue, saturation, 1.0])
            colors.append(rgb)

    # Create patch collection
    p = PatchCollection(patches, facecolors=colors, edgecolors='none')
    ax.add_collection(p)

    # Add angle labels around the wheel (skip 0 and 90 to avoid overlap with axis labels)
    label_angles_internal = [45, 135, 180, -135, -90, -45]
    label_radius = 1.15

    for angle_internal in label_angles_internal:
        # Display angle is negated (Y-down coordinate)
        angle_display = -angle_internal
        rad = np.deg2rad(angle_display)
        lx = label_radius * np.cos(rad)
        ly = label_radius * np.sin(rad)

        # Format label
        if angle_internal == 180:
            label = "+/-180"
        else:
            label = f"{angle_internal}"

        ax.annotate(label, (lx, ly), ha='center', va='center', fontsize=10, fontweight='bold')

    # Add axis arrows through center
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5, mutation_scale=12)
    axis_length = 1.05

    # d0 axis (horizontal): arrow pointing right (+d0)
    ax.annotate('', xy=(axis_length, 0), xytext=(-axis_length, 0), arrowprops=arrow_props)
    ax.text(axis_length + 0.08, 0, '+d0 (0)', ha='left', va='center', fontsize=10, fontweight='bold')

    # d1 axis (vertical): arrow pointing down (+d1)
    ax.annotate('', xy=(0, -axis_length), xytext=(0, axis_length), arrowprops=arrow_props)
    ax.text(0, -axis_length - 0.08, '+d1 (90)', ha='center', va='top', fontsize=10, fontweight='bold')

    # Center dot
    ax.plot(0, 0, 'ko', markersize=3)

    # Add tilt angle scale on the right side (radial axis = tilt)
    # Draw a vertical bar showing the tilt scale
    scale_x = 1.45
    scale_y_bottom = -0.8
    scale_y_top = 0.8
    scale_height = scale_y_top - scale_y_bottom

    # Draw scale bar background (gradient from white to colored)
    n_scale_segments = 50
    for i in range(n_scale_segments):
        y_bot = scale_y_bottom + i * scale_height / n_scale_segments
        y_top = scale_y_bottom + (i + 1) * scale_height / n_scale_segments
        sat = i / n_scale_segments
        # Use a reference hue (e.g., red = 0)
        rgb = hsv_to_rgb([0.0, sat, 1.0])
        ax.fill_between([scale_x - 0.03, scale_x + 0.03], y_bot, y_top,
                       color=rgb, edgecolor='none')

    # Draw scale bar border
    ax.plot([scale_x - 0.03, scale_x - 0.03], [scale_y_bottom, scale_y_top], 'k-', lw=0.5)
    ax.plot([scale_x + 0.03, scale_x + 0.03], [scale_y_bottom, scale_y_top], 'k-', lw=0.5)
    ax.plot([scale_x - 0.03, scale_x + 0.03], [scale_y_bottom, scale_y_bottom], 'k-', lw=0.5)
    ax.plot([scale_x - 0.03, scale_x + 0.03], [scale_y_top, scale_y_top], 'k-', lw=0.5)

    # Add tilt scale labels
    tilt_range = saturation_max - saturation_min
    n_ticks = 5
    for i in range(n_ticks):
        frac = i / (n_ticks - 1)
        y_pos = scale_y_bottom + frac * scale_height
        tilt_val = saturation_min + frac * tilt_range
        ax.plot([scale_x + 0.03, scale_x + 0.06], [y_pos, y_pos], 'k-', lw=0.5)
        ax.text(scale_x + 0.08, y_pos, f'{tilt_val:.0f}', ha='left', va='center', fontsize=8)

    # Scale title
    ax.text(scale_x, scale_y_top + 0.12, 'Tilt', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlim(-1.5, 1.7)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Azimuth Phase Wheel", fontsize=12, fontweight='bold', pad=5)

    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()

    return output_path


if __name__ == "__main__":
    # Test with default settings
    create_phase_wheel("test_phase_wheel.svg", saturation_max=45.0, show=True)
