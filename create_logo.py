"""
Generate ACSC Logo - Minimal Kink Band Design
Represents fiber misalignment and kink bands in composite materials
"""
import numpy as np
import matplotlib.pyplot as plt


def create_acsc_logo(output_path='assets/acsc_logo.png', size_inches=4, dpi=300):
    """
    Minimal geometric version - clean and modern
    Abstract representation of fiber misalignment
    """
    fig, ax = plt.subplots(figsize=(size_inches, size_inches), facecolor='white')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Color scheme - minimal monochrome with accent
    primary = '#2c3e50'  # Dark blue-gray
    accent = '#3498db'   # Bright blue

    # Ultra-minimal: 3 lines total representing kink band
    # Left fiber
    ax.plot([-0.2, -0.2], [-0.5, 0.5], color=primary, linewidth=22,
           solid_capstyle='round', alpha=0.9, zorder=1)

    # Center fiber (accent color)
    ax.plot([0, 0], [-0.5, 0.5], color=accent, linewidth=22,
           solid_capstyle='round', alpha=1.0, zorder=2)

    # Right fiber
    ax.plot([0.2, 0.2], [-0.5, 0.5], color=primary, linewidth=22,
           solid_capstyle='round', alpha=0.9, zorder=1)

    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"Logo saved to: {output_path}")
    plt.close()

    return output_path


if __name__ == "__main__":
    print("Generating ACSC logo...")
    create_acsc_logo()
    print("\nLogo generated successfully!")
    print("- assets/acsc_logo.png: Minimal geometric version")
