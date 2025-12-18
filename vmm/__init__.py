"""
VMM-FRC - Virtual Microstructure Modeling for Fiber Reinforced Polymer Composites

A toolkit for analyzing fiber-reinforced composite materials.
"""

__version__ = "0.0.9"

from vmm.segment import (
    estimate_local_vf,
    estimate_vf_distribution,
    estimate_vf_slice_by_slice,
    compute_vf_map_3d,
    threshold_otsu,
    threshold_percentile,
    apply_morphological_cleaning,
)
