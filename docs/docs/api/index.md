---
sidebar_position: 1
title: API Reference
slug: /
---

# VMM-FRC API Reference

Complete API documentation for VMM-FRC (Virtual Microstructure Modeling for Fiber Reinforced Polymer Composites).

## Modules

| Module | Description |
|--------|-------------|
| [vmm.io](/api/io) | Image and volume I/O utilities |
| [vmm.analysis](/api/analysis) | Structure tensor and orientation analysis |
| [vmm.segment](/api/segment) | Segmentation and volume fraction estimation |
| [vmm.simulation](/api/simulation) | Virtual microstructure simulation |
| [vmm.fiber_trajectory](/api/fiber_trajectory) | Fiber trajectory generation |
| [vmm.visualize](/api/visualize) | 3D visualization utilities |

## Installation

```bash
pip install vmm-frc
```

## Quick Start

```python
import vmm
from vmm.io import import_image_sequence
from vmm.analysis import compute_structure_tensor, compute_orientation
from vmm.segment import estimate_local_vf, threshold_otsu

# Load CT volume
volume = import_image_sequence("path/to/images", 100, 4, "tif")

# Compute fiber orientation
tensor = compute_structure_tensor(volume, noise_scale=2)
orientation = compute_orientation(tensor, reference_vector=[0, 0, 1])

# Estimate volume fraction
binary = volume > threshold_otsu(volume)
vf_map = estimate_local_vf(binary, window_size=50)
```
