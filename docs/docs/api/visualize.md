---
sidebar_position: 7
title: vmm.visualize
---

# vmm.visualize

3D visualization utilities using PyVista.

:::note
This module provides visualization utilities integrated with the GUI application.
For most use cases, the GUI provides interactive visualization features.
:::

## Usage

The visualization features are primarily accessed through the VMM-FRC GUI application:

```bash
vmm-frc-gui
```

For programmatic access, use PyVista directly with exported data:

```python
import pyvista as pv
from vmm.fiber_trajectory import FiberTrajectory

# Generate trajectories
trajectory = FiberTrajectory()
trajectory.initialize(shape=(100, 200, 200), fiber_diameter=7.0, fiber_volume_fraction=0.5)

# Convert to PyVista and visualize
polydata = trajectory.to_pyvista()

plotter = pv.Plotter()
plotter.add_mesh(polydata, scalars='TiltAngle', cmap='viridis')
plotter.show()
```

## Export to VTP

Fiber trajectories can be exported to VTP format for visualization in ParaView:

```python
# Using the GUI: File > Export > VTK

# Or programmatically:
polydata.save('fiber_trajectories.vtp')
```

The exported VTP file includes:
- **TiltAngle**: Fiber tilt angle from reference axis (degrees)
- **AzimuthAngle**: Fiber azimuthal angle (degrees)
- **FiberID**: Unique fiber identifier
