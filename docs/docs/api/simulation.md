---
sidebar_position: 5
title: vmm.simulation
---

# vmm.simulation

Virtual microstructure simulation and compression strength estimation.

## Classes

### MaterialParams

```python
@dataclass
class MaterialParams:
    longitudinal_modulus: float
    transverse_modulus: float
    poisson_ratio: float
    shear_modulus: float
    tau_y: float
    K: float
    n: float
```

Immutable container for composite material properties used in strength calculations.

**Attributes:**
- `longitudinal_modulus` (float): E1, elastic modulus parallel to fiber direction (MPa).
- `transverse_modulus` (float): E2, elastic modulus perpendicular to fibers (MPa).
- `poisson_ratio` (float): ν, ratio of transverse to longitudinal strain.
- `shear_modulus` (float): G, resistance to shear deformation (MPa).
- `tau_y` (float): Yield stress in shear for plasticity model (MPa).
- `K` (float): Hardening coefficient for power-law plasticity.
- `n` (float): Hardening exponent for power-law plasticity.

**Example:**

```python
from vmm.simulation import MaterialParams

material = MaterialParams(
    longitudinal_modulus=140000,  # MPa
    transverse_modulus=10000,     # MPa
    poisson_ratio=0.3,
    shear_modulus=5000,           # MPa
    tau_y=50,                     # MPa
    K=0.01,
    n=5
)
```

---

## Functions

### estimate_compression_strength_from_profile

```python
def estimate_compression_strength_from_profile(
    orientation_profile: np.ndarray,
    material_params: MaterialParams,
    maximum_shear_stress: float = 100.0,
    shear_stress_step_size: float = 0.1,
    maximum_axial_strain: float = 0.02,
    maximum_fiber_misalignment: float = 20,
    fiber_misalignment_step_size: float = 0.1,
    kink_width: float = None,
    gauge_length: float = None
) -> Tuple[float, float, np.ndarray, np.ndarray]
```

Calculate compression strength from measured fiber orientation distribution.

Implements a micromechanical model that accounts for fiber misalignment effects on composite compression strength.

**Args:**
- `orientation_profile` (np.ndarray): 3D array of fiber misalignment angles in degrees.
- `material_params` (MaterialParams): Material properties for the composite.
- `maximum_shear_stress` (float): Upper bound for shear stress integration (MPa).
- `shear_stress_step_size` (float): Resolution for shear stress discretization (MPa).
- `maximum_axial_strain` (float): Maximum compressive strain for analysis.
- `maximum_fiber_misalignment` (float): Upper bound for misalignment angle range (degrees).
- `fiber_misalignment_step_size` (float): Angular resolution (degrees).

**Returns:**
- `compression_strength` (float): Peak compressive stress (MPa).
- `ultimate_strain` (float): Strain at peak stress.
- `stress_curve` (np.ndarray): Complete stress-strain curve array.
- `strain_array` (np.ndarray): Corresponding strain values.

**Example:**

```python
from vmm.simulation import MaterialParams, estimate_compression_strength_from_profile

material = MaterialParams(
    longitudinal_modulus=140000,
    transverse_modulus=10000,
    poisson_ratio=0.3,
    shear_modulus=5000,
    tau_y=50,
    K=0.01,
    n=5
)

strength, strain, stress_curve, strain_array = estimate_compression_strength_from_profile(
    orientation_profile,
    material
)
print(f"Compression strength: {strength:.1f} MPa")
```
