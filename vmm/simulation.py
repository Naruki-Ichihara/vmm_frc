import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from scipy.optimize import brentq
from scipy.signal import argrelmax, argrelmin, find_peaks
import scipy.stats as stats
from dataclasses import dataclass
from typing import Callable, Tuple
from vmm.logger import get_logger

logger = get_logger()

@dataclass
class MaterialParams:
    """
    Immutable container for composite material properties used in strength calculations.

    Defines the mechanical properties needed for fiber-reinforced composite analysis,
    including elastic moduli, plasticity parameters, and geometric factors.

    Attributes:
        longitudinal_modulus: E1, elastic modulus parallel to fiber direction (MPa).
        transverse_modulus: E2, elastic modulus perpendicular to fibers (MPa).
        poisson_ratio: ν, ratio of transverse to longitudinal strain under axial loading.
        shear_modulus: G, resistance to shear deformation (MPa).
        tau_y: Yield stress in shear for plasticity model (MPa).
        K: Hardening coefficient for power-law plasticity.
        n: Hardening exponent for power-law plasticity (dimensionless).
        Vf: (Optional) 3D fiber volume fraction (dimensionless, 0-1).
        fiber_diameter: (Optional) Fiber diameter (same unit as kink width).

    Note:
        Power-law plasticity model: γ = τ/G + K(τ/τ_y)^n where γ is shear strain.
        Vf and fiber_diameter are required for kink band failure analysis.
    """
    longitudinal_modulus: float
    transverse_modulus: float
    poisson_ratio: float
    shear_modulus: float
    tau_y: float
    K: float
    n: float
    Vf: float = None
    fiber_diameter: float = None

def estimate_compression_strength_from_profile(orientation_profile: np.ndarray,
                                               material_params: MaterialParams,
                                               maximum_shear_stress: float = 100.0,
                                               shear_stress_step_size: float = 0.1,
                                               maximum_axial_strain: float = 0.02,
                                               maximum_fiber_misalignment: float = 20,
                                               fiber_misalignment_step_size: float = 0.1,
                                               kink_width: float = None,
                                               gauge_length: float = None) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate compression strength from measured fiber orientation distribution.
    
    Implements a micromechanical model that accounts for fiber misalignment effects
    on composite compression strength. Uses power-law plasticity for matrix behavior
    and superposition of stress contributions from different misalignment angles.

    Args:
        orientation_profile: 3D array of fiber misalignment angles in degrees.
                           Should contain the actual measured orientation data.
        material_params: Complete set of material properties for the composite.
        maximum_shear_stress: Upper bound for shear stress integration (MPa).
        shear_stress_step_size: Resolution for shear stress discretization (MPa).
        maximum_axial_strain: Maximum compressive strain for analysis.
        maximum_fiber_misalignment: Upper bound for misalignment angle range (degrees).
        fiber_misalignment_step_size: Angular resolution for misalignment discretization (degrees).

    Returns:
        Tuple containing:
        - compression_strength: Peak compressive stress (MPa)
        - ultimate_strain: Strain at peak stress
        - stress_curve: Complete stress-strain curve array
        - strain_array: Corresponding strain values
        
    Raises:
        ValueError: If the misalignment range doesn't capture enough probability mass (< 99.9%).
        
    Note:
        Uses histogram analysis of the orientation profile to determine probability
        weights for different misalignment angles. Applies incremental loading with
        power-law matrix plasticity model.
    """
    logger.info(f"Computing compression strength from orientation profile: profile_shape={orientation_profile.shape}")
    logger.debug(f"Material params: E1={material_params.longitudinal_modulus}, E2={material_params.transverse_modulus}, G={material_params.shear_modulus}")
    logger.debug(f"Analysis params: max_shear_stress={maximum_shear_stress}, max_strain={maximum_axial_strain}, max_misalignment={maximum_fiber_misalignment}°")

    E1 = material_params.longitudinal_modulus
    E2 = material_params.transverse_modulus
    nu = material_params.poisson_ratio
    G = material_params.shear_modulus
    tau_y = material_params.tau_y
    K = material_params.K
    n = material_params.n

    shear_stress_array = np.linspace(0, maximum_shear_stress, int(maximum_shear_stress/shear_stress_step_size)+1)
    shear_strain_array = (shear_stress_array/G) + K*(shear_stress_array/tau_y)**n
    misalignment_array = np.linspace(0, maximum_fiber_misalignment, int(maximum_fiber_misalignment/fiber_misalignment_step_size)+1)

    # Secant shear modulus: G_sec = τ/γ
    with np.errstate(divide='ignore', invalid='ignore'):
        secant_shear_modulus_array = shear_stress_array / shear_strain_array
    secant_shear_modulus_array[0] = G

    logger.debug(f"Discretization: n_shear_steps={len(shear_stress_array)}, n_misalignment_steps={len(misalignment_array)}")

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/secant_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

    # Replace NaN values with maximum value
    np.nan_to_num(axial_stress_matrix[0, :], copy=False)
    axial_stress_matrix[0, 0] = np.max(axial_stress_matrix[0, :])

    axial_strain_matrix = axial_stress_matrix*axial_compliance_matrix

    # Replace constant intervals of strain array with interpolated values
    # Zero fiber misalignment case
    maximum_axial_stress_value = np.max(axial_stress_matrix[0, :])
    maximum_axial_strain_value = axial_strain_matrix[0, np.argmax(axial_stress_matrix[0, :])]
    interpolation_function = interp1d(np.array([0, maximum_axial_strain_value]), np.array([0, maximum_axial_stress_value]), kind='linear', bounds_error=False, fill_value="extrapolate")
    constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
    constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
    axial_strain_matrix[0, :] = constant_interval_strain_array
    axial_stress_matrix[0, :] = constant_interval_stress_array

    # Non-zero fiber misalignment cases
    for i in range(1, len(misalignment_array)):
        maximum_argment = argrelmax(axial_strain_matrix[i, :])[0]
        minimum_argment = argrelmin(axial_strain_matrix[i, :])[0]
        if len(maximum_argment) == 0:
            interpolation_function = interp1d(axial_strain_matrix[i, :], axial_stress_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array
        else:
            interpolation_function_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], axial_stress_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolation_function_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], axial_stress_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            for j, value in enumerate(constant_interval_strain_array):
                if value <= axial_strain_matrix[i, maximum_argment[0]]:
                    constant_interval_stress_array[j] = interpolation_function_left(value)
                else:
                    constant_interval_stress_array[j] = interpolation_function_right(value)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array

    # Probability distribution of fiber misalignment from measured data
    flatten_orientation_profile = orientation_profile.ravel()
    logger.debug(f"Orientation profile stats: mean={np.mean(flatten_orientation_profile):.2f}°, std={np.std(flatten_orientation_profile):.2f}°, range=[{np.min(flatten_orientation_profile):.2f}, {np.max(flatten_orientation_profile):.2f}]°")

    probability, bins = np.histogram(flatten_orientation_profile,
                                     bins=int(maximum_fiber_misalignment/fiber_misalignment_step_size),
                                     density=True, range=(0, maximum_fiber_misalignment))
    for i in range(len(probability)):
        probability[i] = probability[i]*fiber_misalignment_step_size

    total_probability = np.sum(probability)
    logger.debug(f"Probability distribution: total_probability={total_probability:.4f}")

    if total_probability < 0.999:
        logger.warning(f"Low total probability: {total_probability:.4f} < 0.999")
        raise ValueError(f"The range of fiber misalignment is too small. Total probability is {total_probability} less than 1.0.\nConsider using another profile.")
    
    # Weighted axial stress matrix
    weighted_axial_stress_matrix = np.copy(axial_stress_matrix)
    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix[i, :] = axial_stress_matrix[i, :]*probability[i-1]
    
    # Superposition of axial stress matrix
    superposition_axial_stress_array = np.ndarray(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_axial_stress_matrix[1:, i])
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    # Apply kink width and gauge length correction if specified
    # ε̄_x = ε_x * (w_k/L_g) + σ_x/E_11 * (1 - w_k/L_g)
    if kink_width is not None and gauge_length is not None and gauge_length > 0:
        w_k_over_L_g = kink_width / gauge_length
        logger.debug(f"Applying kink correction: w_k={kink_width}, L_g={gauge_length}, ratio={w_k_over_L_g:.4f}")
        axial_strain_array = (axial_strain_array * w_k_over_L_g +
                              superposition_axial_stress_array / E1 * (1 - w_k_over_L_g))

    if not find_peaks(superposition_axial_stress_array)[0].size == 0:
        compression_strength_index = find_peaks(superposition_axial_stress_array)[0][0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
        logger.info(f"Compression strength (from profile) found at peak: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]
        logger.info(f"Compression strength (from profile) at maximum: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array

def estimate_compression_strength(initial_misalignment: float,
                         standard_deviation: float,
                         material_params: MaterialParams,
                         maximum_shear_stress: float = 100.0,
                         shear_stress_step_size: float = 0.1,
                         maximum_axial_strain: float = 0.02,
                         maximum_fiber_misalignment: float = 20,
                         fiber_misalignment_step_size: float = 0.1,
                         kink_width: float = None,
                         gauge_length: float = None) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Calculate compression strength assuming Gaussian distribution of fiber misalignment.
    
    Theoretical model variant that uses a normal distribution to describe fiber
    misalignment rather than measured data. Applies symmetric probability weights
    for positive and negative misalignment angles around the mean.

    Args:
        initial_misalignment: Mean fiber misalignment angle in degrees.
        standard_deviation: Standard deviation of misalignment distribution in degrees.
        material_params: Complete set of material properties for the composite.
        maximum_shear_stress: Upper bound for shear stress integration (MPa).
        shear_stress_step_size: Resolution for shear stress discretization (MPa).
        maximum_axial_strain: Maximum compressive strain for analysis.
        maximum_fiber_misalignment: Upper bound for misalignment angle range (degrees).
        fiber_misalignment_step_size: Angular resolution for misalignment discretization (degrees).

    Returns:
        Tuple containing:
        - compression_strength: Peak compressive stress (MPa)
        - ultimate_strain: Strain at peak stress
        - stress_curve: Complete stress-strain curve array
        - strain_array: Corresponding strain values

    Raises:
        ValueError: If the misalignment range doesn't capture enough probability mass (< 99.9%).

    Note:
        Uses scipy.stats.norm.cdf for probability calculations. Considers both positive
        and negative misalignments symmetrically around the mean value. Requires larger
        misalignment range for highly dispersed distributions.
    """
    logger.info(f"Computing compression strength with Gaussian distribution: mean={initial_misalignment:.2f}°, std={standard_deviation:.2f}°")
    logger.debug(f"Material params: E1={material_params.longitudinal_modulus}, E2={material_params.transverse_modulus}, G={material_params.shear_modulus}")
    logger.debug(f"Analysis params: max_shear_stress={maximum_shear_stress}, max_strain={maximum_axial_strain}, max_misalignment={maximum_fiber_misalignment}°")

    E1 = material_params.longitudinal_modulus
    E2 = material_params.transverse_modulus
    nu = material_params.poisson_ratio
    G = material_params.shear_modulus
    tau_y = material_params.tau_y
    K = material_params.K
    n = material_params.n

    shear_stress_array = np.linspace(0, maximum_shear_stress, int(maximum_shear_stress/shear_stress_step_size)+1)
    shear_strain_array = (shear_stress_array/G) + K*(shear_stress_array/tau_y)**n
    misalignment_array = np.linspace(0, maximum_fiber_misalignment, int(maximum_fiber_misalignment/fiber_misalignment_step_size)+1)

    # Secant shear modulus: G_sec = τ/γ
    with np.errstate(divide='ignore', invalid='ignore'):
        secant_shear_modulus_array = shear_stress_array / shear_strain_array
    secant_shear_modulus_array[0] = G

    logger.debug(f"Discretization: n_shear_steps={len(shear_stress_array)}, n_misalignment_steps={len(misalignment_array)}")

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/secant_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

    # Replace NaN values with maximum value
    np.nan_to_num(axial_stress_matrix[0, :], copy=False)
    axial_stress_matrix[0, 0] = np.max(axial_stress_matrix[0, :])

    axial_strain_matrix = axial_stress_matrix*axial_compliance_matrix

    # Replace constant intervals of strain array with interpolated values
    # Zero fiber misalignment case
    maximum_axial_stress_value = np.max(axial_stress_matrix[0, :])
    maximum_axial_strain_value = axial_strain_matrix[0, np.argmax(axial_stress_matrix[0, :])]
    interpolation_function = interp1d(np.array([0, maximum_axial_strain_value]), np.array([0, maximum_axial_stress_value]), kind='linear', bounds_error=False, fill_value="extrapolate")
    constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
    constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
    axial_strain_matrix[0, :] = constant_interval_strain_array
    axial_stress_matrix[0, :] = constant_interval_stress_array

    # Non-zero fiber misalignment cases
    for i in range(1, len(misalignment_array)):
        maximum_argment = argrelmax(axial_strain_matrix[i, :])[0]
        minimum_argment = argrelmin(axial_strain_matrix[i, :])[0]
        if len(maximum_argment) == 0:
            interpolation_function = interp1d(axial_strain_matrix[i, :], axial_stress_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array
        else:
            interpolation_function_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], axial_stress_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolation_function_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], axial_stress_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            for j, value in enumerate(constant_interval_strain_array):
                if value <= axial_strain_matrix[i, maximum_argment[0]]:
                    constant_interval_stress_array[j] = interpolation_function_left(value)
                else:
                    constant_interval_stress_array[j] = interpolation_function_right(value)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array

    # Probability distribution of fiber misalignment
    mean_value = np.deg2rad(initial_misalignment)
    std_value = np.deg2rad(standard_deviation)

    # Right-side (positive misalignments)
    probabilties_right_array = []
    for i in range(1, len(misalignment_array)):
        probabilty = stats.norm.cdf(np.deg2rad(misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)\
                   - stats.norm.cdf(np.deg2rad(misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        probabilties_right_array.append(probabilty)
    probabilties_right_array = np.array(probabilties_right_array)

    # Left-side (negative misalignments)
    probabilties_left_array = []
    for i in range(1, len(misalignment_array)):
        probabilty = stats.norm.cdf(np.deg2rad(-misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)\
                   - stats.norm.cdf(np.deg2rad(-misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        probabilties_left_array.append(probabilty)
    probabilties_left_array = np.array(probabilties_left_array)

    # Center (near-zero misalignments)
    probabilty_center = stats.norm.cdf(np.deg2rad(fiber_misalignment_step_size/2), mean_value, std_value)\
                        - stats.norm.cdf(np.deg2rad(-fiber_misalignment_step_size/2), mean_value, std_value)

    total_probability = np.sum(probabilties_right_array) + np.sum(probabilties_left_array) + probabilty_center
    logger.debug(f"Probability distribution: total_probability={total_probability:.4f}")

    if total_probability < 0.999:
        logger.warning(f"Low total probability: {total_probability:.4f} < 0.999")
        raise ValueError(f"The range of fiber misalignment is too small. Total probability is {total_probability} less than 1.0.\nPlease increase the range of fiber misalignment.")

    # Weighted axial stress matrix
    weighted_axial_stress_matrix_right = np.copy(axial_stress_matrix)
    weighted_axial_stress_matrix_left = np.copy(axial_stress_matrix)

    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_right[i, :] = axial_stress_matrix[i, :]*probabilties_right_array[i-1]
    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_left[i, :] = axial_stress_matrix[i, :]*probabilties_left_array[i-1]
    weighted_axial_stress_center = axial_stress_matrix[0, :]*probabilty_center

    # Superposition of axial stress matrix (ε-space)
    superposition_axial_stress_array = np.ndarray(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_axial_stress_matrix_right[1:, i]) + np.sum(weighted_axial_stress_matrix_left[1:, i])
    superposition_axial_stress_array += weighted_axial_stress_center
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    # Apply kink width and gauge length correction if specified
    # ε̄_x = ε_x * (w_k/L_g) + σ_x/E_11 * (1 - w_k/L_g)
    if kink_width is not None and gauge_length is not None and gauge_length > 0:
        w_k_over_L_g = kink_width / gauge_length
        logger.debug(f"Applying kink correction: w_k={kink_width}, L_g={gauge_length}, ratio={w_k_over_L_g:.4f}")
        axial_strain_array = (axial_strain_array * w_k_over_L_g +
                              superposition_axial_stress_array / E1 * (1 - w_k_over_L_g))

    if not find_peaks(superposition_axial_stress_array)[0].size == 0:
        compression_strength_index = find_peaks(superposition_axial_stress_array)[0][0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
        logger.info(f"Compression strength (Gaussian) found at peak: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]
        logger.info(f"Compression strength (Gaussian) at maximum: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")

    # Log the shear strain (gamma) at 1σ misalignment angle corresponding to the CS peak
    theta_1sigma = initial_misalignment + standard_deviation
    theta_1sigma_rad = np.deg2rad(theta_1sigma)
    axial_stress_1sigma = shear_stress_array / (theta_1sigma_rad + shear_strain_array)
    axial_compliance_1sigma = (1/E1 + (1/E2)*(theta_1sigma_rad + shear_strain_array)**4
                               + (1/secant_shear_modulus_array - 2*nu/E1)*(theta_1sigma_rad + shear_strain_array)**2)
    axial_strain_1sigma = axial_stress_1sigma * axial_compliance_1sigma
    max_strain_arg = argrelmax(axial_strain_1sigma)[0]
    search_end = max_strain_arg[0] if len(max_strain_arg) > 0 else len(axial_strain_1sigma)
    idx_1sigma = np.argmin(np.abs(axial_strain_1sigma[:search_end] - ultimate_strain))
    gamma_at_1sigma = shear_strain_array[idx_1sigma]
    tau_at_1sigma = shear_stress_array[idx_1sigma]
    logger.info(f"Shear strain at 1σ misalignment (θ={theta_1sigma:.2f}°): "
                f"γ={gamma_at_1sigma:.6f}, τ={tau_at_1sigma:.2f} MPa")

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array


def compute_critical_stress(material_params: MaterialParams,
                            kink_width: float,
                            maximum_shear_stress: float = 100.0,
                            shear_stress_step_size: float = 0.1,
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute critical stress σ_cr as a function of shear strain for a given kink width.

    σ_cr(γ) = G_tan(γ) + (π² V_f^2D E_f / 12) · (d_f / w)²

    where G_tan = dτ/dγ is the tangent shear modulus from the power-law plasticity model.

    Args:
        material_params: Material properties (must include Vf and fiber_diameter).
        kink_width: Kink band width (same unit as fiber_diameter).
        maximum_shear_stress: Upper bound for shear stress discretization (MPa).
        shear_stress_step_size: Resolution for shear stress discretization (MPa).

    Returns:
        Tuple containing:
        - critical_stress_array: σ_cr values (MPa)
        - shear_strain_array: Corresponding shear strain values
        - shear_stress_array: Corresponding shear stress values

    Raises:
        ValueError: If material_params is missing Vf or fiber_diameter.
    """
    if material_params.Vf is None or material_params.fiber_diameter is None:
        raise ValueError("MaterialParams must include Vf and fiber_diameter for critical stress calculation.")

    Vf = material_params.Vf
    fiber_diameter = material_params.fiber_diameter
    E1 = material_params.longitudinal_modulus
    G = material_params.shear_modulus
    tau_y = material_params.tau_y
    K = material_params.K
    n = material_params.n

    # Derived parameters
    tm = fiber_diameter * (np.sqrt(np.pi / (2 * np.sqrt(3) * Vf)) - 1)
    Vf_2D = fiber_diameter / (fiber_diameter + tm)
    Ef = E1 / Vf

    shear_stress_array = np.linspace(0, maximum_shear_stress,
                                     int(maximum_shear_stress / shear_stress_step_size) + 1)
    shear_strain_array = shear_stress_array / G + K * (shear_stress_array / tau_y) ** n

    # Tangent shear modulus: G_tan = dτ/dγ = 1 / (dγ/dτ)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_gamma_d_tau = 1 / G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
        tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    coefficient = (np.pi ** 2 * Vf_2D * Ef) / 12
    critical_stress_array = tangent_shear_modulus_array + coefficient * (fiber_diameter / kink_width) ** 2

    logger.info(f"Critical stress computed: w={kink_width}, Vf_2D={Vf_2D:.4f}, Ef={Ef:.1f} MPa, "
                f"σ_cr range=[{critical_stress_array[-1]:.1f}, {critical_stress_array[0]:.1f}] MPa")

    return critical_stress_array, shear_strain_array, shear_stress_array


def estimate_tangent_kink_width(initial_misalignment: float,
                                standard_deviation: float,
                                material_params: MaterialParams,
                                maximum_shear_stress: float = 100.0,
                                shear_stress_step_size: float = 0.1,
                                maximum_axial_strain: float = 0.02,
                                maximum_fiber_misalignment: float = 20,
                                fiber_misalignment_step_size: float = 0.1) -> dict:
    """
    Estimate the tangent kink band width where σ_cr and σ_x are tangent in ε-space.

    Finds the kink width w such that σ_cr(ε_x) = E[G_tan](ε_x) + C(w) touches
    the superposition stress σ_x(ε_x) at exactly one point (tangent condition).

    The tangent point occurs at the maximum of (σ_x - E[G_tan]), giving:
        w = d_f · √(coeff / max(σ_x - E[G_tan]))

    Args:
        initial_misalignment: Mean fiber misalignment angle in degrees.
        standard_deviation: Standard deviation of misalignment distribution in degrees.
        material_params: Material properties (must include Vf and fiber_diameter).
        maximum_shear_stress: Upper bound for shear stress integration (MPa).
        shear_stress_step_size: Resolution for shear stress discretization (MPa).
        maximum_axial_strain: Maximum compressive strain for analysis.
        maximum_fiber_misalignment: Upper bound for misalignment angle range (degrees).
        fiber_misalignment_step_size: Angular resolution for misalignment discretization (degrees).

    Returns:
        dict with keys:
        - tangent_kink_width: Kink width where σ_cr and σ_x are tangent
        - tangent_strain: Axial strain at the tangent point
        - tangent_stress: Axial stress at the tangent point
        - compression_strength: Peak compressive stress (MPa)
        - ultimate_strain: Strain at peak stress
        - axial_stress_array: Superposition stress in ε-space (MPa)
        - axial_strain_array: Corresponding axial strain values
        - expected_g_tan_array: E[G_tan] in ε-space (MPa)
        - critical_stress_array: σ_cr at tangent kink width in ε-space (MPa)

    Raises:
        ValueError: If material_params is missing Vf or fiber_diameter.
        ValueError: If σ_x never exceeds E[G_tan].
    """
    if material_params.Vf is None or material_params.fiber_diameter is None:
        raise ValueError("MaterialParams must include Vf and fiber_diameter for kink band analysis.")

    logger.info(f"Estimating tangent kink width: mean={initial_misalignment:.2f}°, std={standard_deviation:.2f}°")

    E1 = material_params.longitudinal_modulus
    E2 = material_params.transverse_modulus
    nu = material_params.poisson_ratio
    G = material_params.shear_modulus
    tau_y = material_params.tau_y
    K = material_params.K
    n = material_params.n
    Vf = material_params.Vf
    fiber_diameter = material_params.fiber_diameter

    # Derived parameters
    tm = fiber_diameter * (np.sqrt(np.pi / (2 * np.sqrt(3) * Vf)) - 1)
    Vf_2D = fiber_diameter / (fiber_diameter + tm)
    Ef = E1 / Vf
    coefficient = (np.pi ** 2 * Vf_2D * Ef) / 12

    shear_stress_array = np.linspace(0, maximum_shear_stress, int(maximum_shear_stress/shear_stress_step_size)+1)
    shear_strain_array = (shear_stress_array/G) + K*(shear_stress_array/tau_y)**n
    misalignment_array = np.linspace(0, maximum_fiber_misalignment, int(maximum_fiber_misalignment/fiber_misalignment_step_size)+1)

    # Secant shear modulus
    with np.errstate(divide='ignore', invalid='ignore'):
        secant_shear_modulus_array = shear_stress_array / shear_strain_array
    secant_shear_modulus_array[0] = G

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/secant_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

    np.nan_to_num(axial_stress_matrix[0, :], copy=False)
    axial_stress_matrix[0, 0] = np.max(axial_stress_matrix[0, :])

    axial_strain_matrix = axial_stress_matrix*axial_compliance_matrix

    # Tangent shear modulus
    with np.errstate(divide='ignore', invalid='ignore'):
        d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
        tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    # G_tan matrix for ε-space interpolation
    g_tan_matrix = np.tile(tangent_shear_modulus_array, (misalignment_array.size, 1))

    # ε-space interpolation (zero misalignment)
    maximum_axial_stress_value = np.max(axial_stress_matrix[0, :])
    maximum_axial_strain_value = axial_strain_matrix[0, np.argmax(axial_stress_matrix[0, :])]
    interpolation_function = interp1d(np.array([0, maximum_axial_strain_value]), np.array([0, maximum_axial_stress_value]), kind='linear', bounds_error=False, fill_value="extrapolate")
    g_tan_interp_function = interp1d(np.array([0, maximum_axial_strain_value]), np.array([G, g_tan_matrix[0, np.argmax(axial_stress_matrix[0, :])]]), kind='linear', bounds_error=False, fill_value="extrapolate")
    constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
    constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
    axial_strain_matrix[0, :] = constant_interval_strain_array
    axial_stress_matrix[0, :] = constant_interval_stress_array
    g_tan_matrix[0, :] = g_tan_interp_function(constant_interval_strain_array)

    # ε-space interpolation (non-zero misalignment)
    for i in range(1, len(misalignment_array)):
        maximum_argment = argrelmax(axial_strain_matrix[i, :])[0]
        minimum_argment = argrelmin(axial_strain_matrix[i, :])[0]
        if len(maximum_argment) == 0:
            interpolation_function = interp1d(axial_strain_matrix[i, :], axial_stress_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            g_tan_interp_function = interp1d(axial_strain_matrix[i, :], g_tan_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array
            g_tan_matrix[i, :] = g_tan_interp_function(constant_interval_strain_array)
        else:
            interpolation_function_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], axial_stress_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolation_function_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], axial_stress_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            g_tan_interp_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], g_tan_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            g_tan_interp_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], g_tan_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
            constant_interval_g_tan_array = np.zeros_like(constant_interval_strain_array)
            for j, value in enumerate(constant_interval_strain_array):
                if value <= axial_strain_matrix[i, maximum_argment[0]]:
                    constant_interval_stress_array[j] = interpolation_function_left(value)
                    constant_interval_g_tan_array[j] = g_tan_interp_left(value)
                else:
                    constant_interval_stress_array[j] = interpolation_function_right(value)
                    constant_interval_g_tan_array[j] = g_tan_interp_right(value)
            axial_strain_matrix[i, :] = constant_interval_strain_array
            axial_stress_matrix[i, :] = constant_interval_stress_array
            g_tan_matrix[i, :] = constant_interval_g_tan_array

    # Probability distribution (Gaussian)
    mean_value = np.deg2rad(initial_misalignment)
    std_value = np.deg2rad(standard_deviation)

    probabilties_right_array = np.array([
        stats.norm.cdf(np.deg2rad(misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)
        - stats.norm.cdf(np.deg2rad(misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        for i in range(1, len(misalignment_array))
    ])
    probabilties_left_array = np.array([
        stats.norm.cdf(np.deg2rad(-misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)
        - stats.norm.cdf(np.deg2rad(-misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        for i in range(1, len(misalignment_array))
    ])
    probabilty_center = stats.norm.cdf(np.deg2rad(fiber_misalignment_step_size/2), mean_value, std_value)\
                        - stats.norm.cdf(np.deg2rad(-fiber_misalignment_step_size/2), mean_value, std_value)

    total_probability = np.sum(probabilties_right_array) + np.sum(probabilties_left_array) + probabilty_center
    if total_probability < 0.999:
        raise ValueError(f"The range of fiber misalignment is too small. Total probability is {total_probability}.")

    # Superposition (ε-space)
    weighted_right = np.copy(axial_stress_matrix)
    weighted_left = np.copy(axial_stress_matrix)
    for i in range(1, len(misalignment_array)):
        weighted_right[i, :] = axial_stress_matrix[i, :] * probabilties_right_array[i-1]
        weighted_left[i, :] = axial_stress_matrix[i, :] * probabilties_left_array[i-1]
    weighted_center = axial_stress_matrix[0, :] * probabilty_center

    superposition_axial_stress_array = np.zeros(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_right[1:, i]) + np.sum(weighted_left[1:, i])
    superposition_axial_stress_array += weighted_center

    # Expected G_tan in ε-space (harmonic mean): Ḡ'₁₂ = 1 / Σ (P_i / G'₁₂_i)
    compliance_sum = np.zeros(axial_stress_matrix.shape[1])
    for i in range(1, len(misalignment_array)):
        compliance_sum += (probabilties_right_array[i-1] + probabilties_left_array[i-1]) / g_tan_matrix[i, :]
    compliance_sum += probabilty_center / g_tan_matrix[0, :]
    expected_g_tan_array = 1.0 / compliance_sum
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    # CS peak
    peaks = find_peaks(superposition_axial_stress_array)[0]
    if len(peaks) > 0:
        compression_strength_index = peaks[0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]
    logger.info(f"Compression strength: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")

    # Tangent kink width: max(σ_x - E[G_tan]) determines the tangent condition
    diff_array = superposition_axial_stress_array - expected_g_tan_array
    tangent_index = np.argmax(diff_array)
    max_diff = diff_array[tangent_index]

    if max_diff <= 0:
        raise ValueError("σ_x never exceeds E[G_tan] — no valid kink width.")

    tangent_kink_width = fiber_diameter * np.sqrt(coefficient / max_diff)
    tangent_strain = axial_strain_array[tangent_index]
    tangent_stress = superposition_axial_stress_array[tangent_index]

    # σ_cr at the tangent kink width
    critical_stress_array = expected_g_tan_array + coefficient * (fiber_diameter / tangent_kink_width) ** 2

    logger.info(f"Tangent kink width: w={tangent_kink_width:.6f}, "
                f"tangent at ε_x={tangent_strain:.6f}, σ_x={tangent_stress:.2f} MPa")

    # Shear strain at kink onset for θ = mean + 1σ
    theta_1sigma = initial_misalignment + standard_deviation
    theta_1sigma_rad = np.deg2rad(theta_1sigma)
    axial_stress_1sigma = shear_stress_array / (theta_1sigma_rad + shear_strain_array)
    axial_compliance_1sigma = (1/E1 + (1/E2)*(theta_1sigma_rad + shear_strain_array)**4
                               + (1/secant_shear_modulus_array - 2*nu/E1)*(theta_1sigma_rad + shear_strain_array)**2)
    axial_strain_1sigma = axial_stress_1sigma * axial_compliance_1sigma
    max_strain_arg = argrelmax(axial_strain_1sigma)[0]
    search_end = max_strain_arg[0] if len(max_strain_arg) > 0 else len(axial_strain_1sigma)
    idx_kink = np.argmin(np.abs(axial_strain_1sigma[:search_end] - tangent_strain))
    gamma_at_kink = shear_strain_array[idx_kink]
    tau_at_kink = shear_stress_array[idx_kink]
    logger.info(f"Shear strain at kink onset (θ=1σ={theta_1sigma:.2f}°): "
                f"γ={gamma_at_kink:.6f}, τ={tau_at_kink:.2f} MPa")

    return {
        "tangent_kink_width": tangent_kink_width,
        "tangent_strain": tangent_strain,
        "tangent_stress": tangent_stress,
        "compression_strength": compression_strength,
        "ultimate_strain": ultimate_strain,
        "axial_stress_array": superposition_axial_stress_array,
        "axial_strain_array": axial_strain_array,
        "expected_g_tan_array": expected_g_tan_array,
        "critical_stress_array": critical_stress_array,
    }


@dataclass
class BendingAnalysisResult:
    """Results from three-point bending analysis."""

    # Input parameters
    h: float  # Beam height (mm)
    L: float  # Span length (mm)
    b: float  # Beam width (mm)

    # Analysis arrays
    displacement: np.ndarray  # Bending displacement (mm)
    moment: np.ndarray  # Bending moment (N*mm)
    reaction_force: np.ndarray  # Reaction force at supports (N)
    neutral_axis: np.ndarray  # Neutral axis position (mm)

    # 2D arrays: (z_position, load_step)
    z_array: np.ndarray  # Through-thickness z coordinates (mm)
    strain_arrays: np.ndarray  # Strain distribution
    stress_arrays: np.ndarray  # Stress distribution (MPa)

    # JIS bending stress-strain
    jis_bending_strain: np.ndarray  # Surface strain
    jis_bending_stress: np.ndarray  # Bending stress (MPa)

    # Peak values
    flexural_strength: float  # Maximum bending stress (MPa)
    strain_at_flexural_strength: float  # Strain at flexural strength


def create_stress_strain_interpolator(
    kink_result: dict,
    material_params: MaterialParams,
    interpolation_kind: str = 'linear',
    tensile_alpha: float = 0.0,
) -> Callable:
    """
    Create a stress-strain interpolation function from estimate_tangent_kink_width result.

    Uses the stress-strain curve from kink band analysis to create an interpolation
    function that maps strain values to stress values.

    Args:
        kink_result: Dict returned by estimate_tangent_kink_width.
        material_params: Material properties for the composite.
        interpolation_kind: Type of interpolation ('linear', 'cubic', 'quadratic', etc.).
        tensile_alpha: Nonlinearity parameter for tensile stiffening due to
            carbon fiber crystallite re-orientation. The tensile stress is computed as
            sigma = E1 * (1 + alpha * epsilon) * epsilon, where epsilon is the
            tensile strain. Typical value for T700-class PAN fibers is ~7.
            Set to 0 for linear elastic tension (default).

    Returns:
        A function that takes a strain array and returns the corresponding stress array.
    """
    # Use stress-strain arrays from kink band result
    stress_array = kink_result["axial_stress_array"]
    strain_array = kink_result["axial_strain_array"]

    E1 = material_params.longitudinal_modulus

    # Create interpolation function from kink band curve directly
    interpolator = interp1d(
        strain_array,
        stress_array,
        kind=interpolation_kind,
        bounds_error=False,
        fill_value=(stress_array[0], stress_array[-1])
    )

    def get_stress_from_strain(strain_input: np.ndarray) -> np.ndarray:
        """
        Compute stress values from strain input.

        For tensile strain (positive), uses nonlinear stiffening model:
            sigma = E1 * (1 + alpha * epsilon) * epsilon
        where alpha accounts for crystallite re-orientation in PAN-based carbon fibers.
        When alpha=0, this reduces to linear elastic (sigma = E1 * epsilon).

        For compressive strain (negative), uses the interpolated compression curve.

        Args:
            strain_input: Array of strain values (positive = tension, negative = compression).

        Returns:
            Array of corresponding stress values (MPa).
        """
        strain_input = np.asarray(strain_input)
        stress_output = np.zeros_like(strain_input, dtype=float)

        # Tensile region (positive strain): nonlinear stiffening
        tensile_mask = strain_input >= 0
        eps_t = strain_input[tensile_mask]
        stress_output[tensile_mask] = E1 * (1.0 + tensile_alpha * eps_t) * eps_t

        # Compressive region (negative strain): use interpolated compression curve
        compressive_mask = strain_input < 0
        abs_strain = -strain_input[compressive_mask]
        comp_stress = interpolator(abs_strain)

        stress_output[compressive_mask] = -comp_stress

        return stress_output

    return get_stress_from_strain


def find_neutral_axis_from_stress(
    z_array: np.ndarray,
    curvature: float,
    stress_func: Callable,
    z_neutral_prev: float | None = None,
    bracket_halfwidth: float | None = None,
) -> float:
    """
    Find neutral axis position from stress equilibrium condition.

    Search for z_n such that integral of sigma(z) dz = 0 using Simpson's rule.

    When the constitutive law sigma(eps) has a softening branch, the equilibrium
    can admit multiple roots for a given curvature. To track the physically
    correct branch continuously across an incremental analysis, pass the previous
    step's neutral axis as ``z_neutral_prev``; the solver will first attempt a
    narrow bracket around it and fall back to a full-range search only if needed.

    Args:
        z_array: Array of z coordinates
        curvature: Curvature kappa
        stress_func: Function that computes stress from strain
        z_neutral_prev: Optional previous-step neutral axis used as a continuation
            seed. When provided, a narrow bracket around this value is tried first.
        bracket_halfwidth: Half-width of the narrow bracket around
            ``z_neutral_prev``. Defaults to 10% of the z-array span.

    Returns:
        z_neutral: Neutral axis position
    """
    z_min, z_max = z_array[0], z_array[-1]
    dz = z_array[1] - z_array[0]

    def force_residual(z_n):
        strain = (z_array - z_n) * curvature
        stress = stress_func(strain)
        axial_force = simpson(stress, x=z_array)
        return axial_force

    if z_neutral_prev is not None:
        if bracket_halfwidth is None:
            bracket_halfwidth = 0.1 * (z_max - z_min)
        lo = max(z_min + dz, z_neutral_prev - bracket_halfwidth)
        hi = min(z_max - dz, z_neutral_prev + bracket_halfwidth)
        if lo < hi:
            try:
                return brentq(force_residual, lo, hi)
            except ValueError:
                pass  # fall back to full-range search

    try:
        z_neutral = brentq(force_residual, z_min + dz, z_max - dz)
    except ValueError as e:
        raise ValueError(
            f"Failed to find neutral axis position. "
            f"Force residual has no sign change in range [{z_min + dz:.4f}, {z_max - dz:.4f}]. "
            f"Original error: {e}"
        )

    return z_neutral


def analyze_three_point_bending(
    h: float,
    L: float,
    b: float,
    stress_func: Callable,
    failure_strain: float,
    disp_step_size: float = 0.1,
    z_step_size: float = 0.1,
    bending_mode: str = '3-point',
    load_span_ratio: float = 0.5,
) -> BendingAnalysisResult:
    """
    Bending analysis with moving neutral axis (3-point or 4-point).

    Performs incremental bending analysis where the neutral axis position
    is calculated from stress equilibrium at each load step.

    Args:
        h: Beam height (mm)
        L: Span length (mm)
        b: Beam width (mm)
        stress_func: Function that returns stress (MPa) given strain
        failure_strain: Maximum compressive strain (absolute value)
        disp_step_size: Displacement increment (mm)
        z_step_size: Through-thickness discretization (mm)
        bending_mode: '3-point' or '4-point' bending
        load_span_ratio: For 4-point bending, ratio of inner span to outer span (default 0.5)

    Returns:
        BendingAnalysisResult: Complete analysis results
    """
    if bending_mode not in ['3-point', '4-point']:
        raise ValueError(f"Invalid bending_mode: {bending_mode}. Must be '3-point' or '4-point'.")
    # Through-thickness discretization
    z_array = np.linspace(-h / 2, h / 2, int(h // z_step_size) + 1)

    # Calculate geometry-dependent parameters
    if bending_mode == '3-point':
        curvature_coeff = 12.0 / L**2
    else:  # 4-point bending
        a = L * (1 - load_span_ratio) / 2
        curvature_coeff = 24.0 / (3 * L**2 - 4 * a**2)

    # Maximum displacement at failure strain
    max_disp = failure_strain / (0.5 * h * curvature_coeff)

    n_steps = int(max_disp // disp_step_size) + 1

    # Storage arrays
    bending_disp = np.zeros(n_steps)
    moment_array = np.zeros(n_steps)
    strain_arrays = np.zeros((len(z_array), n_steps))
    stress_arrays = np.zeros((len(z_array), n_steps))
    neutral_axis_array = np.zeros(n_steps)

    # Incremental analysis loop
    for step in range(n_steps):
        delta = step * disp_step_size
        bending_disp[step] = delta

        curvature = curvature_coeff * delta

        if step == 0 or curvature < 1e-12:
            z_neutral = 0.0
            strain_profile = np.zeros_like(z_array)
            stress_profile = np.zeros_like(z_array)
        else:
            z_neutral = find_neutral_axis_from_stress(z_array, curvature, stress_func)
            strain_profile = (z_array - z_neutral) * curvature
            stress_profile = stress_func(strain_profile)

        neutral_axis_array[step] = z_neutral
        strain_arrays[:, step] = strain_profile
        stress_arrays[:, step] = stress_profile

        moment = simpson(stress_profile * z_array, x=z_array) * b
        moment_array[step] = moment

    # Calculate reaction force and bending stress based on mode
    if bending_mode == '3-point':
        reaction_force = 4 * moment_array / L
        jis_bending_stress = 3 * reaction_force * L / (2 * b * h**2)
        jis_bending_strain = 6 * bending_disp * h / L**2
    else:  # 4-point
        a = L * (1 - load_span_ratio) / 2
        reaction_force = 2 * moment_array / a
        jis_bending_stress = 6 * moment_array / (b * h**2)
        jis_bending_strain = curvature_coeff * bending_disp * h / 2

    # Find maximum bending stress (flexural strength)
    max_stress_idx = np.argmax(jis_bending_stress)
    flexural_strength = jis_bending_stress[max_stress_idx]
    strain_at_max = jis_bending_strain[max_stress_idx]

    return BendingAnalysisResult(
        h=h,
        L=L,
        b=b,
        displacement=bending_disp,
        moment=moment_array,
        reaction_force=reaction_force,
        neutral_axis=neutral_axis_array,
        z_array=z_array,
        strain_arrays=strain_arrays,
        stress_arrays=stress_arrays,
        jis_bending_strain=jis_bending_strain,
        jis_bending_stress=jis_bending_stress,
        flexural_strength=flexural_strength,
        strain_at_flexural_strength=strain_at_max,
    )
