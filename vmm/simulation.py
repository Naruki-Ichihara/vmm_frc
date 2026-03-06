import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin, find_peaks
import scipy.stats as stats
from dataclasses import dataclass
from typing import Tuple
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

    # Tangent shear modulus: G_tan = 1 / (dγ/dτ)
    d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
    tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    logger.debug(f"Discretization: n_shear_steps={len(shear_stress_array)}, n_misalignment_steps={len(misalignment_array)}")

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/tangent_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

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

    # Tangent shear modulus: G_tan = 1 / (dγ/dτ)
    d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
    tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    logger.debug(f"Discretization: n_shear_steps={len(shear_stress_array)}, n_misalignment_steps={len(misalignment_array)}")

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/tangent_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

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

    # Superposition of axial stress matrix
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

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array


def estimate_compression_strength_tangent(initial_misalignment: float,
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
    Calculate compression strength using incremental tangent modulus approach.

    Same as estimate_compression_strength but computes axial strain incrementally:
    - dε_x = dσ_x / E_x_tan (incremental strain from tangent modulus)
    - E_x_tan uses tangent shear modulus G_tan = dτ/dγ instead of secant G_sec = τ/γ
    - dσ_x = dτ / (θ + γ) (stress increment from shear stress increment)

    Args:
        initial_misalignment: Mean fiber misalignment angle in degrees.
        standard_deviation: Standard deviation of misalignment distribution in degrees.
        material_params: Complete set of material properties for the composite.
        maximum_shear_stress: Upper bound for shear stress integration (MPa).
        shear_stress_step_size: Resolution for shear stress discretization (MPa).
        maximum_axial_strain: Maximum compressive strain for analysis.
        maximum_fiber_misalignment: Upper bound for misalignment angle range (degrees).
        fiber_misalignment_step_size: Angular resolution for misalignment discretization (degrees).
        kink_width: Kink band width for gauge length correction (mm).
        gauge_length: Gauge length for strain correction (mm).

    Returns:
        Tuple containing:
        - compression_strength: Peak compressive stress (MPa)
        - ultimate_strain: Strain at peak stress
        - stress_curve: Complete stress-strain curve array
        - strain_array: Corresponding strain values

    Raises:
        ValueError: If the misalignment range doesn't capture enough probability mass (< 99.9%).
    """
    logger.info(f"Computing compression strength (tangent) with Gaussian distribution: mean={initial_misalignment:.2f}°, std={standard_deviation:.2f}°")

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

    # Tangent shear modulus: G_tan = dτ/dγ = 1 / (dγ/dτ)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
        tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    # Shear stress and strain increments
    d_tau = np.diff(shear_stress_array, prepend=0)
    d_gamma = np.diff(shear_strain_array, prepend=0)

    # Matrix initialization
    axial_stress_matrix = np.zeros((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.zeros((misalignment_array.size, shear_stress_array.size))
    axial_strain_matrix = np.zeros((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        theta_rad = np.deg2rad(theta)
        sigma_x = 0.0

        # Sequential incremental axial stress: dσ_x = (dτ - σ_x·dγ) / (θ + γ)
        for j in range(shear_stress_array.size):
            denom = theta_rad + shear_strain_array[j]
            if denom > 0:
                d_sigma = (d_tau[j] - sigma_x * d_gamma[j]) / denom
            else:
                d_sigma = 0.0
            sigma_x += d_sigma
            axial_stress_matrix[i, j] = sigma_x

        # Tangent compliance using G_tan instead of G_sec
        axial_compliance_matrix[i, :] = (1/E1
            + (1/E2) * (theta_rad + shear_strain_array)**4
            + (1/tangent_shear_modulus_array - 2*nu/E1) * (theta_rad + shear_strain_array)**2)

    # Incremental strain: dε_x = dσ_x * S_x + σ_x * dS_x (product rule of ε = σ * S)
    d_sigma_x_matrix = np.diff(axial_stress_matrix, axis=1, prepend=0)
    d_compliance_matrix = np.diff(axial_compliance_matrix, axis=1, prepend=0)
    # Use σ_x from previous step for the σ_x * dS_x term
    sigma_prev = np.zeros_like(axial_stress_matrix)
    sigma_prev[:, 1:] = axial_stress_matrix[:, :-1]
    d_epsilon_x_matrix = d_sigma_x_matrix * axial_compliance_matrix + sigma_prev * d_compliance_matrix
    axial_strain_matrix = np.cumsum(d_epsilon_x_matrix, axis=1)

    # --- DEBUG: Raw per-angle SS curves (before interpolation) ---
    if True:  # Set to False to disable
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        # Skip small angles (extreme values due to small denominator)
        min_theta_plot = 0.0  # degrees
        for i in range(1, len(misalignment_array)):
            if misalignment_array[i] < min_theta_plot:
                continue
            ax.plot(axial_strain_matrix[i, :], axial_stress_matrix[i, :],
                    linewidth=1, alpha=0.6,
                    label=f'θ={misalignment_array[i]:.2f}°')
        ax.set_xlim(0, maximum_axial_strain)
        ax.set_ylim(0, None)
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title(f'DEBUG: Tangent - Raw Per-Angle SS (before interp, θ≥{min_theta_plot}°)')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    # --- END DEBUG (raw) ---

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
        constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
        if len(maximum_argment) == 0:
            interpolation_function = interp1d(axial_strain_matrix[i, :], axial_stress_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
        elif len(minimum_argment) == 0:
            # Strain has a peak but no subsequent minimum (severe snap-back without recovery)
            # Use ascending portion for pre-peak, and map post-peak softening stress
            # to a linearly continuing strain scale
            peak_idx = maximum_argment[0]
            peak_strain = axial_strain_matrix[i, peak_idx]
            # Find stress peak index (may differ from strain peak)
            stress_peak_idx = np.argmax(axial_stress_matrix[i, :])
            # Ascending portion
            interpolation_function_left = interp1d(
                axial_strain_matrix[i, :peak_idx+1], axial_stress_matrix[i, :peak_idx+1],
                kind='linear', bounds_error=False, fill_value=(0.0, axial_stress_matrix[i, peak_idx]))
            # Post-peak: map softening stress to continuation strains
            n_post = shear_stress_array.size - stress_peak_idx
            if n_post > 1 and peak_idx > 0:
                avg_strain_step = peak_strain / peak_idx
                post_strains = peak_strain + np.arange(n_post) * avg_strain_step
                interpolation_function_right = interp1d(
                    post_strains, axial_stress_matrix[i, stress_peak_idx:stress_peak_idx+n_post],
                    kind='linear', bounds_error=False, fill_value="extrapolate")
                constant_interval_stress_array = np.zeros_like(constant_interval_strain_array)
                for j, value in enumerate(constant_interval_strain_array):
                    if value <= peak_strain:
                        constant_interval_stress_array[j] = interpolation_function_left(value)
                    else:
                        constant_interval_stress_array[j] = interpolation_function_right(value)
            else:
                constant_interval_stress_array = interpolation_function_left(constant_interval_strain_array)
        else:
            interpolation_function_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], axial_stress_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolation_function_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], axial_stress_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_stress_array = np.zeros_like(constant_interval_strain_array)
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

    # Superposition of axial stress matrix
    superposition_axial_stress_array = np.ndarray(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_axial_stress_matrix_right[1:, i]) + np.sum(weighted_axial_stress_matrix_left[1:, i])
    superposition_axial_stress_array += weighted_axial_stress_center
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    # --- DEBUG: Per-angle unweighted SS curves ---
    if True:  # Set to False to disable
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        # Center (theta=0)
        ax.plot(axial_strain_array, axial_stress_matrix[0, :],
                linewidth=1.5, alpha=0.8, label=f'θ=0°')
        # Per-angle (unweighted, raw curves)
        for i in range(1, len(misalignment_array)):
            p_sum = probabilties_right_array[i-1] + probabilties_left_array[i-1]
            if p_sum > 0.001:
                ax.plot(axial_strain_array, axial_stress_matrix[i, :],
                        linewidth=1, alpha=0.5,
                        label=f'θ={misalignment_array[i]:.2f}°')
        # Superposed (weighted sum for reference)
        ax.plot(axial_strain_array, superposition_axial_stress_array,
                linewidth=2.5, color='red', label='Superposed (weighted)')
        peak_idx = np.argmax(superposition_axial_stress_array)
        ax.plot(axial_strain_array[peak_idx], superposition_axial_stress_array[peak_idx],
                'o', markersize=8, color='red', markeredgecolor='black', zorder=5)
        ax.set_xlabel('Strain')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('DEBUG: Tangent - Per-Angle Unweighted SS Curves')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    # --- END DEBUG ---

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
        logger.info(f"Compression strength (tangent) found at peak: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]
        logger.info(f"Compression strength (tangent) at maximum: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array


def _estimate_compression_strength_secant_gtan(
    initial_misalignment: float,
    standard_deviation: float,
    material_params: MaterialParams,
    maximum_shear_stress: float = 100.0,
    shear_stress_step_size: float = 0.1,
    maximum_axial_strain: float = 0.02,
    maximum_fiber_misalignment: float = 20,
    fiber_misalignment_step_size: float = 0.1,
    kink_width: float = None,
    gauge_length: float = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Secant axial stress with tangent shear modulus in compliance (internal).

    Same as estimate_compression_strength but replaces G_sec = τ/γ with
    G_tan = dτ/dγ in the axial compliance calculation.
    Axial stress is still computed as σ_x = τ/(θ + γ) (secant formula).
    Strain is ε_x = σ_x * S_x (direct, not incremental).
    """
    logger.info(f"Computing compression strength (secant+G_tan): mean={initial_misalignment:.2f}°, std={standard_deviation:.2f}°")

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

    # Tangent shear modulus: G_tan = dτ/dγ = 1 / (dγ/dτ)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
        tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        # Secant axial stress: σ_x = τ / (θ + γ)
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        # Compliance with G_tan instead of G_sec
        axial_compliance_matrix[i, ...] = (1/E1
            + (1/E2) * (np.deg2rad(theta)+shear_strain_array)**4
            + (1/tangent_shear_modulus_array - 2*nu/E1) * (np.deg2rad(theta)+shear_strain_array)**2)

    # Replace NaN values (theta=0, gamma=0 case)
    np.nan_to_num(axial_stress_matrix[0, :], copy=False)
    axial_stress_matrix[0, 0] = np.max(axial_stress_matrix[0, :])

    axial_strain_matrix = axial_stress_matrix * axial_compliance_matrix

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
        constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
        if len(maximum_argment) == 0:
            interpolation_function = interp1d(axial_strain_matrix[i, :], axial_stress_matrix[i, :], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
        elif len(minimum_argment) == 0:
            # Strain has a peak but no subsequent minimum (severe snap-back without recovery)
            peak_idx = maximum_argment[0]
            peak_strain = axial_strain_matrix[i, peak_idx]
            stress_peak_idx = np.argmax(axial_stress_matrix[i, :])
            interpolation_function_left = interp1d(
                axial_strain_matrix[i, :peak_idx+1], axial_stress_matrix[i, :peak_idx+1],
                kind='linear', bounds_error=False, fill_value=(0.0, axial_stress_matrix[i, peak_idx]))
            n_post = shear_stress_array.size - stress_peak_idx
            if n_post > 1 and peak_idx > 0:
                avg_strain_step = peak_strain / peak_idx
                post_strains = peak_strain + np.arange(n_post) * avg_strain_step
                interpolation_function_right = interp1d(
                    post_strains, axial_stress_matrix[i, stress_peak_idx:stress_peak_idx+n_post],
                    kind='linear', bounds_error=False, fill_value="extrapolate")
                constant_interval_stress_array = np.zeros_like(constant_interval_strain_array)
                for j, value in enumerate(constant_interval_strain_array):
                    if value <= peak_strain:
                        constant_interval_stress_array[j] = interpolation_function_left(value)
                    else:
                        constant_interval_stress_array[j] = interpolation_function_right(value)
            else:
                constant_interval_stress_array = interpolation_function_left(constant_interval_strain_array)
        else:
            interpolation_function_left = interp1d(axial_strain_matrix[i, :maximum_argment[0]], axial_stress_matrix[i, :maximum_argment[0]], kind='linear', bounds_error=False, fill_value="extrapolate")
            interpolation_function_right = interp1d(axial_strain_matrix[i, minimum_argment[0]:], axial_stress_matrix[i, minimum_argment[0]:], kind='linear', bounds_error=False, fill_value="extrapolate")
            constant_interval_stress_array = np.zeros_like(constant_interval_strain_array)
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
    if total_probability < 0.999:
        raise ValueError(f"The range of fiber misalignment is too small. Total probability is {total_probability} less than 1.0.\nPlease increase the range of fiber misalignment.")

    # Weighted axial stress matrix
    weighted_axial_stress_matrix_right = np.copy(axial_stress_matrix)
    weighted_axial_stress_matrix_left = np.copy(axial_stress_matrix)

    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_right[i, :] = axial_stress_matrix[i, :]*probabilties_right_array[i-1]
    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_left[i, :] = axial_stress_matrix[i, :]*probabilties_left_array[i-1]
    weighted_axial_stress_center = axial_stress_matrix[0, :]*probabilty_center

    # Superposition of axial stress matrix
    superposition_axial_stress_array = np.ndarray(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_axial_stress_matrix_right[1:, i]) + np.sum(weighted_axial_stress_matrix_left[1:, i])
    superposition_axial_stress_array += weighted_axial_stress_center
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    # Apply kink width and gauge length correction if specified
    if kink_width is not None and gauge_length is not None and gauge_length > 0:
        w_k_over_L_g = kink_width / gauge_length
        axial_strain_array = (axial_strain_array * w_k_over_L_g +
                              superposition_axial_stress_array / E1 * (1 - w_k_over_L_g))

    if not find_peaks(superposition_axial_stress_array)[0].size == 0:
        compression_strength_index = find_peaks(superposition_axial_stress_array)[0][0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
        logger.info(f"Compression strength (secant+G_tan) found at peak: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]
        logger.info(f"Compression strength (secant+G_tan) at maximum: σ_c={compression_strength:.2f} MPa, ε_ult={ultimate_strain:.4f}")

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array


def _compute_axial_stress_response(
    initial_misalignment: float,
    standard_deviation: float,
    material_params: MaterialParams,
    maximum_shear_stress: float,
    shear_stress_step_size: float,
    maximum_axial_strain: float,
    maximum_fiber_misalignment: float,
    fiber_misalignment_step_size: float,
):
    """
    Compute weighted axial stress response from fiber misalignment distribution.

    Shared computation for kink band analysis methods. Calculates the superposition
    of axial stress contributions from fibers at different misalignment angles,
    weighted by a Gaussian probability distribution.

    Returns:
        Tuple of (axial_strain_array, superposition_axial_stress_array,
                  shear_stress_array, tangent_shear_modulus_array, Vf_2D, Ef)
    """
    if material_params.Vf is None or material_params.fiber_diameter is None:
        raise ValueError("MaterialParams must include Vf and fiber_diameter for kink band analysis.")

    Vf = material_params.Vf
    fiber_diameter = material_params.fiber_diameter
    E1 = material_params.longitudinal_modulus
    E2 = material_params.transverse_modulus
    nu = material_params.poisson_ratio
    G = material_params.shear_modulus
    tau_y = material_params.tau_y
    K = material_params.K
    n = material_params.n

    # Calculate derived parameters
    # 2D volume fraction from hexagonal packing
    tm = fiber_diameter * (np.sqrt(np.pi / (2 * np.sqrt(3) * Vf)) - 1)
    Vf_2D = fiber_diameter / (fiber_diameter + tm)
    # Fiber modulus
    Ef = E1 / Vf

    logger.debug(f"Derived params: Vf_2D={Vf_2D:.4f}, Ef={Ef:.2f} MPa")

    # Calculate shear stress and strain arrays
    shear_stress_array = np.linspace(0, maximum_shear_stress, int(maximum_shear_stress/shear_stress_step_size)+1)
    shear_strain_array = (shear_stress_array/G) + K*(shear_stress_array/tau_y)**n
    misalignment_array = np.linspace(0, maximum_fiber_misalignment, int(maximum_fiber_misalignment/fiber_misalignment_step_size)+1)

    # Calculate tangent shear modulus: G' = dτ/dγ
    with np.errstate(divide='ignore', invalid='ignore'):
        d_gamma_d_tau = 1/G + K * n / tau_y * (shear_stress_array / tau_y) ** (n - 1)
        tangent_shear_modulus_array = 1 / d_gamma_d_tau
    tangent_shear_modulus_array[0] = G

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/tangent_shear_modulus_array - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2

    # Replace NaN values
    np.nan_to_num(axial_stress_matrix[0, :], copy=False)
    axial_stress_matrix[0, 0] = np.max(axial_stress_matrix[0, :])

    axial_strain_matrix = axial_stress_matrix * axial_compliance_matrix

    # Replace constant intervals of strain array with interpolated values
    maximum_axial_stress_value = np.max(axial_stress_matrix[0, :])
    maximum_axial_strain_value = axial_strain_matrix[0, np.argmax(axial_stress_matrix[0, :])]
    interpolation_function = interp1d(np.array([0, maximum_axial_strain_value]), np.array([0, maximum_axial_stress_value]), kind='linear', bounds_error=False, fill_value="extrapolate")
    constant_interval_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)
    constant_interval_stress_array = interpolation_function(constant_interval_strain_array)
    axial_strain_matrix[0, :] = constant_interval_strain_array
    axial_stress_matrix[0, :] = constant_interval_stress_array

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

    # Probability distribution of fiber misalignment (Gaussian)
    mean_value = np.deg2rad(initial_misalignment)
    std_value = np.deg2rad(standard_deviation)

    probabilties_right_array = []
    for i in range(1, len(misalignment_array)):
        probabilty = stats.norm.cdf(np.deg2rad(misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)\
                   - stats.norm.cdf(np.deg2rad(misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        probabilties_right_array.append(probabilty)
    probabilties_right_array = np.array(probabilties_right_array)

    probabilties_left_array = []
    for i in range(1, len(misalignment_array)):
        probabilty = stats.norm.cdf(np.deg2rad(-misalignment_array[i]+fiber_misalignment_step_size/2), mean_value, std_value)\
                   - stats.norm.cdf(np.deg2rad(-misalignment_array[i]-fiber_misalignment_step_size/2), mean_value, std_value)
        probabilties_left_array.append(probabilty)
    probabilties_left_array = np.array(probabilties_left_array)

    probabilty_center = stats.norm.cdf(np.deg2rad(fiber_misalignment_step_size/2), mean_value, std_value)\
                        - stats.norm.cdf(np.deg2rad(-fiber_misalignment_step_size/2), mean_value, std_value)

    total_probability = np.sum(probabilties_right_array) + np.sum(probabilties_left_array) + probabilty_center
    if total_probability < 0.999:
        raise ValueError(f"The range of fiber misalignment is too small. Total probability is {total_probability}.")

    # Weighted axial stress
    weighted_axial_stress_matrix_right = np.copy(axial_stress_matrix)
    weighted_axial_stress_matrix_left = np.copy(axial_stress_matrix)

    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_right[i, :] = axial_stress_matrix[i, :]*probabilties_right_array[i-1]
    for i in range(1, len(misalignment_array)):
        weighted_axial_stress_matrix_left[i, :] = axial_stress_matrix[i, :]*probabilties_left_array[i-1]
    weighted_axial_stress_center = axial_stress_matrix[0, :]*probabilty_center

    superposition_axial_stress_array = np.ndarray(axial_stress_matrix.shape[1])
    for i in range(axial_stress_matrix.shape[1]):
        superposition_axial_stress_array[i] = np.sum(weighted_axial_stress_matrix_right[1:, i]) + np.sum(weighted_axial_stress_matrix_left[1:, i])
    superposition_axial_stress_array += weighted_axial_stress_center
    axial_strain_array = np.linspace(0, maximum_axial_strain, shear_stress_array.size)

    return (axial_strain_array, superposition_axial_stress_array,
            shear_stress_array, tangent_shear_modulus_array, Vf_2D, Ef)


@dataclass
class KinkBandResult:
    """
    Container for kink band failure analysis results.

    Attributes:
        critical_stress_array: Critical stress σ_cr at tangent kink width (MPa).
        axial_stress_array: Axial compressive stress σ_x (MPa).
        axial_strain_array: Axial strain values.
        shear_stress_array: Shear stress values (MPa).
        tangent_shear_modulus_array: Tangent shear modulus G'_tan (MPa).
        tangent_kink_width: Minimum kink width where σ_cr and σ_x are tangent.
        strain_at_tangent: Axial strain at the tangent point.
        strain_at_max_stress: Axial strain at maximum axial stress.
        max_axial_stress: Maximum axial stress (MPa).
        tangent_shear_stress: Shear stress at tangent point (MPa).
        kink_width_at_max_stress: Kink width where σ_cr passes through max axial stress point.
        shear_stress_at_max_stress: Shear stress at maximum axial stress point (MPa).
    """
    critical_stress_array: np.ndarray
    axial_stress_array: np.ndarray
    axial_strain_array: np.ndarray
    shear_stress_array: np.ndarray
    tangent_shear_modulus_array: np.ndarray
    tangent_kink_width: float
    strain_at_tangent: float
    strain_at_max_stress: float
    max_axial_stress: float
    tangent_shear_stress: float
    kink_width_at_max_stress: float
    shear_stress_at_max_stress: float


def estimate_kink_band_width(initial_misalignment: float,
                                standard_deviation: float,
                                material_params: MaterialParams,
                                maximum_shear_stress: float = 100.0,
                                shear_stress_step_size: float = 0.1,
                                maximum_axial_strain: float = 0.02,
                                maximum_fiber_misalignment: float = 20,
                                fiber_misalignment_step_size: float = 0.1) -> KinkBandResult:
    """
    Estimate kink band failure parameters including minimum kink width.

    Calculates the tangent kink width where critical stress σ_cr and axial stress σ_x
    curves are tangent (touch at exactly one point). Uses the formula:
    σ̄_cr = Ḡ'₁₂ + (π² V_f^2D E_f / 12) * (d_f / w)²

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
        KinkBandResult containing all analysis results.

    Raises:
        ValueError: If material_params is missing Vf or fiber_diameter.
        ValueError: If tangent point cannot be found.
    """
    logger.info(f"Computing kink band failure: mean={initial_misalignment:.2f}°, std={standard_deviation:.2f}°")

    (axial_strain_array, superposition_axial_stress_array,
     shear_stress_array, tangent_shear_modulus_array, Vf_2D, Ef) = _compute_axial_stress_response(
        initial_misalignment, standard_deviation, material_params,
        maximum_shear_stress, shear_stress_step_size, maximum_axial_strain,
        maximum_fiber_misalignment, fiber_misalignment_step_size)

    fiber_diameter = material_params.fiber_diameter

    # Find tangent kink width
    # Condition: dG'₁₂/dτ = dσ_x/dτ (slopes equal)
    dG12_dtau = np.gradient(tangent_shear_modulus_array, shear_stress_array)
    dsigma_x_dtau = np.gradient(superposition_axial_stress_array, shear_stress_array)
    slope_diff = dG12_dtau - dsigma_x_dtau

    sign_changes = np.where(np.diff(np.sign(slope_diff)))[0]
    if len(sign_changes) == 0:
        raise ValueError("No tangent point found. Curves may not intersect properly.")

    idx = sign_changes[0]
    t = -slope_diff[idx] / (slope_diff[idx + 1] - slope_diff[idx])
    tangent_index = idx if t < 0.5 else idx + 1

    # Calculate tangent kink width from: σ_cr(τ*) = σ_x(τ*)
    coefficient = (np.pi**2 * Vf_2D * Ef) / 12
    stress_diff = superposition_axial_stress_array[tangent_index] - tangent_shear_modulus_array[tangent_index]

    if stress_diff <= 0:
        raise ValueError("σ_x must be greater than G'₁₂ at tangent point for valid kink width.")

    tangent_kink_width = fiber_diameter * np.sqrt(coefficient / stress_diff)

    # Calculate critical stress array at tangent kink width
    critical_stress_array = tangent_shear_modulus_array + coefficient * (fiber_diameter / tangent_kink_width) ** 2

    # Find strain at maximum axial stress
    max_stress_index = np.argmax(superposition_axial_stress_array)
    strain_at_max_stress = axial_strain_array[max_stress_index]
    max_axial_stress = superposition_axial_stress_array[max_stress_index]
    shear_stress_at_max_stress = shear_stress_array[max_stress_index]

    strain_at_tangent = axial_strain_array[tangent_index]
    tangent_shear_stress = shear_stress_array[tangent_index]

    # Calculate kink width where σ_cr passes through max stress point
    # At max stress: σ_cr = G'_tan + coeff * (df/w)² = max_stress
    # Solving: w = df * sqrt(coeff / (max_stress - G'_tan_at_max))
    G_tan_at_max = tangent_shear_modulus_array[max_stress_index]
    stress_diff_at_max = max_axial_stress - G_tan_at_max

    if stress_diff_at_max > 0:
        kink_width_at_max_stress = fiber_diameter * np.sqrt(coefficient / stress_diff_at_max)
    else:
        # max_stress <= G'_tan at max point: no valid kink width
        kink_width_at_max_stress = np.inf

    logger.info(f"Kink band analysis complete: w_tangent={tangent_kink_width:.6f}, "
                f"w_at_max={kink_width_at_max_stress:.6f}, "
                f"ε_tangent={strain_at_tangent:.6f}, ε_max={strain_at_max_stress:.6f}")

    return KinkBandResult(
        critical_stress_array=critical_stress_array,
        axial_stress_array=superposition_axial_stress_array,
        axial_strain_array=axial_strain_array,
        shear_stress_array=shear_stress_array,
        tangent_shear_modulus_array=tangent_shear_modulus_array,
        tangent_kink_width=tangent_kink_width,
        strain_at_tangent=strain_at_tangent,
        strain_at_max_stress=strain_at_max_stress,
        max_axial_stress=max_axial_stress,
        tangent_shear_stress=tangent_shear_stress,
        kink_width_at_max_stress=kink_width_at_max_stress,
        shear_stress_at_max_stress=shear_stress_at_max_stress
    )
