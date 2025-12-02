import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import argrelmax, argrelmin, find_peaks
import scipy.stats as stats
from dataclasses import dataclass
from typing import Tuple

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
        
    Note:
        Power-law plasticity model: γ = τ/G + K(τ/τ_y)^n where γ is shear strain.
    """
    longitudinal_modulus: float
    transverse_modulus: float
    poisson_ratio: float
    shear_modulus: float
    tau_y: float
    K: float
    n: float

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

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/(shear_stress_array/shear_strain_array) - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2
        axial_compliance_matrix[i, 0] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array[0])**4 + (1/G - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array[0])**2

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
    probability, bins = np.histogram(flatten_orientation_profile,
                                     bins=int(maximum_fiber_misalignment/fiber_misalignment_step_size), 
                                     density=True, range=(0, maximum_fiber_misalignment))
    for i in range(1, len(probability)):
        probability[i] = probability[i]*fiber_misalignment_step_size
    
    total_probability = np.sum(probability)
    if total_probability < 0.999:
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
        axial_strain_array = (axial_strain_array * w_k_over_L_g +
                              superposition_axial_stress_array / E1 * (1 - w_k_over_L_g))

    if not find_peaks(superposition_axial_stress_array)[0].size == 0:
        compression_strength_index = find_peaks(superposition_axial_stress_array)[0][0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]

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

    # Matrix initialization
    axial_stress_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))
    axial_compliance_matrix = np.ndarray((misalignment_array.size, shear_stress_array.size))

    for i, theta in enumerate(misalignment_array):
        axial_stress_matrix[i, ...] = shear_stress_array/(np.deg2rad(theta)+shear_strain_array)
        axial_compliance_matrix[i, ...] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array)**4 + (1/(shear_stress_array/shear_strain_array) - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array)**2
        axial_compliance_matrix[i, 0] = 1/E1 + (1/E2)*(np.deg2rad(theta)+shear_strain_array[0])**4 + (1/G - 2*nu/E1)*(np.deg2rad(theta)+shear_strain_array[0])**2

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
    # ε̄_x = ε_x * (w_k/L_g) + σ_x/E_11 * (1 - w_k/L_g)
    if kink_width is not None and gauge_length is not None and gauge_length > 0:
        w_k_over_L_g = kink_width / gauge_length
        axial_strain_array = (axial_strain_array * w_k_over_L_g +
                              superposition_axial_stress_array / E1 * (1 - w_k_over_L_g))

    if not find_peaks(superposition_axial_stress_array)[0].size == 0:
        compression_strength_index = find_peaks(superposition_axial_stress_array)[0][0]
        compression_strength = superposition_axial_stress_array[compression_strength_index]
        ultimate_strain = axial_strain_array[compression_strength_index]
    else:
        compression_strength = np.max(superposition_axial_stress_array)
        ultimate_strain = axial_strain_array[np.argmax(superposition_axial_stress_array)]

    return compression_strength, ultimate_strain, superposition_axial_stress_array, axial_strain_array