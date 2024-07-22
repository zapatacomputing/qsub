import numpy as np

from qsub.subroutine_model import SubroutineModel

from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    IterativeQuantumAmplitudeEstimationCircuit,
    IterativeQuantumAmplitudeEstimationAlgorithm,
)
from qsub.quantum_algorithms.differential_equation_solvers.linear_ode_solvers import (
    TaylorQuantumODESolver,
)

from qsub.quantum_algorithms.fluid_dynamics.lattice_boltzmann import (
    LBMDragEstimation,
    LBMDragCoefficientsReflection,
    LBMLinearTermBlockEncoding,
    LBMQuadraticTermBlockEncoding,
    LBMCubicTermBlockEncoding,
)

from qsub.quantum_algorithms.differential_equation_solvers.linearization_methods import (
    CarlemanBlockEncoding,
)

import matplotlib.pyplot as plt
from qsub.utils import calculate_max_of_solution_norm

def generate_graphs(
    evolution_time=0.0001,
    failure_tolerance=1e-1,
    kappa_P=1,
    relative_estimation_error=0.1,
    n_grids =  4.096e10,
    uniform_density_deviation = 0.001,
    fluid_nodes= 5.116*10**3
):
    """Used to generate graphs in this paper https://arxiv.org/abs/2406.06323 Look at 
       Table 7 for source of numerical constants.
    """

    mu_P_A = -0.00001
    norm_inhomogeneous_term_vector = 0.0  # ODE is homogeneous
    norm_x_t = calculate_max_of_solution_norm(fluid_nodes,
        uniform_density_deviation 
    )
    A_stable = False
    number_of_spatial_grid_points = n_grids
    number_of_velocity_grid_points = 27
    x_length_in_meters = 0.1
    y_length_in_meters = 0.08
    z_length_in_meters = 0.08
    sphere_radius_in_meters = 0.005
    time_discretization_in_seconds = 5.1928e-5
    # relative_estimation_error = 0.1

    cell_volume = (
        x_length_in_meters
        * y_length_in_meters
        * z_length_in_meters
        / number_of_spatial_grid_points
    )
    cell_face_area = cell_volume ** (2 / 3)

    number_of_cells_incident_on_face = (
        2 * np.pi * sphere_radius_in_meters**2 / cell_face_area
    )
    rough_estimate_of_drag_force = (
        cell_volume / time_discretization_in_seconds
    ) * number_of_cells_incident_on_face

    linear_term_block_encoding = LBMLinearTermBlockEncoding()
    linear_term_block_encoding.set_requirements(
        number_of_spatial_grid_points=number_of_spatial_grid_points,
        number_of_velocity_grid_points=number_of_velocity_grid_points,
    )
    quadratic_term_block_encoding = LBMQuadraticTermBlockEncoding()
    quadratic_term_block_encoding.set_requirements(
        number_of_spatial_grid_points=number_of_spatial_grid_points,
        number_of_velocity_grid_points=number_of_velocity_grid_points,
    )
    cubic_term_block_encoding = LBMCubicTermBlockEncoding()
    cubic_term_block_encoding.set_requirements(
        number_of_spatial_grid_points=number_of_spatial_grid_points,
        number_of_velocity_grid_points=number_of_velocity_grid_points,
    )


    carleman_block_encoding = CarlemanBlockEncoding(
        block_encode_linear_term=linear_term_block_encoding,
        block_encode_quadratic_term=quadratic_term_block_encoding,
        block_encode_cubic_term=cubic_term_block_encoding,
    )
    carleman_block_encoding.set_requirements(
        kappa_P=kappa_P,
        mu_P_A=mu_P_A,
        A_stable=A_stable,
    )

    # Initialize Taylor QLSA
    taylor_qlsa = TaylorQLSA()

    # Initialize Taylor Quantum ODE Solver with choice of amplitude amplification
    taylor_ode = TaylorQuantumODESolver(
        amplify_amplitude=ObliviousAmplitudeAmplification(),
    )
    taylor_ode.set_requirements(
        solve_linear_system=taylor_qlsa,
        ode_matrix_block_encoding=carleman_block_encoding,
    )
    # Initialize LBM drag estimation
    mark_drag_vector = LBMDragCoefficientsReflection()

    amp_est_circuit = IterativeQuantumAmplitudeEstimationCircuit()

    amplitude_estimation_alg = IterativeQuantumAmplitudeEstimationAlgorithm(
        run_iterative_qae_circuit=amp_est_circuit
    )

    drag_est = LBMDragEstimation(estimate_amplitude=amplitude_estimation_alg)
    drag_est.set_requirements(
        evolution_time=evolution_time,
        relative_estimation_error=relative_estimation_error,
        estimated_drag_force=rough_estimate_of_drag_force,
        mu_P_A=mu_P_A,
        kappa_P=kappa_P,
        failure_tolerance=failure_tolerance,
        norm_inhomogeneous_term_vector=norm_inhomogeneous_term_vector,
        norm_x_t=norm_x_t,
        A_stable=A_stable,
        solve_quantum_ode=taylor_ode,
        number_of_spatial_grid_points=number_of_spatial_grid_points,
        number_of_velocity_grid_points=number_of_velocity_grid_points,
        x_length_in_meters=x_length_in_meters,
        y_length_in_meters=y_length_in_meters,
        z_length_in_meters=z_length_in_meters,
        sphere_radius_in_meters=sphere_radius_in_meters,
        time_discretization_in_seconds=time_discretization_in_seconds,
        mark_drag_vector=mark_drag_vector,
    )

    # Run the solver and get the query count
    drag_est.run_profile(verbose=False)
    drag_est.print_profile()

    counts = drag_est.count_subroutines()

    n_qubits= drag_est.count_qubits()

    return counts["t_gate"], n_qubits

# failure_tolerance_values = [0.001, 0.01, 0.05]
failure_tolerance_values = [0.01]
grid_points = [5120.00, 5120000.00, 40960000000.00]
fluid_nodes = [5.116*10**3,5.116*10*6, 4.093*10**10]
evolution_times = [43.33 , 240.72 , 7703.04]


resources = {}

for error in failure_tolerance_values:
    resources[error]= []
    for g, evol_time, n_fluid_nodes in zip(grid_points, evolution_times, fluid_nodes):
        tcounts, qubits = generate_graphs(n_grids=g, 
            evolution_time=evol_time, 
            failure_tolerance=error,
            fluid_nodes=n_fluid_nodes
            )
        resources[error].append((tcounts, qubits))

# Increase font sizes
plt.rcParams.update({'font.size': 12})  # General font size
plt.rcParams.update({'axes.titlesize': 12})  # Title font size
plt.rcParams.update({'axes.labelsize': 12})  # Axis labels font size
plt.rcParams.update({'xtick.labelsize': 8})  # X-tick labels font size
plt.rcParams.update({'ytick.labelsize': 12})  # Y-tick labels font size
plt.rcParams.update({'legend.fontsize': 12})  # Legend font size
# Dark colors for different tolerances
dark_colors = ['#000000', '#8B0000', '#00008B', '#006400', '#008B8B', '#8B008B', '#FF8C00', '#4B0082', '#2F4F4F', '#800000']
markers = ['x', '+', 'v']
# Tolerances and Reynolds numbers
tolerances = list(resources.keys())
reynolds_numbers = [1, 20, 500]




# Plot T-gate Counts vs Failure Tolerance
fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
ax1 = axes[0]
for tol_idx, tolerance in enumerate(tolerances):
    for reynolds_idx, reynolds in enumerate(reynolds_numbers):
        t_counts, num_qubits = resources[tolerance][reynolds_idx]
        ax1.scatter(tolerance, t_counts, color=dark_colors[reynolds_idx % len(dark_colors)], marker=markers[reynolds_idx % len(markers)], label=f'Re {reynolds}' if tol_idx == 0 else "")
ax1.set_yscale('log')
ax1.set_ylabel('$T$ gate Counts')
ax1.set_title('$T$ gate Counts vs Failure Tolerance')
ax1.grid(True)
ax1.legend(title='Legend')

# Plot Number of Qubits vs Failure Tolerance
ax2 = axes[1]
for tol_idx, tolerance in enumerate(tolerances):
    for reynolds_idx, reynolds in enumerate(reynolds_numbers):
        t_counts, num_qubits = resources[tolerance][reynolds_idx]
        ax2.scatter(tolerance, num_qubits, color=dark_colors[reynolds_idx % len(dark_colors)], marker=markers[reynolds_idx % len(markers)], label=f'Re {reynolds}' if tol_idx == 0 else "")
ax2.set_yscale('log')
ax2.set_xlabel('Failure Tolerance')
ax2.set_ylabel('Number of Qubits')
ax2.set_title('Number of Qubits vs Failure Tolerance')
ax2.grid(True)
ax2.legend(title='Legend')
plt.tight_layout()
plt.show()


# Plot
fig, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('$T$ gate counts  vs. Number of Qubits for different Reynolds numbers')

# Dark colors for different tolerances
dark_colors = ['#000000', '#00008B', '#006400', '#8B0000', '#008B8B', '#8B008B', '#FF8C00', '#4B0082', '#2F4F4F', '#800000']
markers = ['x', 'o', '^', 's', 'p', '*', 'D', 'v', '<', '>']

for i, reynolds in enumerate(reynolds_numbers):
    ax = axes2[i]
    for tol_idx, tolerance in enumerate(tolerances):
        t_counts, num_qubits = resources[tolerance][i]
        ax.scatter(num_qubits, t_counts, color=dark_colors[tol_idx], marker=markers[tol_idx], label=f'Tolerance={tolerance}')
    ax.set_title(f'Reynolds={reynolds}')
    ax.set_xlabel('Number of Qubits')
    ax.set_yscale('log')
    ax.legend(title="Failure Tolerance")
    ax.grid(True)

axes2[0].set_ylabel('$T$ gate counts')
# fig.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

fig, axes3 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
# # T counts * # of qubits vs qubits
for i, reynolds in enumerate(reynolds_numbers):
    ax = axes3[i]
    for tol_idx, tolerance in enumerate(tolerances):
        t_counts, num_qubits = resources[tolerance][i]
        product_t_counts_num_qubits = t_counts*num_qubits
        ax.scatter(num_qubits, product_t_counts_num_qubits, color=dark_colors[tol_idx], marker=markers[tol_idx], label=f'Tolerance={tolerance}')
    ax.set_title(f'Reynolds={reynolds}')
    ax.set_xlabel('Number of Qubits')
    ax.set_yscale('log')
    ax.legend(title="Failure Tolerance")
    ax.grid(True)

axes3[0].set_ylabel('$T$ counts $\\times$ Number of Logical Qubits')
# fig.legend(loc='center right', bbox_to_anchor=(1.1, 0.5))
plt.tight_layout(rect=[0, 0, 1, 0.96])# 
plt.show()

