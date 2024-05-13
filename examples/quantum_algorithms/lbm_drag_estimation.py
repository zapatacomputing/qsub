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

from qsub.quantum_arithmetic_operations.quantum_comparators import GidneyComparator
from qsub.quantum_arithmetic_operations.quantum_adders import GidneyAdder
from qsub.quantum_arithmetic_operations.quantum_square_roots import GidneySquareRoot
from qsub.quantum_arithmetic_operations.quantum_multipliers import GidneyMultiplier
import matplotlib.pyplot as plt

def generate_graphs(
    evolution_time=0.0001,
    failure_tolerance=1e-1,
    kappa_P=1,
    relative_estimation_error=0.1,
    n_grids =  4.096e10
):

    mu_P_A = -0.00001
    norm_inhomogeneous_term_vector = 0.0  # ODE is homogeneous

    norm_x_t = 6.86 *10**(6) 
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

    quantum_comparator = GidneyComparator()
    quantum_adder = GidneyAdder()
    quantum_sqrt = GidneySquareRoot()
    quantum_square = GidneyMultiplier()

    # Initialize LBM drag estimation
    mark_drag_vector = LBMDragCoefficientsReflection(
        # quantum_comparator=quantum_comparator,
        # quantum_adder=quantum_adder,
        # quantum_sqrt=quantum_sqrt,
        # quantum_square=quantum_square,
    )

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
    # drag_est.print_profile()
    # drag_est.print_qubit_usage()

    # Add child subroutines to root_subroutine...
    # taylor_ode.create_tree()
    # print()
    # print("Tree of subtasks and subroutines:")
    # drag_est.display_tree()

    counts = drag_est.count_subroutines()
    # print()
    # print("Counts of subtasks:")
    # for key, value in counts.items():
    #     print(f"'{key}': {value},")

    n_qubits= drag_est.count_qubits()

    # graph = drag_est.display_hierarchy()
    # graph.view(cleanup=False)  # This will open the generated diagram
    # print("counts: ", counts)
    return counts["t_gate"], n_qubits

failure_tolerance_values = [0.0001,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
grid_points = [5120.00, 5120000.00, 40960000000.00]
evolution_times = [43.33 , 240.72 , 7703.04]


resources = {}

for error in failure_tolerance_values:
    resources[error]= []
    for g, evol_time in zip(grid_points, evolution_times):
        tcounts, qubits = generate_graphs(n_grids=g, evolution_time=evol_time, failure_tolerance=error)
        resources[error].append((tcounts, qubits))

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))  # Two subplots in one column

# # Define the maximum number of tuples per failure tolerance to generate consistent color mapping
# max_tuples = max(len(values) for values in resources.values())

# # Color map for different instances
# colors = plt.cm.viridis(np.linspace(0.5, 1, max_tuples))

# # Function to plot data
# def plot_data(ax, values_function, ylabel, set_log_scale=False):
#     legend_handles = []  # Store legend handles to add only once
#     for ft, tuples in resources.items():
#         for index, (count, qubits) in enumerate(tuples):
#             color = colors[index]
#             marker = 'o' if ax == ax1 else '^'
#             # Plot and label only the first appearance of each instance across the dataset
#             label = f'Problem Instance {index + 1}' if ft == list(resources.keys())[0] else None
#             handle = ax.scatter(ft, values_function(count, qubits), color=color, marker=marker, label=label)
#             if ft == list(resources.keys())[0]:  # Add handle for legend
#                 legend_handles.append(handle)
#     ax.set_xlabel('Failure Tolerance')
#     ax.set_ylabel(ylabel)
#     ax.grid(True)
#     if set_log_scale:
#         ax.set_yscale('log')
#     return legend_handles

# # Plot T-gate counts on the first subplot
# handles1 = plot_data(ax1, lambda count, qubits: count, 'T-gate Counts', True)
# ax1.set_title('T-gate Counts vs Failure Tolerance')

# # Plot Number of Qubits on the second subplot
# handles2 = plot_data(ax2, lambda count, qubits: qubits, 'Number of Qubits', True)
# ax2.set_title('Number of Qubits vs Failure Tolerance')

# # Add a single legend for the first subplot
# fig.legend(handles1, [f'Re {i}' for i in [1, 20, 500]], title="Legend", loc='upper right', bbox_to_anchor=(0.9, 0.7))

# plt.tight_layout()
# plt.show()

reynolds_names = {1: 'Reynolds=1', 20: 'Reynolds=20', 500: 'Reynolds=500'}
print(resources)
# Creating the plots with a logarithmic scale on the y-axis
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# A common ylabel
fig.suptitle('T-counts x Number of Qubits vs. Number of Qubits for different Reynolds numbers (Log Scale)')
fig.supylabel('T-counts x Number of Qubits (log scale)')

# Creating each subplot with a log scale for the y-axis
for i, (reynolds, ax) in enumerate(zip(resources.keys(), axes.flatten()), 1):
    for data_point in resources[reynolds]:
        print(data_point)
        ax.scatter(data_point[1], data_point[0], label=f'Tolerance={data_point[2]}', alpha=0.7)
    ax.set_title(reynolds_names[reynolds])
    ax.set_xlabel('Number of Qubits')
    ax.set_yscale('log')  # Setting y-axis to log scale
    ax.legend(title="Failure Tolerance")

# Show plot
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the subplots to give space for the common ylabel
plt.show()
