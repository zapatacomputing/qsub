import numpy as np
from qsub.subroutine_model import SubroutineModel
from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    IterativeQuantumAmplitudeEstimationAlgorithm,
    IterativeQuantumAmplitudeEstimationCircuit
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
from qsub.data_classes import (TaylorQLSAData, 
    LBMLinearTermBlockEncodingData,
    LBMQuadraticTermBlockEncodingData,
    LBMCubicTermBlockEncodingData,
    LBMDragEstimationData,
    CarlemanBlockEncodingData,
    TaylorQuantumODESolverData,
    LBMDragCoefficientsReflectionData
)

def generate_graphs(
    evolution_time=0.0001,
    failure_tolerance=1e-1,
    kappa_P=1,
    relative_estimation_error=0.1,
    n_grids =  4.096e10
):
   # ------------------------- Calculate or set Data ------------------------------------
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
    
    linear_term_block_encoding_data = LBMLinearTermBlockEncodingData()
    linear_term_block_encoding_data.number_of_spatial_grid_points = number_of_spatial_grid_points
    linear_term_block_encoding_data.number_of_velocity_grid_points = number_of_velocity_grid_points

    quadratic_term_block_encoding_data = LBMQuadraticTermBlockEncodingData()
    quadratic_term_block_encoding_data.number_of_spatial_grid_points = number_of_spatial_grid_points
    quadratic_term_block_encoding_data.number_of_velocity_grid_points = number_of_velocity_grid_points

    cubic_term_block_encoding_data = LBMCubicTermBlockEncodingData()
    cubic_term_block_encoding_data.number_of_spatial_grid_points = number_of_spatial_grid_points
    cubic_term_block_encoding_data.number_of_velocity_grid_points = number_of_velocity_grid_points

    # -------------------- set requirements for collision terms -------------------------------
    linear_term_block_encoding = LBMLinearTermBlockEncoding()
    linear_term_block_encoding.set_requirements(linear_term_block_encoding_data)
    quadratic_term_block_encoding = LBMQuadraticTermBlockEncoding()
    quadratic_term_block_encoding.set_requirements(quadratic_term_block_encoding_data)

    cubic_term_block_encoding = LBMCubicTermBlockEncoding()
    cubic_term_block_encoding.set_requirements(cubic_term_block_encoding_data)


    carleman_block_encoding = CarlemanBlockEncoding(
        block_encode_linear_term=linear_term_block_encoding,
        block_encode_quadratic_term=quadratic_term_block_encoding,
        block_encode_cubic_term=cubic_term_block_encoding,
    )
    # ----------------- set requirement for Carleman block encoding -------------------------
    carleman_block_encoding_data = CarlemanBlockEncodingData()
    carleman_block_encoding_data.A_stable = A_stable
    carleman_block_encoding_data.kappa_P = kappa_P
    carleman_block_encoding_data.mu_P_A = mu_P_A
    carleman_block_encoding.set_requirements(carleman_block_encoding_data)

    #--------------------- initialize and set requirements for Taylor QuantumODESolver -----------------------
    taylor_qlsa = TaylorQLSA()

    # Initialize Taylor Quantum ODE Solver with choice of amplitude amplification
    taylor_ode_data = TaylorQuantumODESolverData()
    taylor_ode_data.solve_linear_system =taylor_qlsa
    taylor_ode_data.ode_matrix_block_encoding= carleman_block_encoding
    taylor_ode = TaylorQuantumODESolver(
        amplify_amplitude=ObliviousAmplitudeAmplification(),
    )
    taylor_ode.set_requirements(taylor_ode_data)

    # Initialize LBM drag estimation
    mark_drag_vector = LBMDragCoefficientsReflection()
    amp_est_circuit = IterativeQuantumAmplitudeEstimationCircuit()
    amplitude_estimation_alg = IterativeQuantumAmplitudeEstimationAlgorithm(
        run_iterative_qae_circuit=amp_est_circuit
    )

    drag_est_data = LBMDragEstimationData()
    drag_est_data.evolution_time = evolution_time
    drag_est_data.relative_estimation_error = relative_estimation_error
    drag_est_data.estimated_drag_force = rough_estimate_of_drag_force
    drag_est_data.mu_P_A = mu_P_A
    drag_est_data.kappa_P = kappa_P
    drag_est_data.failure_tolerance= failure_tolerance
    drag_est_data.norm_inhomogeneous_term_vector = norm_inhomogeneous_term_vector
    drag_est_data.norm_x_t = norm_x_t
    drag_est_data.A_stable = A_stable
    drag_est_data.solve_quantum_ode = taylor_ode
    drag_est_data.number_of_spatial_grid_points = number_of_spatial_grid_points
    drag_est_data.number_of_velocity_grid_points = number_of_velocity_grid_points
    drag_est_data.x_length_in_meters = x_length_in_meters
    drag_est_data.y_length_in_meters = y_length_in_meters
    drag_est_data.z_length_in_meters = z_length_in_meters
    drag_est_data.sphere_radius_in_meters=sphere_radius_in_meters
    drag_est_data.time_discretization_in_seconds = time_discretization_in_seconds
    drag_est_data.mark_drag_vector = mark_drag_vector

    drag_est = LBMDragEstimation(estimate_amplitude=amplitude_estimation_alg)
    drag_est.set_requirements(drag_est_data)

    # Run the solver and get the query count
    drag_est.run_profile(verbose=True)
    # drag_est.print_profile()
    # drag_est.print_qubit_usage()

    # Add child subroutines to root_subroutine...
    # taylor_ode.create_tree()
    # print()
    # print("Tree of subtasks and subroutines:")
    # drag_est.display_tree()
    counts = drag_est.count_subroutines()
    n_qubits= drag_est.count_qubits()
    # graph = drag_est.display_hierarchy()
    # graph.view(cleanup=False)  # This will open the generated diagram

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

# # print(resources)
# # Extracting the data for plotting
# failure_tolerance = list(resources.keys())
# tgate_counts = [list(zip(*resources[ft]))[0] for ft in failure_tolerance]
# qubits_counts = [list(zip(*resources[ft]))[1] for ft in failure_tolerance]

# # Flatten the lists for plotting
# ft_values = []
# tgate_values = []
# qubits_values = []
# reynolds_numbers = [1, 20, 500]

# for i, ft in enumerate(failure_tolerance):
#     for j in range(len(reynolds_numbers)):
#         ft_values.append(ft)
#         tgate_values.append(tgate_counts[i][j])
#         qubits_values.append(qubits_counts[i][j])

# # Plotting the graphs
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# # T-gate Counts vs Failure Tolerance
# colors = ['black', 'red', 'blue']
# markers = ['x', '+', '1']
# for i, re in enumerate(reynolds_numbers):
#     ax1.scatter([ft_values[j] for j in range(len(ft_values)) if j % 3 == i],
#                 [tgate_values[j] for j in range(len(tgate_values)) if j % 3 == i],
#                 label=f'Re {re}', color=colors[i], marker=markers[i])
# ax1.set_yscale('log')
# ax1.set_xlabel('Failure Tolerance')
# ax1.set_ylabel('T-gate Counts')
# ax1.set_title('T-gate Counts vs Failure Tolerance')
# ax1.legend(title='Legend')
# ax1.grid(True)  # Add grid

# # Number of Qubits vs Failure Tolerance
# for i, re in enumerate(reynolds_numbers):
#     ax2.scatter([ft_values[j] for j in range(len(ft_values)) if j % 3 == i],
#                 [qubits_values[j] for j in range(len(qubits_values)) if j % 3 == i],
#                 label=f'Re {re}', color=colors[i], marker=markers[i])
# ax2.set_yscale('log')
# ax2.set_xlabel('Failure Tolerance')
# ax2.set_ylabel('Number of Qubits')
# ax2.set_title('Number of Qubits vs Failure Tolerance')
# ax2.legend(title='Legend')
# ax2.grid(True)  # Add grid

# plt.tight_layout()
# plt.show()





# for i, ft in enumerate(failure_tolerance):
#     for j in range(len(reynolds_numbers)):
#         ft_values.append(ft)
#         tgate_values.append(tgate_counts[i][j])
#         qubits_values.append(qubits_counts[i][j])

# # Setting up colors and markers for the plot
# colors = ['black', 'red', 'blue', 'green', 'purple', 'brown', 'orange']
# markers = ['x', 'x', 'x', 'x', 'x', 'x', 'x']

# # Plotting the graphs
# fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# # Titles for each subplot
# titles = ['Reynolds=1', 'Reynolds=20', 'Reynolds=500']

# for i, ax in enumerate(axs):
#     re_index = i
#     for j, ft in enumerate(failure_tolerance):
#         ax.scatter(qubits_counts[j][re_index], tgate_counts[j][re_index],
#                    color=colors[j], marker=markers[j], label=f'Tolerance={ft}' if i == 0 else "")
#     ax.set_xscale('linear')
#     ax.set_yscale('log')
#     ax.set_xlabel('Number of Qubits')
#     ax.set_title(titles[i])
#     ax.grid(True)

# axs[0].set_ylabel('T-counts x Number of Qubits')
# axs[0].legend(title='Failure Tolerance', bbox_to_anchor=(1.00, 1), loc='upper right')

# fig.suptitle('T-counts * Number of Qubits vs. Number of Qubits for different Reynolds numbers')
# plt.tight_layout()
# plt.show()

# # Creating each subplot with a log scale for the y-axis
# for i, (reynolds, ax) in enumerate(zip(resources.keys(), axes.flatten()), 1):
#     for data_point in resources[reynolds]:
#         print(data_point)
#         ax.scatter(data_point[1], data_point[0], label=f'Tolerance={data_point[2]}', alpha=0.7)
#     ax.set_title(reynolds_names[reynolds])
#     ax.set_xlabel('Number of Qubits')
#     ax.set_yscale('log')  # Setting y-axis to log scale
#     ax.legend(title="Failure Tolerance")

# # Show plot
# plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust the subplots to give space for the common ylabel
# # plt.show()
