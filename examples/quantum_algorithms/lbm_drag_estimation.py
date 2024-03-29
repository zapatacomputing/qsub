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


def generate_graphs(
    evolution_time=0.0001,
    failure_tolerance=1e-1,
    kappa_P=1,
    relative_estimation_error=0.1,
):
    # TODO: determine range of evolution_time
    # evolution_time = 0.0001  # Example value
    # failure_tolerance = 1e-1  # Example value
    # TODO: determine range of mu_P_A
    mu_P_A = -0.001
    norm_inhomogeneous_term_vector = 0.0  # ODE is homogeneous
    # TODO: determine range of norm_x_t
    norm_x_t = 1.0  # Example value
    A_stable = False
    # TODO: determine range of kappa_P
    # kappa_P = 1
    number_of_spatial_grid_points = 4.096e10
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
    cubic_term_block_encoding = LBMCubicTermBlockEncoding()

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
        quantum_comparator=quantum_comparator,
        quantum_adder=quantum_adder,
        quantum_sqrt=quantum_sqrt,
        quantum_square=quantum_square,
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
    # print(
    #     "QLSA is:",
    #     drag_est.requirements["solve_quantum_ode"].requirements["solve_linear_system"],
    # )

    # Run the solver and get the query count
    drag_est.run_profile(verbose=True)
    drag_est.print_profile()
    # drag_est.print_qubit_usage()

    # Add child subroutines to root_subroutine...
    # taylor_ode.create_tree()
    print()
    print("Tree of subtasks and subroutines:")
    drag_est.display_tree()

    counts = drag_est.count_subroutines()
    print()
    print("Counts of subtasks:")
    for key, value in counts.items():
        print(f"'{key}': {value},")

    print("qubits =", drag_est.count_qubits())

    graph = drag_est.display_hierarchy()
    graph.view(cleanup=True)  # This will open the generated diagram

    return counts["t_gate"]


low_cost = generate_graphs(
    evolution_time=0.001,
    failure_tolerance=0.1,
    kappa_P=1,
    relative_estimation_error=0.1,
)
high_cost = generate_graphs(
    evolution_time=1000,
    failure_tolerance=0.0001,
    kappa_P=1000,
    relative_estimation_error=0.001,
)

low_kappa = generate_graphs(
    evolution_time=1,
    failure_tolerance=0.1,
    kappa_P=1,
    relative_estimation_error=0.001,
)
medium_kappa = generate_graphs(
    evolution_time=1,
    failure_tolerance=0.1,
    kappa_P=100,
    relative_estimation_error=0.001,
)
high_kappa = generate_graphs(
    evolution_time=1,
    failure_tolerance=0.1,
    kappa_P=10000,
    relative_estimation_error=0.001,
)

print("'low cost' t gate count", low_cost)
print("high cost t gate count", high_cost)

print("kappa = 1 t gate count", low_kappa)
print("kappa = 100 t gate count", medium_kappa)
print("kappa = 10000 t gate count", high_kappa)
