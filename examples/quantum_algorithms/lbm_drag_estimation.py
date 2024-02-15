import numpy as np

from qsub.subroutine_model import SubroutineModel

from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
)
from qsub.quantum_algorithms.differential_equation_solvers.linear_ode_solvers import (
    TaylorQuantumODESolver,
)

from qsub.quantum_algorithms.fluid_dynamics.lattice_boltzmann import (
    LBMDragEstimation,
    LBMDragReflection,
    SphereBoundaryOracle,
)

from qsub.quantum_algorithms.differential_equation_solvers.linearization_methods import (
    CarlemanBlockEncoding,
)


def generate_graphs():
    evolution_time = 10000  # Example value
    drag_force = 0.0001  # Example value
    failure_tolerance = 1e-10  # Example value
    mu_P_A = -0.001
    norm_inhomogeneous_term_vector = 0.0  # Example value
    norm_x_t = 1.0  # Example value
    A_stable = False
    kappa_P = 1
    radius = 35
    grid_spacing = 10000
    estimation_error = 0.0001

    carleman_block_encoding = CarlemanBlockEncoding()
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

    sphere_oracle = SphereBoundaryOracle()
    sphere_oracle.set_requirements(radius=radius, grid_spacing=grid_spacing)
    mark_drag_vector = LBMDragReflection(compute_boundary=sphere_oracle)

    drag_est = LBMDragEstimation(estimate_amplitude=QuantumAmplitudeEstimation())
    drag_est.set_requirements(
        evolution_time=evolution_time,
        estimation_error=estimation_error,
        estimated_drag_force=drag_force,
        mu_P_A=mu_P_A,
        kappa_P=kappa_P,
        failure_tolerance=failure_tolerance,
        norm_inhomogeneous_term_vector=norm_inhomogeneous_term_vector,
        norm_x_t=norm_x_t,
        A_stable=A_stable,
        solve_quantum_ode=taylor_ode,
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


generate_graphs()
