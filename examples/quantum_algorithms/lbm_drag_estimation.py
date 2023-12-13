import numpy as np
import matplotlib.pyplot as plt


from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
)
from qsub.quantum_algorithms.differential_equation_solvers.ode_solvers import (
    TaylorQuantumODESolver,
    CarlemanBlockEncoding,
    LBMDragEstimation,
    LBMDragOperator,
    SphereBoundaryOracle,
    ODEHistoryBVector,
)


def generate_graphs():
    evolution_time = 10000  # Example value
    failure_tolerance = 1e-10  # Example value
    mu_P_A = -0.001
    norm_b = 0.0  # Example value
    norm_x_t = 1.0  # Example value
    A_stable = True
    kappa_P = 1
    radius = 35
    grid_spacing = 10000
    estimation_error = 0.0001

    b_vector = ODEHistoryBVector()
    linear_system_block_encoding = CarlemanBlockEncoding()
    taylor_qlsa = TaylorQLSA(
        linear_system_block_encoding=linear_system_block_encoding,
        prepare_b_vector=b_vector,
    )
    # Initialize Taylor Quantum ODE Solver with your actual implementation
    taylor_ode = TaylorQuantumODESolver(
        amplify_amplitude=ObliviousAmplitudeAmplification(),
    )
    taylor_ode.set_requirements(qlsa_subroutine=taylor_qlsa)

    sphere_oracle = SphereBoundaryOracle()
    sphere_oracle.set_requirements(radius=radius, grid_spacing=grid_spacing)
    block_encode_drag_operator = LBMDragOperator(compute_boundary=sphere_oracle)

    drag_est = LBMDragEstimation(estimate_amplitude=QuantumAmplitudeEstimation())
    drag_est.set_requirements(
        evolution_time=evolution_time,
        estimation_error=estimation_error,
        mu_P_A=mu_P_A,
        kappa_P=kappa_P,
        failure_tolerance=failure_tolerance,
        norm_b=norm_b,
        norm_x_t=norm_x_t,
        A_stable=A_stable,
        solve_quantum_ode=taylor_ode,
        block_encode_drag_operator=block_encode_drag_operator,
    )

    # Run the solver and get the query count
    drag_est.run_profile()
    drag_est.print_profile()

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

    graph = drag_est.display_hierarchy()
    graph.view()  # This will open the generated diagram

    # Add child subroutines to drag_est...
    # drag_est.plot_graph()


generate_graphs()
