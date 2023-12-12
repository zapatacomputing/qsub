import numpy as np
import matplotlib.pyplot as plt
from qsub.subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget


from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
    get_taylor_qlsa_num_block_encoding_calls,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
    compute_number_of_grover_iterates_for_obl_amp,
)
from qsub.quantum_algorithms.differential_equation_solvers.ode_solvers import (
    TaylorQuantumODESolver,
    get_QLSA_parameters_for_taylor_ode,
)
from typing import Optional
import math


def make_psi_quantum_plot():
    # Define the range for kappa
    kappa_values = np.linspace(
        100, 10000000, 50
    )  # Kappa starts from sqrt(12) as per the theorem
    epsilon = 1e-10  # Given epsilon value
    alpha = 1  # Given alpha value

    # Compute Q* for each kappa
    Q_star_values = [
        get_taylor_qlsa_num_block_encoding_calls(epsilon, alpha, kappa) / kappa
        for kappa in kappa_values
    ]

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.plot(kappa_values, Q_star_values, label="Our QLSA Algorithm", color="blue")

    plt.title("Query Count of Our QLSA Algorithm in Units of Kappa")
    plt.xlabel("Kappa")
    plt.ylabel("Query Count per kappa")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


make_psi_quantum_plot()


def cost_out_taylor_ode(
    evolution_time,
    total_failure_tolerance,
    mu_P_A,
    norm_b,
    kappa_P,
    norm_x_t,
    A_stable,
):
    # Cost out algorithm without using SubroutineModel

    # Error budgeting
    remaining_failure_tolerance = total_failure_tolerance

    # Allot time discretization budget
    (
        time_discretization_failure_tolerance,
        remaining_failure_tolerance,
    ) = consume_fraction_of_error_budget(0.25, remaining_failure_tolerance)
    # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
    epsilon_td = time_discretization_failure_tolerance / 4

    # Allot amplitude amplification budget
    (
        amplification_failure_tolerance,
        remaining_failure_tolerance,
    ) = consume_fraction_of_error_budget(0.75, remaining_failure_tolerance)

    (
        kappa_L,
        omega_L,
        state_preparation_probability,
    ) = get_QLSA_parameters_for_taylor_ode(
        evolution_time,
        epsilon_td,
        mu_P_A,
        norm_b,
        kappa_P,
        norm_x_t,
        A_stable,
    )

    # Allot quantum linear system algorithm budget
    (
        qlsa_failure_tolerance,
        remaining_failure_tolerance,
    ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
    # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
    epsilon_ls = (
        qlsa_failure_tolerance
        * state_preparation_probability
        / (2 + qlsa_failure_tolerance)
    )

    number_of_grover_iterates = compute_number_of_grover_iterates_for_obl_amp(
        amplification_failure_tolerance, state_preparation_probability
    )

    number_of_qlsa_be_calls = get_taylor_qlsa_num_block_encoding_calls(
        epsilon_ls, omega_L, kappa_L
    )

    number_of_qlsa_sp_calls = 2 * number_of_qlsa_be_calls

    total_number_of_qlsa_be_calls = number_of_grover_iterates * number_of_qlsa_be_calls
    total_number_of_qlsa_sp_calls = number_of_grover_iterates * number_of_qlsa_sp_calls

    return total_number_of_qlsa_be_calls, total_number_of_qlsa_sp_calls
