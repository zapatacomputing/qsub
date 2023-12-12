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
from typing import Optional
import math


class ODEFinalTimePrep(SubroutineModel):
    def __init__(
        self,
        task_name="prepare_ode_final_time_state",
        requirements=None,
        prepare_ode_history_state: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if prepare_ode_history_state is not None:
            self.prepare_ode_history_state = prepare_ode_history_state
        else:
            self.prepare_ode_history_state = SubroutineModel(
                "prepare_ode_history_state"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        state_preparation_probability: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Note: This subroutine consumes no failure probability.
        # Rather, it properly costs the error budget consumption of
        # the qlsa subroutine that prepares the history state on account
        # of the need to post-select on the final state

        state_preparation_probability = self.requirements[
            "state_preparation_probability"
        ]
        failure_tolerance = self.requirements["failure_tolerance"]
        # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
        epsilon_ls = (
            failure_tolerance * state_preparation_probability / (2 + failure_tolerance)
        )

        # The subroutine is only called once
        self.prepare_ode_history_state.number_of_times_called = 1

        # Set prepare ode history state requirements
        self.prepare_ode_history_state.set_requirements(
            failure_tolerance=epsilon_ls,
        )


class TaylorQuantumODESolver(SubroutineModel):
    def __init__(
        self,
        task_name="solve_quantum_ode",
        requirements=None,
        amplify_amplitude: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if amplify_amplitude is not None:
            self.amplify_amplitude = amplify_amplitude
        else:
            self.amplify_amplitude = SubroutineModel("amplify_amplitude")

    def set_requirements(
        self,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        failure_tolerance: float = None,
        norm_b: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        qlsa_subroutine: SubroutineModel = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot time discretization budget
        (
            time_discretization_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
        epsilon_td = time_discretization_failure_tolerance / 4

        # Compute classical inputs
        (
            kappa_L,
            omega_L,
            state_preparation_probability,
        ) = get_QLSA_parameters_for_taylor_ode(
            self.requirements["evolution_time"],
            epsilon_td,
            self.requirements["mu_P_A"],
            self.requirements["norm_b"],
            self.requirements["kappa_P"],
            self.requirements["norm_x_t"],
            self.requirements["A_stable"],
        )

        # Set number of calls to the amplify amplitude task to one
        self.amplify_amplitude.number_of_times_called = 1

        # Set amp amp requirements
        self.amplify_amplitude.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
            input_state_squared_overlap=state_preparation_probability,
        )

        # Set amp_amp st prep subroutine as final_state_prep
        self.amplify_amplitude.state_preparation_oracle = ODEFinalTimePrep()

        # Set final_state_prep probability requirement
        self.amplify_amplitude.state_preparation_oracle.set_requirements(
            state_preparation_probability=state_preparation_probability
        )

        # Set final_state_prep subroutine as qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state = (
            self.requirements["qlsa_subroutine"]
        )

        # Set a subset of requirements for qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.set_requirements(
            subnormalization=omega_L, condition_number=kappa_L
        )


def get_QLSA_parameters_for_taylor_ode(
    evolution_time, epsilon_td, mu_P_A, norm_b, kappa_P, norm_x_t, A_stable
):
    """
    Compute the parameters for the Q_QLSA function based on the theorem "Explicit query counts for ODE-solver".

    Parameters:
    evolution_time (float): Total time for the ODE solver.
    epsilon_td (float): Time discretization error.
    mu_P_A (float): A parameter related to matrix A.
    norm_b (float): Norm of vector b.
    kappa_P (float): Condition number of the preconditioner used in the QLSA.
    norm_x_t (float): Maximum norm of the vector x(t) over the interval [0, T].
    A_stable (bool): Indicates if matrix A is stable.

    Returns:
    kappa_L (float): The condition number bound.
    subnormalization_of_A_block_encoding (float): subnormalization of block encoding for ODE matrix.
    state_preparation_probability: (ideal) overlap squared of ODE solution state before amplitude amplification.
    """
    # Constants
    I_0_2 = 2.2796  # Approximation of I_0(2)
    e_constant = math.exp(1)

    # # Step 1: Time discretization error
    # epsilon_td = failure_tolerance / 8

    # Step 2: Compute the Taylor truncation
    x_star = max(
        evolution_time
        * e_constant**3
        / epsilon_td
        * (1 + evolution_time * e_constant**2 * norm_b / norm_x_t),
        10,
    )
    k = math.ceil(
        (3 * math.log(x_star) / 2 + 1) / math.log(1 + math.log(x_star) / 2) - 1
    )

    # Step 3: Set the idling parameter p
    if A_stable:
        p = math.ceil(math.sqrt(evolution_time) / (k + 1)) * (k + 1)
    else:
        p = math.ceil(evolution_time / (k + 1)) * (k + 1)

    # Step 4: Compute subnormalization_of_A_block_encoding
    subnormalization_of_A_block_encoding = (1 + math.sqrt(k + 1) + e_constant) / (
        math.sqrt(k + 1) + 2
    )

    # Step 5: Compute the upper bound on the condition number kappa_L
    g_k = sum(
        [
            (
                math.factorial(s)
                * sum([1 / math.factorial(j) for j in range(s, k + 1)]) ** 2
            )
            for s in range(1, k + 1)
        ]
    )
    kappa_L = math.sqrt(
        (
            (1 + epsilon_td) ** 2
            * (1 + g_k)
            * kappa_P
            * (
                p
                * (1 - math.exp(2 * evolution_time * mu_P_A + 2 * mu_P_A))
                / (1 - math.exp(2 * mu_P_A))
                + I_0_2
                * (
                    math.exp(2 * mu_P_A * (evolution_time + 2))
                    + evolution_time
                    + 1
                    - math.exp(2 * mu_P_A) * (2 + evolution_time)
                )
                / ((1 - math.exp(2 * mu_P_A)) ** 2)
                + p * (p + 1) / 2
                + (p + evolution_time * k) * (I_0_2 - 1)
            )
        )
        * (math.sqrt(k + 1) + 2)
    )

    # Step 6: Compute success probabilities Pr_H and Pr_F
    K = (3 - e_constant) ** 2 if norm_b != 0 else 1
    # Pr_H = K / (K - 1 + I_0_2)
    Pr_F = 1 / (
        (1 - (I_0_2 - 1) / ((p + 1) * K))
        + (evolution_time + 1)
        * (I_0_2 - 1)
        / ((p + 1) * K)
        * ((1 + epsilon_td) / (1 - epsilon_td)) ** 2
        * e_constant**2
    )

    # # Step 7: Compute epsilon_LS
    # epsilon_LS = (
    #     failure_tolerance * Pr_F / (4 + failure_tolerance)
    # )  # Using Pr_H as the success probability

    state_preparation_probability = Pr_F

    return (
        # epsilon_LS,
        kappa_L,
        subnormalization_of_A_block_encoding,
        state_preparation_probability,
    )


class TaylorQuantumODESolver(SubroutineModel):
    def __init__(
        self,
        task_name="solve_quantum_ode",
        requirements=None,
        amplify_amplitude: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if amplify_amplitude is not None:
            self.amplify_amplitude = amplify_amplitude
        else:
            self.amplify_amplitude = SubroutineModel("amplify_amplitude")

    def set_requirements(
        self,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        failure_tolerance: float = None,
        norm_b: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        qlsa_subroutine: SubroutineModel = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot time discretization budget
        (
            time_discretization_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        # Consumption is related to discretization error as follows according to Eq. 29 of https://arxiv.org/abs/2309.07881
        epsilon_td = time_discretization_failure_tolerance / 4

        # Compute classical inputs
        (
            kappa_L,
            omega_L,
            state_preparation_probability,
        ) = get_QLSA_parameters_for_taylor_ode(
            self.requirements["evolution_time"],
            epsilon_td,
            self.requirements["mu_P_A"],
            self.requirements["norm_b"],
            self.requirements["kappa_P"],
            self.requirements["norm_x_t"],
            self.requirements["A_stable"],
        )

        # Set number of calls to the amplify amplitude task to one
        self.amplify_amplitude.number_of_times_called = 1

        # Allot amplitude amplification budget
        (
            amplitude_amplification_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set amp amp requirements
        self.amplify_amplitude.set_requirements(
            failure_tolerance=amplitude_amplification_failure_tolerance,
            input_state_squared_overlap=state_preparation_probability,
        )

        # Set amp_amp st prep subroutine as final_state_prep
        self.amplify_amplitude.state_preparation_oracle = ODEFinalTimePrep()

        # Set final_state_prep probability requirement
        self.amplify_amplitude.state_preparation_oracle.set_requirements(
            state_preparation_probability=state_preparation_probability
        )

        # Set final_state_prep subroutine as qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state = (
            self.requirements["qlsa_subroutine"]
        )

        # Set a subset of requirements for qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.set_requirements(
            subnormalization=omega_L, condition_number=kappa_L
        )


class CarlemanBlockEncoding(SubroutineModel):
    def __init__(
        self,
        task_name="block_encode_carleman_linearization",
        requirements=None,
        block_encode_linear_term: Optional[SubroutineModel] = None,
        block_encode_quadratic_term: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_linear_term is not None:
            self.block_encode_linear_term = block_encode_linear_term
        else:
            self.block_encode_linear_term = SubroutineModel("block_encode_linear_term")

        if block_encode_quadratic_term is not None:
            self.block_encode_quadratic_term = block_encode_quadratic_term
        else:
            self.block_encode_quadratic_term = SubroutineModel(
                "block_encode_quadratic_term"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot time discretization budget
        (
            truncation_error,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        be_costs = get_block_encoding_costs_from_carleman_requirements(truncation_error)

        # Set number of calls to the linear term block encoding
        self.block_encode_linear_term.number_of_times_called = be_costs

        # Set linear term block encoding requirements
        self.block_encode_linear_term.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

        # Set number of calls to the quadratic term block encoding
        self.block_encode_quadratic_term.number_of_times_called = be_costs

        # Set quadratic term block encoding requirements
        self.block_encode_quadratic_term.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )


def get_block_encoding_costs_from_carleman_requirements(truncation_error):
    # NOT IMPLEMENTED FULLY YET
    return 1 / truncation_error
