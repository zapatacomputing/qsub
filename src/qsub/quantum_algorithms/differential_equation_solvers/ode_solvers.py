from qsub.subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget

from typing import Optional
import math
import warnings


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

    def count_qubits(self):
        return self.prepare_ode_history_state.count_qubits()


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
        subnormalization_of_A: float = None,
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
            n_ancilla_qubits,
        ) = get_QLSA_parameters_for_taylor_ode(
            self.requirements["evolution_time"],
            epsilon_td,
            self.requirements["subnormalization_of_A"],
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
            failure_tolerance=remaining_failure_tolerance,
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

        # Set qlsa b_vector_prep as ODEHistoryBVector()
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state = (
            self.requirements["qlsa_subroutine"]
        )

        # Set qlsa block encoding subroutine as ODEHistoryBlockEncoding
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.linear_system_block_encoding = (
            ODEHistoryBlockEncoding()
        )

        # TODO: finish this!
        # omega_L = (1 + math.sqrt(taylor_truncation + 1) + subnormalization_of_A) / (
        #     math.sqrt(taylor_truncation + 1) + 2
        # )
        # omega_L = (
        #     self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.linear_system_block_encoding.get_subnormalization()
        # )
        omega_L = 1.0
        # Set a subset of requirements for qlsa
        self.amplify_amplitude.state_preparation_oracle.prepare_ode_history_state.set_requirements(
            subnormalization=omega_L, condition_number=kappa_L
        )

    def count_qubits(self):
        return self.amplify_amplitude.count_qubits()


def get_QLSA_parameters_for_taylor_ode(
    evolution_time,
    epsilon_td,
    subnormalization_of_A,
    mu_P_A,
    norm_b,
    kappa_P,
    norm_x_t,
    A_stable,
):
    """
    Compute the parameters for the Q_QLSA function based on the theorem "Explicit query counts for ODE-solver".

    Arguments:
        evolution_time (float): Total time for the ODE solver.
        epsilon_td (float): Time discretization error.
        subnormalization_of_A (float): subnormalization of block encoding for ODE matrix.
        mu_P_A (float): A parameter related to matrix A.
        norm_b (float): Norm of vector b.
        kappa_P (float): Condition number of the preconditioner used in the QLSA.
        norm_x_t (float): Maximum norm of the vector x(t) over the interval [0, T].
        A_stable (bool): Indicates if matrix A is stable.

    Returns:
        kappa_L (float): The condition number bound.
        subnormalization_of_L (float): subnormalization of block encoding of ODE History block encoding.
        state_preparation_probability: (float) overlap squared of ODE solution state before amplitude amplification.
        n_ancilla_qubits: (int) number of ancilla qubits used for the clock and Taylor truncation
    """
    # Constants
    I_0_2 = 2.2796  # Approximation of I_0(2)
    e_constant = math.exp(1)

    # # Step 1: Time discretization error
    # epsilon_td = failure_tolerance / 8

    # Step 2: Compute the Taylor truncation
    taylor_truncation = compute_ode_taylor_truncation(
        evolution_time, epsilon_td, norm_b, norm_x_t
    )
    # x_star = max(
    #     evolution_time
    #     * e_constant**3
    #     / epsilon_td
    #     * (1 + evolution_time * e_constant**2 * norm_b / norm_x_t),
    #     10,
    # )
    # k = math.ceil(
    #     (3 * math.log(x_star) / 2 + 1) / math.log(1 + math.log(x_star) / 2) - 1
    # )

    # Step 3: Set the idling parameter p
    p = set_ode_idling_parameter(evolution_time, taylor_truncation, A_stable)
    # if A_stable:
    #     p = math.ceil(math.sqrt(evolution_time) / (k + 1)) * (k + 1)
    # else:
    #     p = math.ceil(evolution_time / (k + 1)) * (k + 1)

    # Step 4: Compute subnormalization_of_A_block_encoding
    subnormalization_of_L_block_encoding = (
        1 + math.sqrt(taylor_truncation + 1) + subnormalization_of_A
    ) / (math.sqrt(taylor_truncation + 1) + 2)

    # Step 5: Compute the upper bound on the condition number kappa_L
    g_k = sum(
        [
            (
                math.factorial(s)
                * sum([1 / math.factorial(j) for j in range(s, taylor_truncation + 1)])
                ** 2
            )
            for s in range(1, taylor_truncation + 1)
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
                + (p + evolution_time * taylor_truncation) * (I_0_2 - 1)
            )
        )
        * (math.sqrt(taylor_truncation + 1) + 2)
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

    ancilla_qubits = math.ceil(
        (math.log((evolution_time + 1) * (taylor_truncation + 1) + p, 2))
    )

    return (
        kappa_L,
        subnormalization_of_L_block_encoding,
        state_preparation_probability,
        ancilla_qubits,
    )


def compute_ode_taylor_truncation(evolution_time, epsilon_td, norm_b, norm_x_t):
    x_star = max(
        evolution_time
        * math.exp(3)
        / epsilon_td
        * (1 + evolution_time * math.exp(2) * norm_b / norm_x_t),
        10,
    )
    return math.ceil(
        (3 * math.log(x_star) / 2 + 1) / math.log(1 + math.log(x_star) / 2) - 1
    )


def set_ode_idling_parameter(evolution_time, taylor_truncation, A_stable):
    if A_stable:
        return math.ceil(math.sqrt(evolution_time) / (taylor_truncation + 1)) * (
            taylor_truncation + 1
        )
    else:
        return math.ceil(evolution_time / (taylor_truncation + 1)) * (
            taylor_truncation + 1
        )


class ODEHistoryBlockEncoding(SubroutineModel):
    def __init__(
        self,
        task_name="block_encode_ode_history_system",
        requirements=None,
        block_encode_ode_matrix: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_ode_matrix is not None:
            self.block_encode_ode_matrix = block_encode_ode_matrix
        else:
            self.block_encode_ode_matrix = SubroutineModel("block_encode_ode_matrix")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        evolution_time: float = None,
        epsilon_td: float = None,
        norm_b: float = None,
        norm_x_t: float = None,
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

        # Set number of calls to the linear term block encoding
        self.block_encode_ode_matrix.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.block_encode_ode_matrix.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        return None

    def count_qubits(self):
        # TODO: count (T+1)(k+1)+p ancilla qubits
        return self.block_encode_ode_matrix.count_qubits()


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
    warnings.warn("This function is not fully implemented.", UserWarning)
    return 1


class LBMDragEstimation(SubroutineModel):
    def __init__(
        self,
        task_name="estimate_drag_from_lbm",
        requirements=None,
        estimate_amplitude: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if estimate_amplitude is not None:
            self.estimate_amplitude = estimate_amplitude
        else:
            self.estimate_amplitude = SubroutineModel("estimate_amplitude")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        estimation_error: float = None,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        norm_b: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        solve_quantum_ode: Optional[SubroutineModel] = None,
        block_encode_drag_operator: Optional[SubroutineModel] = None,
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
        # Rather, it properly allocates requirements and subroutines
        # to the subtasks of amplitude estimation

        block_encode_drag_operator = self.requirements["block_encode_drag_operator"]
        solve_quantum_ode = self.requirements["solve_quantum_ode"]

        # Set number of calls to the amplitude estimation task to one
        self.estimate_amplitude.number_of_times_called = 1

        amplitude_estimation_error = compute_amp_est_error_from_block_encoding(
            self.requirements["estimation_error"],
            block_encode_drag_operator.get_subnormalization(),
        )

        # Set amp est requirements
        self.estimate_amplitude.set_requirements(
            estimation_error=amplitude_estimation_error,
            failure_tolerance=self.requirements["failure_tolerance"],
        )

        # Set amp est st prep subroutine as ode solver
        self.estimate_amplitude.state_preparation_oracle = solve_quantum_ode

        # Set amp est mark subspace subroutine as block_encode_drag_operator
        self.estimate_amplitude.mark_subspace = block_encode_drag_operator

        # Set final_state_prep requirements
        self.estimate_amplitude.state_preparation_oracle.set_requirements(
            evolution_time=self.requirements["evolution_time"],
            mu_P_A=self.requirements["mu_P_A"],
            kappa_P=self.requirements["kappa_P"],
            norm_b=self.requirements["norm_b"],
            norm_x_t=self.requirements["norm_x_t"],
            A_stable=self.requirements["A_stable"],
        )

    def count_qubits(self):
        return self.estimate_amplitude.count_qubits()


def compute_amp_est_error_from_block_encoding(estimation_error, failure_tolerance):
    warnings.warn("This function is not fully implemented.", UserWarning)
    amplitude_estimation_error = 0.1
    return amplitude_estimation_error

    # #########

    # remaining_failure_tolerance = self.requirements["failure_tolerance"]

    # # Allot time discretization budget
    # (
    #     truncation_error,
    #     remaining_failure_tolerance,
    # ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

    # be_costs = get_block_encoding_costs_from_carleman_requirements(truncation_error)

    # # Set number of calls to the linear term block encoding
    # self.block_encode_linear_term.number_of_times_called = be_costs

    # # Set linear term block encoding requirements
    # self.block_encode_linear_term.set_requirements(
    #     failure_tolerance=self.requirements["failure_tolerance"],
    # )

    # # Set number of calls to the quadratic term block encoding
    # self.block_encode_quadratic_term.number_of_times_called = be_costs

    # # Set quadratic term block encoding requirements
    # self.block_encode_quadratic_term.set_requirements(
    #     failure_tolerance=self.requirements["failure_tolerance"],
    # )


class LBMDragOperator(SubroutineModel):
    def __init__(
        self,
        task_name="mark_drag_operator_subspace",
        requirements=None,
        compute_boundary: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if compute_boundary is not None:
            self.compute_boundary = compute_boundary
        else:
            self.compute_boundary = SubroutineModel("compute_boundary")

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
        # Set number of calls to the quadratic term block encoding
        self.compute_boundary.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.compute_boundary.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        warnings.warn("This function is not fully implemented.", UserWarning)
        return 42


class SphereBoundaryOracle(SubroutineModel):
    def __init__(
        self,
        task_name="compute_boundary",
        requirements=None,
        quantum_adder: Optional[SubroutineModel] = None,
        quantum_comparator: Optional[SubroutineModel] = None,
        quantum_square: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if quantum_adder is not None:
            self.quantum_adder = quantum_adder
        else:
            self.quantum_adder = SubroutineModel("quantum_adder")

        if quantum_comparator is not None:
            self.quantum_comparator = quantum_comparator
        else:
            self.quantum_comparator = SubroutineModel("quantum_comparator")

        if quantum_square is not None:
            self.quantum_square = quantum_square
        else:
            self.quantum_square = SubroutineModel("quantum_square")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        radius: float = None,
        grid_spacing: float = None,
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
            quantum_square_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        (
            quantum_adder_failure_tolerance,
            quantum_comparator_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set number of calls to the quantum_adder: two for adding y^2 and z^2 to x^2
        self.quantum_adder.number_of_times_called = 2

        # Set quantum_adder requirements
        self.quantum_adder.set_requirements(
            failure_tolerance=quantum_adder_failure_tolerance,
        )

        # Set number of calls to the quantum_comparator: two for comparing x^2+y^2+z^2 to r^2
        self.quantum_adder.number_of_times_called = 1

        # Set quantum_comparator requirements
        self.quantum_adder.set_requirements(
            failure_tolerance=quantum_comparator_failure_tolerance,
        )

        # Set number of calls to the quantum_square: three for squaring x^2, y^2, and z^2
        self.quantum_square.number_of_times_called = 3

        # Set quantum_comparator requirements
        self.quantum_square.set_requirements(
            failure_tolerance=quantum_square_failure_tolerance,
        )


class ODEHistoryBVector(SubroutineModel):
    """
    Costing of the ODE History b vector according to https://arxiv.org/abs/2309.07881.
    """

    def __init__(
        self,
        task_name="prepare_ode_history_b_vector",
        requirements=None,
        prepare_inhomogeneous_term_vector: Optional[SubroutineModel] = None,
        prepare_initial_vector: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if prepare_inhomogeneous_term_vector is not None:
            self.prepare_inhomogeneous_term_vector = prepare_inhomogeneous_term_vector
        else:
            self.prepare_inhomogeneous_term_vector = SubroutineModel(
                "prepare_inhomogeneous_term_vector"
            )

        if prepare_initial_vector is not None:
            self.prepare_initial_vector = prepare_initial_vector
        else:
            self.prepare_initial_vector = SubroutineModel("prepare_initial_vector")

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
            prepare_inhomogeneous_term_vector_failure_tolerance,
            prepare_initial_vector_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set number of calls to the linear term block encoding
        self.prepare_inhomogeneous_term_vector.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.prepare_inhomogeneous_term_vector.set_requirements(
            failure_tolerance=prepare_inhomogeneous_term_vector_failure_tolerance,
        )

        # Set number of calls to the quadratic term block encoding
        self.prepare_initial_vector.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.prepare_initial_vector.set_requirements(
            failure_tolerance=prepare_initial_vector_failure_tolerance,
        )
