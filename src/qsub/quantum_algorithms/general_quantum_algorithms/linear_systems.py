################################################################################
# © Copyright 2023 Zapata Computing Inc.
################################################################################
import numpy as np
from ...subroutine_model import SubroutineModel
from typing import Optional
import math
from qsub.utils import consume_fraction_of_error_budget
from data_classes import LinearSystemBlockEncodingData, StatePreparationOracleData


class TaylorQLSA(SubroutineModel):
    def __init__(
        self,
        linear_system_block_encoding: SubroutineModel,
        prepare_b_vector: SubroutineModel,
        task_name="solve_quantum_linear_system",
    ):
        super().__init__(task_name)
        self.linear_system_block_encoding = linear_system_block_encoding
        self.prepare_b_vector = prepare_b_vector

    def populate_requirements_for_subroutines(self):
        # Allocate failure tolerance
        remaining_failure_tolerance = self.requirements["failure_tolerance"]
        (
            solve_linear_system_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        (
            linear_system_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        prepare_b_vector_failure_tolerance = remaining_failure_tolerance

        subnormalization = self.linear_system_block_encoding.get_subnormalization()
        condition_number = self.linear_system_block_encoding.get_condition_number()

        # Compute number of calls to block encoding of linear system and b vector prep
        (n_calls_to_A, n_calls_to_b) = get_taylor_qlsa_num_block_encoding_calls(
            solve_linear_system_failure_tolerance,
            subnormalization,
            condition_number,
            # self.requirements["condition_number"],
        )
        self.linear_system_block_encoding.number_of_times_called = n_calls_to_A
        self.prepare_b_vector.number_of_times_called = n_calls_to_b

        # Set block encoding requirements
        LinearSystemBlockEncodingData.failure_tolerance = linear_system_block_encoding_failure_tolerance
        StatePreparationOracleData.failure_tolerance=prepare_b_vector_failure_tolerance
        self.linear_system_block_encoding.set_requirements(LinearSystemBlockEncodingData)
        self.prepare_b_vector.set_requirements(StatePreparationOracleData)

    def count_qubits(self):
        # From Theorem 1 in https://arxiv.org/abs/2305.11352
        return self.linear_system_block_encoding.count_qubits() + 6


def get_taylor_qlsa_num_block_encoding_calls(
    failure_probability: float,
    subnormalization: float,
    condition_number: float,
) -> tuple[float, float]:
    """Get the number of block encoding calls for the Taylor QLSA algorithm
    of arXiv:2305.11352 as shown in Thm. 1.

    Args:
        failure_probability: The tolerable probability of the algorithm failing due to
            approximation error in the solution state.

    Returns:
        number_of_calls_to_A (float): The number of block encoding calls to A.
        number_of_calls_to_b (float): The number of queries to state preparation of b.
    """
    if failure_probability > 0.24 or condition_number < math.sqrt(12):
        raise ValueError(
            "failure_probability must be less than or equal to 0.24 and condition_number must be greater than or equal to sqrt(12)."
        )

    term1 = (1741 * subnormalization * np.exp(1) / 500) * math.sqrt(
        condition_number**2 + 1
    )
    term2 = (
        (133 / 125) + (4 / (25 * condition_number ** (1 / 3)))
    ) * math.pi * math.log(2 * condition_number + 3) + 1
    term3 = (351 / 50) * math.log(2 * condition_number + 3) ** 2
    term4 = (
        math.log((451 * math.log(2 * condition_number + 3) ** 2) / failure_probability)
        + 1
    )
    term5 = subnormalization * condition_number * math.log(32 / failure_probability)

    Q_star = term1 * term2 + term3 * term4 + term5

    number_of_calls_to_A = Q_star / (0.39 - 0.204 * failure_probability)
    number_of_calls_to_b = 2 * number_of_calls_to_A

    return (
        number_of_calls_to_A,
        number_of_calls_to_b,
    )
