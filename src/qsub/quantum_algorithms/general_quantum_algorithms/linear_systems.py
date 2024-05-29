################################################################################
# © Copyright 2023 Zapata Computing Inc.
################################################################################
import numpy as np
from qsub.subroutine_model import SubroutineModel
from typing import Union, Optional
from qsub.utils import consume_fraction_of_error_budget
from sympy import Basic, sqrt, log
from qsub.data_classes import LinearSystemBlockEncodingData, StatePreparationOracleData
from typing import Optional


class TaylorQLSA(SubroutineModel):
    def __init__(
        self,
        linear_system_block_encoding: Optional[SubroutineModel]=None,
        prepare_b_vector:Optional[SubroutineModel]=None,
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
        )
        self.linear_system_block_encoding.number_of_times_called = n_calls_to_A
        self.prepare_b_vector.number_of_times_called = n_calls_to_b

        # Set block encoding requirements
        linear_system_block_encoding_data = LinearSystemBlockEncodingData()
        state_preparation_oracle_data = StatePreparationOracleData()
        linear_system_block_encoding_data.failure_tolerance = linear_system_block_encoding_failure_tolerance
        state_preparation_oracle_data.failure_tolerance=prepare_b_vector_failure_tolerance
        self.linear_system_block_encoding.set_requirements(linear_system_block_encoding_data)
        self.prepare_b_vector.set_requirements(state_preparation_oracle_data)

    def count_qubits(self):
        # From Theorem 1 in https://arxiv.org/abs/2305.11352
        return self.linear_system_block_encoding.count_qubits() + 6


def get_taylor_qlsa_num_block_encoding_calls(
    failure_probability: float,
    subnormalization: Union[float, Basic],
    condition_number: Union[float, Basic],
) -> tuple[Union[float, Basic], Union[float, Basic]]:
    """Get the number of block encoding calls for the Taylor QLSA algorithm
    of arXiv:2305.11352 as shown in Thm. 1.

    Args:
        failure_probability: The tolerable probability of the algorithm failing due to
            approximation error in the solution state.
        subnormalization: The subnormalization value (can be a sympy object).
        condition_number: The condition number (can be a sympy object).

    Returns:
        number_of_calls_to_A: The number of block encoding calls to A (can be a float or sympy object).
        number_of_calls_to_b: The number of queries to state preparation of b (can be a float or sympy object).
    """
    if not isinstance(condition_number, Basic) and (
        failure_probability > 0.24 or condition_number < sqrt(12)
    ):
        raise ValueError(
            "failure_probability must be less than or equal to 0.24 and condition_number must be greater than or equal to sqrt(12)."
        )

    if isinstance(condition_number, float):
        term1 = (1741 * subnormalization * np.exp(1) / 500) * np.sqrt(
            condition_number**2 + 1
        )
        term2 = (
            (133 / 125) + (4 / (25 * condition_number ** (1 / 3)))
        ) * np.pi * np.log(2 * condition_number + 3) + 1
        term3 = (351 / 50) * np.log(2 * condition_number + 3) ** 2
        term4 = (
            np.log((451 * np.log(2 * condition_number + 3) ** 2) / failure_probability)
            + 1
        )
        term5 = subnormalization * condition_number * np.log(32 / failure_probability)

        Q_star = term1 * term2 + term3 * term4 + term5

        number_of_calls_to_A = Q_star / (0.39 - 0.204 * failure_probability)
        number_of_calls_to_b = 2 * number_of_calls_to_A
    else:
        term1 = (1741 * subnormalization * np.exp(1) / 500) * sqrt(
            condition_number**2 + 1
        )
        term2 = ((133 / 125) + (4 / (25 * condition_number ** (1 / 3)))) * np.pi * log(
            2 * condition_number + 3
        ) + 1
        term3 = (351 / 50) * log(2 * condition_number + 3) ** 2
        term4 = (
            log((451 * log(2 * condition_number + 3) ** 2) / failure_probability) + 1
        )
        term5 = subnormalization * condition_number * np.log(32 / failure_probability)

        Q_star = term1 * term2 + term3 * term4 + term5

        number_of_calls_to_A = Q_star / (0.39 - 0.204 * failure_probability)
        number_of_calls_to_b = 2 * number_of_calls_to_A


    return number_of_calls_to_A, number_of_calls_to_b
