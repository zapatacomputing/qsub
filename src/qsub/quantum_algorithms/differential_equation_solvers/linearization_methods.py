from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget
from dataclasses import dataclass
from typing import Optional
import warnings
from sympy import symbols, Max, ceiling, log, Basic


class CarlemanBlockEncoding(GenericBlockEncoding):
    def __init__(
        self,
        task_name="block_encode_carleman_linearization",
        requirements=None,
        block_encode_linear_term: Optional[GenericBlockEncoding] = None,
        block_encode_quadratic_term: Optional[GenericBlockEncoding] = None,
        block_encode_cubic_term: Optional[GenericBlockEncoding] = None,
    ):
        super().__init__(task_name, requirements)

        if block_encode_linear_term is not None:
            self.block_encode_linear_term = block_encode_linear_term
        else:
            self.block_encode_linear_term = GenericBlockEncoding(
                "block_encode_linear_term"
            )

        if block_encode_quadratic_term is not None:
            self.block_encode_quadratic_term = block_encode_quadratic_term
        else:
            self.block_encode_quadratic_term = GenericBlockEncoding(
                "block_encode_quadratic_term"
            )


    def populate_requirements_for_subroutines(self):
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot truncation level failure tolerance budget
        (
            truncation_error,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # TODO: may eventually want the Carleman block encoding to consume some
        # failure tolerance based on the degree that it uses (or at some point we may want
        # to allow this degree to be set by this failure rate consumption)
        (
            linear_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.3, remaining_failure_tolerance)

        (
            quadratic_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.3, remaining_failure_tolerance)

        (
            cubic_block_encoding_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.3, remaining_failure_tolerance)

        # Set number of calls to the linear term block encoding
        self.block_encode_linear_term.number_of_times_called = 1

        # Set linear term block encoding requirements
        self.block_encode_linear_term.set_requirements(
            failure_tolerance=linear_block_encoding_failure_tolerance,
        )

        # Set number of calls to the quadratic term block encoding
        self.block_encode_quadratic_term.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.block_encode_quadratic_term.set_requirements(
            failure_tolerance=quadratic_block_encoding_failure_tolerance,
        )

        # Set number of calls to the cubic term block encoding
        self.block_encode_cubic_term.number_of_times_called = 1

        # Set cubic term block encoding requirements
        self.block_encode_cubic_term.set_requirements(
            failure_tolerance=cubic_block_encoding_failure_tolerance,
        )

    def get_subnormalization(self):
        # TODO: make these numbers variable inputs, possibly fed in through requirements
        carleman_truncation_level = 3
        ode_degree = 3
        if (
            isinstance(self.block_encode_linear_term.get_subnormalization(), Basic)
            or isinstance(
                self.block_encode_quadratic_term.get_subnormalization(), Basic
            )
            or isinstance(self.block_encode_cubic_term.get_subnormalization(), Basic)
        ):
            max_block_encoding_subnormalization = Max(
                self.block_encode_linear_term.get_subnormalization(),
                self.block_encode_quadratic_term.get_subnormalization(),
                self.block_encode_cubic_term.get_subnormalization(),
            )
        else:
            max_block_encoding_subnormalization = max(
                self.block_encode_linear_term.get_subnormalization(),
                self.block_encode_quadratic_term.get_subnormalization(),
                self.block_encode_cubic_term.get_subnormalization(),
            )

        # Upper bound on subnormalization from paper (TODO: add reference)
        subnormalization = (
            ode_degree
            * carleman_truncation_level
            * (carleman_truncation_level + 1)
            * max_block_encoding_subnormalization
            / 2
        )

        return subnormalization

    def count_qubits(self):
        # TODO: update this to use the number of qubits from the block encodings
        carleman_truncation_level = 3
        ode_degree = 3
        number_of_qubits_encoding_system = (
            self.block_encode_linear_term.count_encoding_qubits()
        )

        # TODO: update this to find max over all ancilla counts
        max_number_of_ancillas_used_to_block_encode_terms = Max(
            self.block_encode_linear_term.count_block_encoding_ancilla_qubits()
        )
        number_of_qubits = (
            number_of_qubits_encoding_system * (carleman_truncation_level + ode_degree)
            + max_number_of_ancillas_used_to_block_encode_terms
            + 3 * ceiling(log(carleman_truncation_level, 2))
            + ceiling(log(ode_degree, 2))
        )
        return number_of_qubits
