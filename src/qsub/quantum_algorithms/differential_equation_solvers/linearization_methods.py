from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget

from typing import Optional
import warnings
from sympy import symbols, Max, ceiling, log


class CarlemanBlockEncoding(SubroutineModel):
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

        if block_encode_cubic_term is not None:
            self.block_encode_cubic_term = block_encode_cubic_term
        else:
            self.block_encode_cubic_term = GenericBlockEncoding(
                "block_encode_cubic_term"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        kappa_P: float = None,
        mu_P_A: float = None,
        A_stable: bool = None,
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

        # Set number of calls to the cubic term block encoding
        self.block_encode_cubic_term.number_of_times_called = be_costs

        # Set cubic term block encoding requirements
        self.block_encode_cubic_term.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        # TODO: make these numbers variable inputs, possibly fed in through requirements
        carleman_truncation_level = 3
        ode_degree = 3
        max_block_encoding_subnormalization = Max(
            self.block_encode_linear_term.get_subnormalization(),
            self.block_encode_quadratic_term.get_subnormalization(),
            self.block_encode_cubic_term.get_subnormalization(),
        )

        return (
            ode_degree
            * carleman_truncation_level
            * (carleman_truncation_level + 1)
            * max_block_encoding_subnormalization
        ) / 2

    def count_qubits(self):
        carleman_truncation_level = 3
        ode_degree = 3
        number_of_qubits_encoding_F1 = 70
        number_of_ancillas_used_to_block_encode_F1 = 5
        number_of_qubits = (
            number_of_qubits_encoding_F1 * (carleman_truncation_level + ode_degree)
            + number_of_ancillas_used_to_block_encode_F1
            + 3 * ceiling(log(carleman_truncation_level, 2))
            + ceiling(log(ode_degree, 2))
        )
        return number_of_qubits


def get_block_encoding_costs_from_carleman_requirements(truncation_error):
    warnings.warn("This function is not fully implemented.", UserWarning)
    return 1
