import numpy as np
from typing import Optional
from ...subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding

from qsub.utils import consume_fraction_of_error_budget
import sympy as sp


class QSPTimeEvolution(SubroutineModel):
    def __init__(
        self,
        task_name="time_evolution",
        requirements=None,
        hamiltonian_block_encoding: Optional[GenericBlockEncoding] = None,
    ):
        super().__init__(task_name, requirements)

        if hamiltonian_block_encoding is not None:
            self.hamiltonian_block_encoding = hamiltonian_block_encoding
        else:
            self.hamiltonian_block_encoding = GenericBlockEncoding(
                "hamiltonian_block_encoding"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        evolution_time: float = None,
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
        # Populate requirements for hamiltonian_block_encoding

        # Allocate failure tolerance
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        (consumed_failure_tolerance, remaining_failure_tolerance) = (
            consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        )

        # Get subnormalization from hamiltonian_block_encoding
        subnormalization = self.hamiltonian_block_encoding.get_subnormalization()

        # Compute number of calls to hamiltonian_block_encoding,
        # spending the consumed failure tolerance
        tau = subnormalization * self.requirements["evolution_time"]
        error = consumed_failure_tolerance
        jacobi_order = compute_jacobi_anger_order(tau, error)

        self.hamiltonian_block_encoding.number_of_times_called = jacobi_order

        self.hamiltonian_block_encoding.set_requirements(
            failure_tolerance=remaining_failure_tolerance
            / self.hamiltonian_block_encoding.number_of_times_called
        )

    def count_qubits(self):
        return self.hamiltonian_block_encoding.count_qubits()


def compute_jacobi_anger_order(tau, error):
    return 1.4 * sp.Abs(tau) + sp.log(1.0 / error)
