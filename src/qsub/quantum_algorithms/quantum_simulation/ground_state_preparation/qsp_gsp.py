from typing import Optional
from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget
import math
from sympy import log


class LinTongGroundStatePreparation(SubroutineModel):
    def __init__(
        self,
        task_name="ground_state_preparation",
        requirements=None,
        hamiltonian_block_encoding: Optional[GenericBlockEncoding] = None,
        initial_state_preparation: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if hamiltonian_block_encoding is not None:
            self.hamiltonian_block_encoding = hamiltonian_block_encoding
        else:
            self.hamiltonian_block_encoding = GenericBlockEncoding(
                "hamiltonian_block_encoding"
            )

        if initial_state_preparation is not None:
            self.initial_state_preparation = initial_state_preparation
        else:
            self.initial_state_preparation = SubroutineModel(
                "initial_state_preparation"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        hamiltonian_gap: float = None,
        initial_state_overlap: float = None,
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
        num_queries = compute_number_of_gsp_block_encoding_queries(
            subnormalization,
            self.requirements["initial_state_overlap"],
            self.requirements["hamiltonian_gap"],
            consumed_failure_tolerance,
        )

        self.hamiltonian_block_encoding.number_of_times_called = num_queries

        self.hamiltonian_block_encoding.set_requirements(
            failure_tolerance=remaining_failure_tolerance
            / self.hamiltonian_block_encoding.number_of_times_called
        )

        # Populate requirements for initial_state_preparation
        self.initial_state_preparation.set_requirements(
            failure_tolerance=remaining_failure_tolerance
        )

    def count_qubits(self):
        return self.hamiltonian_block_encoding.count_qubits()


def compute_number_of_gsp_block_encoding_queries(alpha, gamma, delta, epsilon):
    """
    Compute the number of block encoding queries to UH required to prepare
    the ground state to within error epsilon from https://arxiv.org/abs/2002.12508.

    Parameters:
    alpha (float): A constant related to the Hamiltonian.
    gamma (float): The lower bound for the initial overlap.
    delta (float): The spectral gap.
    epsilon (float): The desired precision.

    Returns:
    int: The number of queries to UH.
    """

    # Calculate the number of queries with the corrected formula
    num_queries = (alpha / (gamma * delta)) * log(1 / (gamma * epsilon))

    # The number of queries should be an integer, so use ceil to round up
    return num_queries
