from typing import Optional
from qsub.subroutine_model import SubroutineModel
import numpy as np


class GidneyMultiplier(SubroutineModel):
    """
    Subroutine model for multiplication based on the formulas provided in the table.
    """

    def __init__(
        self,
        task_name="quantum_multiply",
        t_gate: Optional[SubroutineModel] = None,
    ):

        super().__init__(task_name)

        if t_gate is not None:
            self.t_gate = t_gate
        else:
            self.t_gate = SubroutineModel("t_gate")


    def populate_requirements_for_subroutines(self):
        # Allot all failure tolerance to T gates
        t_gate_failure_tolerance = self.requirements["failure_tolerance"]

        n_bits = self.requirements["number_of_bits_total"]
        # Set max number of bits needed to encode the square
        number_of_bits_above_decimal_place = self.requirements[
            "number_of_bits_above_decimal_place"
        ]

        # Set number of times T gate is called using the MUL_n,p T-count formula
        # TODO: double check this formula with the paper: https://arxiv.org/pdf/2312.15871.pdf
        self.t_gate.number_of_times_called = (
            4 * n_bits**2
            - 8 * n_bits
            + 8 * n_bits * number_of_bits_above_decimal_place
            - 8 * number_of_bits_above_decimal_place**2
            + 8 * number_of_bits_above_decimal_place
        )

        # Set the requirements for the T gate
        self.t_gate.set_requirements(
            failure_tolerance=t_gate_failure_tolerance
            / self.t_gate.number_of_times_called
        )

    def count_qubits(self):
        # From https://quantum-journal.org/papers/q-2018-06-18-74/
        n_bits = self.requirements["number_of_bits"]
        number_of_data_qubits = n_bits
        number_of_ancilla_qubits = 2 * n_bits - 1
        return number_of_data_qubits + number_of_ancilla_qubits
