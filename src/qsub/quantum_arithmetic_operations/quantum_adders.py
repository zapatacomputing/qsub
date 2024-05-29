from typing import Optional
from ..subroutine_model import SubroutineModel


class GidneyAdder(SubroutineModel):
    """
    Subroutine model for the adder of https://quantum-journal.org/papers/q-2018-06-18-74/.
    """

    def __init__(
        self,
        task_name="quantum_add",
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

        # Set number of times T gate is called
        self.t_gate.number_of_times_called = 4 * self.requirements["number_of_bits"] - 4

        # Set the requirements for the T gate
        self.t_gate.set_requirements(
            failure_tolerance=t_gate_failure_tolerance
            / self.t_gate.number_of_times_called
        )

    def count_qubits(self):
        # From https://quantum-journal.org/papers/q-2018-06-18-74/
        number_of_data_qubits = self.requirements["number_of_bits"]
        number_of_ancilla_qubits = self.requirements["number_of_bits"] - 1
        return number_of_data_qubits + number_of_ancilla_qubits
