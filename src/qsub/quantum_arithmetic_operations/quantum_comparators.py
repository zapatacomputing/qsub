from typing import Optional
from ..subroutine_model import SubroutineModel


class GidneyComparator(SubroutineModel):
    """
    Subroutine model for the comparator of https://quantum-journal.org/papers/q-2018-06-18-74/.
    """

    def __init__(
        self,
        task_name="quantum_compare",
        requirements=None,
        t_gate: Optional[SubroutineModel] = None,
    ):

        super().__init__(task_name, requirements)

        if t_gate is not None:
            self.t_gate = t_gate
        else:
            self.t_gate = SubroutineModel("t_gate")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_bits: float = None,
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

        # Allot all failure tolerance to T gates
        t_gate_failure_tolerance = self.requirements["failure_tolerance"]

        # Set number of times T gate is called
        self.t_gate.number_of_times_called = (
            8 * self.requirements["number_of_bits"] - 16
        )

        # Set the requirements for the T gate
        self.t_gate.set_requirements(
            failure_tolerance=t_gate_failure_tolerance
            / self.t_gate.number_of_times_called
        )

    def count_qubits(self):
        # From https://quantum-journal.org/papers/q-2018-06-18-74/
        number_of_data_qubits = self.requirements["number_of_bits"]
        number_of_ancilla_qubits = 3 * self.requirements["number_of_bits"] - 2
        return number_of_data_qubits + number_of_ancilla_qubits
