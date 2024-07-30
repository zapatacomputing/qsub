from typing import Optional
from ..subroutine_model import SubroutineModel


class GidneyMultiControlledZGate(SubroutineModel):
    """
    Subroutine model for the multi-controlled Z gate.
    """

    def __init__(
        self,
        task_name="multi_controlled_z_gate",
        requirements=None,
        toffoli_gate: Optional[SubroutineModel] = None,
    ):

        super().__init__(task_name, requirements)

        if toffoli_gate is not None:
            self.toffoli_gate = toffoli_gate
        else:
            self.toffoli_gate = SubroutineModel("toffoli_gate")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_controls: float = None,
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

        # Allot all failure tolerance to Toffoli gates
        toffoli_failure_tolerance = self.requirements["failure_tolerance"]

        # Set number of times Toffoli gate is called
        self.toffoli_gate.number_of_times_called = self.requirements[
            "number_of_controls"
        ]

        # Set the requirements for the Toffoli gate
        self.toffoli_gate.set_requirements(
            failure_tolerance=toffoli_failure_tolerance
            / self.toffoli_gate.number_of_times_called
        )

    def count_qubits(self):
        # From https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html
        return 2 * (self.num_controls + 1)
