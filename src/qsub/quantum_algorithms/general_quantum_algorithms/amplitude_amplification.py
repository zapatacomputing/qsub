from typing import Optional
import numpy as np
from qsub.subroutine_model import SubroutineModel

class ObliviousAmplitudeAmplification(SubroutineModel):
    def __init__(
        self,
        task_name="amplify_amplitude",
        requirements=None,
        state_preparation_oracle: Optional[SubroutineModel] = None,
        mark_subspace: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if state_preparation_oracle is not None:
            self.state_preparation_oracle = state_preparation_oracle
        else:
            self.state_preparation_oracle = SubroutineModel("state_preparation_oracle")

        if mark_subspace is not None:
            self.mark_subspace = mark_subspace
        else:
            self.mark_subspace = SubroutineModel("mark_subspace")

    def set_requirements(
        self,
        input_state_squared_overlap: float = None,
        failure_tolerance: float = None,
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
        # Populate requirements for state_preparation_oracle and mark_subspace

        # Allocate failure tolerance
        allocation = 0.5
        consumed_failure_tolerance = allocation * self.requirements["failure_tolerance"]
        remaining_failure_tolerance = (
            self.requirements["failure_tolerance"] - consumed_failure_tolerance
        )

        # Set subroutine error budget allocation
        subroutine_error_budget_allocation = [0.5, 0.5]

        # Compute number of Grover iterates needed
        number_of_grover_iterates = compute_number_of_grover_iterates_for_obl_amp(
            consumed_failure_tolerance, self.requirements["input_state_squared_overlap"]
        )

        # Set number of times called to number of Grover iterates
        self.state_preparation_oracle.number_of_times_called = number_of_grover_iterates
        self.mark_subspace.number_of_times_called = number_of_grover_iterates

        self.state_preparation_oracle.set_requirements(
            failure_tolerance=subroutine_error_budget_allocation[0]
            * remaining_failure_tolerance,
        )

        self.mark_subspace.set_requirements(
            failure_tolerance=subroutine_error_budget_allocation[1]
            * remaining_failure_tolerance,
        )

    def count_qubits(self):
        return self.state_preparation_oracle.count_qubits()


def compute_number_of_grover_iterates_for_obl_amp(
    failure_tolerance, input_state_squared_overlap
):
    # Compute number of Grover iterates needed for oblivious amplitude amplification
    number_of_grover_iterates = np.log(1 / failure_tolerance) / np.sqrt(
        input_state_squared_overlap
    )

    return number_of_grover_iterates


def test_oblivious_amplitude_amplification():
    obl_amp = ObliviousAmplitudeAmplification()

    input_state_squared_overlap = 0.2
    failure_tolerance = 0.01

    obl_amp.set_requirements(
        input_state_squared_overlap=input_state_squared_overlap,
        failure_tolerance=failure_tolerance,
    )

    # Run the profile for this subroutine
    obl_amp.run_profile()
    obl_amp.print_profile()

    return print(obl_amp.count_subroutines())


# test_oblivious_amplitude_amplification()
