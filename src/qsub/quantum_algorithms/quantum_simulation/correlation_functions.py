from typing import Optional
from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from qsub.utils import consume_fraction_of_error_budget
from sympy import symbols, Max


class CorrelationFunctionUnitary(SubroutineModel):
    def __init__(
        self,
        task_name="encode_correlation_function_unitary",
        time_evolution: Optional[SubroutineModel] = None,
        initial_state_preparation: Optional[SubroutineModel] = None,
        controlled_block_encoding_A: Optional[GenericBlockEncoding] = None,
        controlled_block_encoding_B: Optional[GenericBlockEncoding] = None,
        requirements=None,
    ):
        super().__init__(task_name, requirements)

        if time_evolution is not None:
            self.time_evolution = time_evolution
        else:
            self.time_evolution = SubroutineModel("time_evolution")

        if initial_state_preparation is not None:
            self.initial_state_preparation = initial_state_preparation
        else:
            self.initial_state_preparation = SubroutineModel(
                "initial_state_preparation"
            )

        if controlled_block_encoding_A is not None:
            self.controlled_block_encoding_A = controlled_block_encoding_A
        else:
            self.controlled_block_encoding_A = GenericBlockEncoding(
                "controlled_block_encoding_A"
            )

        if controlled_block_encoding_B is not None:
            self.controlled_block_encoding_B = controlled_block_encoding_B
        else:
            self.controlled_block_encoding_B = GenericBlockEncoding(
                "controlled_block_encoding_B"
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
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        # Allot error budget
        (
            truncation_error,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        (
            time_evolution_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        (
            initial_state_preparation_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        (
            controlled_block_encoding_A_failure_tolerance,
            controlled_block_encoding_B_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set requirements for subroutines

        self.time_evolution.set_requirements(
            failure_tolerance=time_evolution_failure_tolerance,
            evolution_time=self.requirements["evolution_time"],
        )
        # Set one call for forward and one for backward evolution
        self.time_evolution.number_of_times_called = 2

        self.initial_state_preparation.set_requirements(
            failure_tolerance=initial_state_preparation_failure_tolerance,
        )
        # Set one call for preparation and one for its inverse
        self.initial_state_preparation.number_of_times_called = 2

        self.controlled_block_encoding_A.set_requirements(
            failure_tolerance=controlled_block_encoding_A_failure_tolerance,
        )
        self.controlled_block_encoding_A.number_of_times_called = 1

        self.controlled_block_encoding_B.set_requirements(
            failure_tolerance=controlled_block_encoding_B_failure_tolerance,
        )
        self.controlled_block_encoding_B.number_of_times_called = 1


class DynamicCorrelationFunctionEstimation(SubroutineModel):
    def __init__(
        self,
        task_name="estimate_dynamic_correlation_function",
        requirements=None,
        estimate_real_part_of_correlation_function: Optional[SubroutineModel] = None,
        estimate_imaginary_part_of_correlation_function: Optional[
            SubroutineModel
        ] = None,
    ):
        super().__init__(task_name, requirements)

        if estimate_real_part_of_correlation_function is not None:
            self.estimate_real_part_of_correlation_function = (
                estimate_real_part_of_correlation_function
            )
        else:
            self.estimate_real_part_of_correlation_function = SubroutineModel(
                "estimate_real_part_of_correlation_function"
            )

        if estimate_imaginary_part_of_correlation_function is not None:
            self.estimate_imaginary_part_of_correlation_function = (
                estimate_imaginary_part_of_correlation_function
            )
        else:
            self.estimate_imaginary_part_of_correlation_function = SubroutineModel(
                "estimate_imaginary_part_of_correlation_function"
            )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        estimation_error: float = None,
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
        # Initialize CorrelationFunctionUnitary object
        correlation_function_unitary = CorrelationFunctionUnitary()

        # Set requirements for time_evolution task of CorrelationFunctionUnitary
        correlation_function_unitary.time_evolution.set_requirements(
            evolution_time=self.requirements["evolution_time"]
        )

        # Allocate failure tolerance evenly to the two subtasks
        failure_tolerance = self.requirements["failure_tolerance"] / 2

        # Set requirements for estimate_real_part_of_correlation_function task
        if self.estimate_real_part_of_correlation_function:
            self.estimate_real_part_of_correlation_function.set_requirements(
                failure_tolerance=failure_tolerance,
                estimation_error=self.requirements["estimation_error"],
            )

        # Set requirements for estimate_imaginary_part_of_correlation_function task
        if self.estimate_imaginary_part_of_correlation_function:
            self.estimate_imaginary_part_of_correlation_function.set_requirements(
                failure_tolerance=failure_tolerance,
                estimation_error=self.requirements["estimation_error"],
            )

    def count_qubits(self):
        return Max(
            self.estimate_real_part_of_correlation_function.count_qubits(),
            self.estimate_imaginary_part_of_correlation_function.count_qubits(),
        )
