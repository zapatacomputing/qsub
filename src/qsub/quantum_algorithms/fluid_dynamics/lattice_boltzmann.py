from typing import Optional
from ...subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    compute_amp_est_error_from_block_encoding,
)
import warnings


class LBMDragEstimation(SubroutineModel):
    def __init__(
        self,
        task_name="estimate_drag_from_lbm",
        requirements=None,
        estimate_amplitude: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if estimate_amplitude is not None:
            self.estimate_amplitude = estimate_amplitude
        else:
            self.estimate_amplitude = SubroutineModel("estimate_amplitude")

        # Initialize the sub-subtask requirements as generic subroutines with task names
        self.requirements["solve_quantum_ode"] = SubroutineModel("solve_quantum_ode")
        self.requirements["block_encode_drag_operator"] = SubroutineModel(
            "block_encode_drag_operator"
        )

    def set_requirements(
        self,
        failure_tolerance: float = None,
        estimation_error: float = None,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        norm_inhomogeneous_term_vector: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        solve_quantum_ode: Optional[SubroutineModel] = None,
        block_encode_drag_operator: Optional[SubroutineModel] = None,
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

        for k, v in args.items():
            if k in self.requirements:
                if not isinstance(self.requirements[k], SubroutineModel):
                    self.requirements[k] = v
            else:
                self.requirements[k] = v if v is not None else SubroutineModel(k)

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Note: This subroutine consumes no failure probability.
        # Rather, it properly allocates requirements and subroutines
        # to the subtasks of amplitude estimation

        block_encode_drag_operator = self.requirements["block_encode_drag_operator"]
        solve_quantum_ode = self.requirements["solve_quantum_ode"]
        # Set number of calls to the amplitude estimation task to one
        self.estimate_amplitude.number_of_times_called = 1

        amplitude_estimation_error = compute_amp_est_error_from_block_encoding(
            self.requirements["estimation_error"],
            block_encode_drag_operator.get_subnormalization(),
        )

        # Set amp est requirements
        self.estimate_amplitude.set_requirements(
            estimation_error=amplitude_estimation_error,
            failure_tolerance=self.requirements["failure_tolerance"],
        )

        # Set amp est st prep subroutine as ode solver
        self.estimate_amplitude.state_preparation_oracle = solve_quantum_ode

        # Set amp est mark subspace subroutine as block_encode_drag_operator
        self.estimate_amplitude.mark_subspace = block_encode_drag_operator

        # Set final_state_prep requirements
        self.estimate_amplitude.state_preparation_oracle.set_requirements(
            evolution_time=self.requirements["evolution_time"],
            mu_P_A=self.requirements["mu_P_A"],
            kappa_P=self.requirements["kappa_P"],
            norm_inhomogeneous_term_vector=self.requirements[
                "norm_inhomogeneous_term_vector"
            ],
            norm_x_t=self.requirements["norm_x_t"],
            A_stable=self.requirements["A_stable"],
        )

    def count_qubits(self):
        return self.estimate_amplitude.count_qubits()


class LBMDragOperator(SubroutineModel):
    def __init__(
        self,
        task_name="mark_drag_operator_subspace",
        requirements=None,
        compute_boundary: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if compute_boundary is not None:
            self.compute_boundary = compute_boundary
        else:
            self.compute_boundary = SubroutineModel("compute_boundary")

    def set_requirements(
        self,
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
        # Set number of calls to the quadratic term block encoding
        self.compute_boundary.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.compute_boundary.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_subnormalization(self):
        warnings.warn("This function is not fully implemented.", UserWarning)
        return 42


class SphereBoundaryOracle(SubroutineModel):
    def __init__(
        self,
        task_name="compute_boundary",
        requirements=None,
        quantum_adder: Optional[SubroutineModel] = None,
        quantum_comparator: Optional[SubroutineModel] = None,
        quantum_square: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if quantum_adder is not None:
            self.quantum_adder = quantum_adder
        else:
            self.quantum_adder = SubroutineModel("quantum_adder")

        if quantum_comparator is not None:
            self.quantum_comparator = quantum_comparator
        else:
            self.quantum_comparator = SubroutineModel("quantum_comparator")

        if quantum_square is not None:
            self.quantum_square = quantum_square
        else:
            self.quantum_square = SubroutineModel("quantum_square")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        radius: float = None,
        grid_spacing: float = None,
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
            quantum_square_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        (
            quantum_adder_failure_tolerance,
            quantum_comparator_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # Set number of calls to the quantum_adder: two for adding y^2 and z^2 to x^2
        self.quantum_adder.number_of_times_called = 2

        # Set quantum_adder requirements
        self.quantum_adder.set_requirements(
            failure_tolerance=quantum_adder_failure_tolerance,
        )

        # Set number of calls to the quantum_comparator: two for comparing x^2+y^2+z^2 to r^2
        self.quantum_adder.number_of_times_called = 1

        # Set quantum_comparator requirements
        self.quantum_adder.set_requirements(
            failure_tolerance=quantum_comparator_failure_tolerance,
        )

        # Set number of calls to the quantum_square: three for squaring x^2, y^2, and z^2
        self.quantum_square.number_of_times_called = 3

        # Set quantum_comparator requirements
        self.quantum_square.set_requirements(
            failure_tolerance=quantum_square_failure_tolerance,
        )
