from typing import Optional
from ...subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget
import warnings
from dataclasses import dataclass

@dataclass
class LBMDragEstimationData:
    failure_tolerance: float = 0
    estimation_error: float = 0
    estimated_drag_force: float = 0
    evolution_time: float = 0
    mu_P_A: float = 0
    kappa_P: float = 0
    norm_inhomogeneous_term_vector: float = 0
    norm_x_t: float = 0
    A_stable: bool = False
    # Intialize subroutines as generic routines with task name
    solve_quantum_ode: SubroutineModel =  SubroutineModel("solve_quantum_ode") 
    mark_drag_vector: SubroutineModel = SubroutineModel("mark_drag_vector")

@dataclass
class LBMDragReflectionData
    failure_tolerance: float = 0

@dataclass
class SphereBoundaryOracleData:
    failure_tolerance: float = None,
    radius: float = None,
    grid_spacing: float = None,

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

        self.set_requirements(LBMDragEstimationData)

    def populate_requirements_for_subroutines(self):
        # Note: This subroutine consumes no failure probability.
        # Rather, it properly allocates requirements and subroutines
        # to the subtasks of amplitude estimation

        mark_drag_vector = self.requirements["mark_drag_vector"]
        solve_quantum_ode = self.requirements["solve_quantum_ode"]

        # Set amp est st prep subroutine as ode solver
        self.estimate_amplitude.state_preparation_oracle = solve_quantum_ode

        # Set amp est mark subspace subroutine as mark_drag_vector
        self.estimate_amplitude.mark_subspace = mark_drag_vector

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

        # Set number of calls to the amplitude estimation task to one
        self.estimate_amplitude.number_of_times_called = 1

        # The QAE amplitude is the square of the estimate of interest
        # and is scaled by known normalization factors in the vectors that encode the
        # initial state and the mark state. One consequence of the square relationship
        # is that the amplitude estimation error is now dependent on the quantity that
        # is to be estimated. This is because smaller amplitudes mean that the square root
        # operation increasingly expands the relative error.
        # TODO: add equation reference from notes on drag estimation
        amplitude_estimation_error = (
            self.requirements["estimation_error"]
            * (2 * self.requirements["estimated_drag_force"])
            / (
                self.estimate_amplitude.mark_subspace.get_normalization_factor()
                * self.estimate_amplitude.state_preparation_oracle.get_normalization_factor()
            )
        )

        # Set amp est requirements
        self.estimate_amplitude.set_requirements(
            estimation_error=amplitude_estimation_error,
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def count_qubits(self):
        return self.estimate_amplitude.count_qubits()

class LBMDragReflection(SubroutineModel):
    def __init__(
        self,
        task_name="mark_drag_vector",
        requirements=None,
        compute_boundary: Optional[SubroutineModel] = None,
    ):
        super().__init__(task_name, requirements)

        if compute_boundary is not None:
            self.compute_boundary = compute_boundary
        else:
            self.compute_boundary = SubroutineModel("compute_boundary")
        self.set_requirements(LBMDragReflectionData)

    def populate_requirements_for_subroutines(self):
        # Set number of calls to the quadratic term block encoding
        self.compute_boundary.number_of_times_called = 1

        # Set quadratic term block encoding requirements
        self.compute_boundary.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def get_normalization_factor(self):
        # Returns the normalization factor for the vector encoding the marked state
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
        self.set_requirements(SphereBoundaryOracleData)

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
