from typing import Optional
from qsub.subroutine_model import SubroutineModel
from qsub.utils import consume_fraction_of_error_budget
import numpy as np
from qsub.generic_block_encoding import GenericBlockEncoding


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
        self.requirements["mark_drag_vector"] = SubroutineModel("mark_drag_vector")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        relative_estimation_error: float = None,
        estimated_drag_force: float = None,
        evolution_time: float = None,
        mu_P_A: float = None,
        kappa_P: float = None,
        norm_inhomogeneous_term_vector: float = None,
        norm_x_t: float = None,
        A_stable: bool = None,
        solve_quantum_ode: Optional[SubroutineModel] = None,
        number_of_spatial_grid_points: float = None,
        number_of_velocity_grid_points: float = None,
        x_length_in_meters: float = None,
        y_length_in_meters: float = None,
        z_length_in_meters: float = None,
        sphere_radius_in_meters: float = None,
        time_discretization_in_seconds: float = None,
        mark_drag_vector: Optional[SubroutineModel] = None,
        deviation_from_uniformity: float = 0.001
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

        solve_quantum_ode = self.requirements["solve_quantum_ode"]
        mark_drag_vector = self.requirements["mark_drag_vector"]

        # Set a subset of solve_quantum_ode requirements
        solve_quantum_ode.set_requirements(
            evolution_time=self.requirements["evolution_time"],
            mu_P_A=self.requirements["mu_P_A"],
            kappa_P=self.requirements["kappa_P"],
            norm_inhomogeneous_term_vector=self.requirements[
                "norm_inhomogeneous_term_vector"
            ],
            norm_x_t=self.requirements["norm_x_t"],
            A_stable=self.requirements["A_stable"],
        )

        # Set a subset of mark_drag_vector requirements
        mark_drag_vector.set_requirements(
            number_of_spatial_grid_points=self.requirements[
                "number_of_spatial_grid_points"
            ],
            number_of_velocity_grid_points=self.requirements[
                "number_of_velocity_grid_points"
            ],
            x_length_in_meters=self.requirements["x_length_in_meters"],
            y_length_in_meters=self.requirements["y_length_in_meters"],
            z_length_in_meters=self.requirements["z_length_in_meters"],
            sphere_radius_in_meters=self.requirements["sphere_radius_in_meters"],
            time_discretization_in_seconds=self.requirements[
                "time_discretization_in_seconds"
            ],
        )

        # Set amp est st prep subroutine as ode solver
        self.estimate_amplitude.run_iterative_qae_circuit.state_preparation_oracle = (
            solve_quantum_ode
        )

        # Set amp est mark subspace subroutine as mark_drag_vector
        self.estimate_amplitude.run_iterative_qae_circuit.mark_subspace = (
            mark_drag_vector
        )

        # Set number of calls to the amplitude estimation task to one
        self.estimate_amplitude.number_of_times_called = 1
      
        # Convert relative error to absolute error for amplitude estimation

        # state_prep_subnorm = self.estimate_amplitude.run_iterative_qae_circuit.state_preparation_oracle.get_subnormalization()
        state_prep_subnorm = np.sqrt(self.requirements["number_of_spatial_grid_points"])*(1 + 
            self.requirements["deviation_from_uniformity"]) # This is an upperbound
        amplitude_estimation_error = (
            self.requirements["estimated_drag_force"]
            * self.requirements["relative_estimation_error"]
            / (
                2
                * self.estimate_amplitude.run_iterative_qae_circuit.mark_subspace.get_subnormalization()
                * state_prep_subnorm
            )
        )

        # Set amp est requirements
        self.estimate_amplitude.set_requirements(
            estimation_error=amplitude_estimation_error,
            failure_tolerance=self.requirements["failure_tolerance"],
        )

    def count_qubits(self):
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]

        number_of_encoding_qubits = compute_number_of_encoding_qubits(
            number_of_spatial_grid_points, number_of_velocity_grid_points
        )

        return (
            self.estimate_amplitude.run_iterative_qae_circuit.state_preparation_oracle.count_qubits()
            + self.estimate_amplitude.run_iterative_qae_circuit.mark_subspace.count_qubits()
            - number_of_encoding_qubits
        )


class LBMDragCoefficientsReflection(SubroutineModel):
    def __init__(
        self,
        task_name="mark_drag_vector",
        requirements=None,
        quantum_adder: Optional[SubroutineModel] = None,
        quantum_comparator: Optional[SubroutineModel] = None,
        quantum_square: Optional[SubroutineModel] = None,
        quantum_sqrt: Optional[SubroutineModel] = None,
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

        if quantum_sqrt is not None:
            self.quantum_sqrt = quantum_sqrt
        else:
            self.quantum_sqrt = SubroutineModel("quantum_sqrt")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_spatial_grid_points: float = None,
        number_of_velocity_grid_points: float = None,
        x_length_in_meters: float = None,
        y_length_in_meters: float = None,
        z_length_in_meters: float = None,
        sphere_radius_in_meters: float = None,
        time_discretization_in_seconds: float = None,
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
            quantum_sqrt_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        (
            quantum_square_failure_tolerance,
            remaining_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        (
            quantum_adder_failure_tolerance,
            quantum_comparator_failure_tolerance,
        ) = consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)

        # TODO: finalize from Bhargav and update description
        self.quantum_adder.number_of_times_called = 2
        # Set quantum_adder requirements
        self.quantum_adder.set_requirements(
            failure_tolerance=quantum_adder_failure_tolerance,
            number_of_bits=compute_number_of_x_register_bits_for_coefficient_reflection(
                number_of_spatial_grid_points=self.requirements[
                    "number_of_spatial_grid_points"
                ],
            ),
        )

        # TODO: finalize from Bhargav and update description
        self.quantum_comparator.number_of_times_called = 2
        # Set quantum_comparator requirements
        self.quantum_comparator.set_requirements(
            failure_tolerance=quantum_comparator_failure_tolerance,
            number_of_bits=compute_number_of_x_register_bits_for_coefficient_reflection(
                number_of_spatial_grid_points=self.requirements[
                    "number_of_spatial_grid_points"
                ],
            ),
        )

        # Set number of calls to the quantum_sqrt
        self.quantum_sqrt.number_of_times_called = 1

        # Set quantum_sqrt requirements
        self.quantum_sqrt.set_requirements(
            failure_tolerance=quantum_sqrt_failure_tolerance,
            number_of_bits=compute_number_of_x_register_bits_for_coefficient_reflection(
                number_of_spatial_grid_points=self.requirements[
                    "number_of_spatial_grid_points"
                ],
            ),
        )

        # Set number of calls to the quantum_square: three for squaring x^2, y^2, and z^2
        self.quantum_square.number_of_times_called = 3

        number_of_bits_per_spatial_register = (
            compute_number_of_x_register_bits_for_coefficient_reflection(
                number_of_spatial_grid_points=self.requirements[
                    "number_of_spatial_grid_points"
                ],
            )
        )

        # Set quantum_square requirements
        self.quantum_square.set_requirements(
            failure_tolerance=quantum_square_failure_tolerance,
            number_of_bits_total=2 * number_of_bits_per_spatial_register,
            number_of_bits_above_decimal_place=2 * number_of_bits_per_spatial_register,
        )

    def get_subnormalization(self):
        # Returns the normalization factor for the vector encoding the marked state
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]

        x_length_in_meters = self.requirements["x_length_in_meters"]
        y_length_in_meters = self.requirements["y_length_in_meters"]
        z_length_in_meters = self.requirements["z_length_in_meters"]
        volume = x_length_in_meters * y_length_in_meters * z_length_in_meters

        time_discretization_in_seconds = self.requirements[
            "time_discretization_in_seconds"
        ]

        sphere_radius_in_meters = self.requirements["sphere_radius_in_meters"]

        # From drag estimation paper
        subnormalization = (
            (9 * np.sqrt(2 * np.pi))
            * (volume ** (2 / 3) * sphere_radius_in_meters)
            / (
                time_discretization_in_seconds
                * number_of_spatial_grid_points ** (2 / 3)
            )
        )

        return subnormalization

    def count_qubits(self):
        # From drag estimation paper
        # TODO: update this to something more accurate
        number_of_bits_for_x_dimension = (
            compute_number_of_x_register_bits_for_coefficient_reflection(
                number_of_spatial_grid_points=self.requirements[
                    "number_of_spatial_grid_points"
                ],
            )
        )

        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]

        number_of_encoding_qubits = compute_number_of_encoding_qubits(
            number_of_spatial_grid_points, number_of_velocity_grid_points
        )

        # TODO: update to include ancilla from subroutines
        return number_of_encoding_qubits


def compute_number_of_encoding_qubits(
    number_of_spatial_grid_points: float, number_of_velocity_grid_points: float
):
    # Returns the number of qubits needed to encode the drag coefficient in the marked state vector
    # From drag estimation paper
    return np.ceil(np.log2(number_of_spatial_grid_points)) + np.ceil(
        np.log2(number_of_velocity_grid_points)
    )


def compute_number_of_x_register_bits_for_coefficient_reflection(
    number_of_spatial_grid_points: float,
):
    # Returns the number of bits needed to represent the reflection of the drag coefficient
    # in the marked state vector
    # From drag estimation paper

    # Each comparator acts on a register that is the number of bits needed to encode
    # the number of spatial grid points in a single dimension
    # TODO: update this to something more accurate
    number_of_spatial_grid_points_in_x_dimension = number_of_spatial_grid_points ** (
        1 / 3
    )
    number_of_bits_for_x_dimension = np.ceil(
        np.log2(number_of_spatial_grid_points_in_x_dimension)
    )

    return number_of_bits_for_x_dimension


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
        number_of_spatial_grid_points: float = None,
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


class LBMLinearTermBlockEncoding(GenericBlockEncoding):
    def __init__(
        self,
        task_name="block_encode_linear_term",
        requirements=None,
    ):
        super().__init__(task_name, requirements)
        self.t_gate = SubroutineModel("t_gate")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_spatial_grid_points: float = None,
        number_of_velocity_grid_points: float = None,
    ):
        args = locals()
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        if not hasattr(self, "requirements"):
            self.requirements = {}
        self.requirements.update(args)
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Set number of calls to the t_gate subroutine (NOTE: T gate counts 
        # will be a sum of t gates from streaming matrix and F1 collision matrix)

        n_f1_tgates = 635 * np.log2(
            555 / self.requirements["failure_tolerance"]
        )  
        n_spatial_qubits = np.log2(self.requirements["number_of_spatial_grid_points"])
        n_streaming_tgates = 12 * n_spatial_qubits**2 + 32 * n_spatial_qubits + 12*(n_spatial_qubits-1) 
        + 72*(n_spatial_qubits-1)

        self.t_gate.number_of_times_called = n_f1_tgates + n_streaming_tgates

        # Set t_gate requirements
        self.t_gate.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"]
            / self.t_gate.number_of_times_called,
        )

    def get_subnormalization(self):
        # NOTE: subnormalization comes from linear combination of block encoding lemma
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        subnormalization = 1 / (1.58950617 * number_of_velocity_grid_points)
        return subnormalization

    def count_encoding_qubits(self):
        """The number of qubits used to store the linear factor in the solution vector
        """
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_spatial_grid_points)) + (
            np.ceil(np.log2(number_of_velocity_grid_points))
        )
        return number_of_qubits

    def count_block_encoding_ancilla_qubits(self):
        """The number of qubits that show up in the triple that defines the block encoding
           This is the sum of ancilla qubits from the streaming and linear collision matrix.
        """
   
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_velocity_grid_points)) + 3 
        + np.ceil(np.log2(number_of_spatial_grid_points)) + 1
        return number_of_qubits

    def count_qubits(self):
        # For now (5/6/2024) this will return count_encoding_qubits + count_block_encoding_ancilla_qubits
        return self.count_encoding_qubits + self.count_block_encoding_ancilla_qubits
       


class LBMQuadraticTermBlockEncoding(GenericBlockEncoding):
    def __init__(
        self,
        task_name="block_encode_quadratic_term",
        requirements=None,
    ):
        super().__init__(task_name, requirements)
        self.t_gate = SubroutineModel("t_gate")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_spatial_grid_points: float = None,
        number_of_velocity_grid_points: float = None,
    ):
        args = locals()
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        if not hasattr(self, "requirements"):
            self.requirements = {}
        self.requirements.update(args)
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Set number of calls to the t_gate subroutine

        log_n_spatial_qubits_squared = np.log2(self.requirements["number_of_spatial_grid_points"]**2)
        self.t_gate.number_of_times_called  = 8*log_n_spatial_qubits_squared + 5965*np.log2(5187/self.requirements["failure_tolerance"])
        -16 + 2* log_n_spatial_qubits_squared*(log_n_spatial_qubits_squared-1)

        # Set t_gate requirements
        self.t_gate.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"]
            / self.t_gate.number_of_times_called,
        )

    def count_qubits(self):
        # As of 5/6/2024 will just add block encoding qubits plus system qubits
        return self.count_block_encoding_ancilla_qubits + self.count_encoding_qubits

    def get_subnormalization(self):
        # The subnormalization is set to a constant value of 1
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        subnormalization = 1 / (1.4444444 * number_of_velocity_grid_points)
        return subnormalization

    def count_encoding_qubits(self):
        """The number of qubits used to store the quadratic factor in the solution vector
        """
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_spatial_grid_points)) + (
            np.ceil(np.log2(number_of_velocity_grid_points))
        )
        return number_of_qubits

    def count_block_encoding_ancilla_qubits(self):
        """The number of qubits that show up in the triple that defines the block encoding
        """
   
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_velocity_grid_points)) + 3
        return number_of_qubits

class LBMCubicTermBlockEncoding(GenericBlockEncoding):
    def __init__(
        self,
        task_name="block_encode_cubic_term",
        requirements=None,
    ):
        super().__init__(task_name, requirements)
        self.t_gate = SubroutineModel("t_gate")

    def set_requirements(
        self,
        failure_tolerance: float = None,
        number_of_spatial_grid_points: float = None,
        number_of_velocity_grid_points: float = None,
    ):
        args = locals()
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        if not hasattr(self, "requirements"):
            self.requirements = {}
        self.requirements.update(args)
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        # Set number of calls to the t_gate subroutine
        log_n_spatial_qubits_cubed = np.log2(self.requirements["number_of_spatial_grid_points"]**3)
        self.t_gate.number_of_times_called  = 8*log_n_spatial_qubits_cubed + 39991*np.log2(34775/self.requirements["failure_tolerance"])
        -16 + 2* log_n_spatial_qubits_cubed*(log_n_spatial_qubits_cubed-1)

        # Set t_gate requirements
        self.t_gate.set_requirements(
            failure_tolerance=self.requirements["failure_tolerance"]
            / self.t_gate.number_of_times_called,
        )

    def get_subnormalization(self):
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        subnormalization = 1 / (1.44444444 * number_of_velocity_grid_points)
        return subnormalization

    def count_qubits(self):
        return self.count_block_encoding_ancilla_qubits + self.count_encoding_qubits

    def count_block_encoding_ancilla_qubits(self):
        """The number of qubits that show up in the triple that defines the block encoding
           This is the sum of ancilla qubits from the streaming and linear collision matrix.
        """
   
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_velocity_grid_points)) + 3
        return number_of_qubits

    def count_encoding_qubits(self):
        """The number of qubits used to store the quadratic factor in the solution vector
        """
        number_of_spatial_grid_points = self.requirements[
            "number_of_spatial_grid_points"
        ]
        number_of_velocity_grid_points = self.requirements[
            "number_of_velocity_grid_points"
        ]
        number_of_qubits = np.ceil(np.log2(number_of_spatial_grid_points)) + (
            np.ceil(np.log2(number_of_velocity_grid_points))
        )
        return number_of_qubits
