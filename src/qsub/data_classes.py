
from dataclasses import dataclass
from qsub.subroutine_model import SubroutineModel
from typing import Any
from qsub.generic_block_encoding import GenericLinearSystemBlockEncoding
class ObliviousAmplitudeAmplificationData:
      input_state_squared_overlap:float = 0.5
      failure_tolerance: float = 0.001

@dataclass
class QuantumAmplitudeEstimationData:
    amplitude: float = 0.6
    failure_tolerance: float = 0.001
    estimation_error: float = 0.001

@dataclass
class GF_LD_GSEEData:
    alpha: float =0.1
    energy_gap: float = 0.1
    square_overlap: float = 0.1
    precision: float = 0.1
    failure_tolerance: float= 0.1
    hamiltonian: Any = None

@dataclass 
class ControlledTimeEvoutionData:
    evolution_time: float=1
    hamiltonian = None
    failure_tolerance: float = 0.1

@dataclass
class IterativeQuantumAmplitudeEstimationCircuitData:
    failure_tolerance:float =0.1
    estimation_error:float =0.1

@dataclass
class IterativeQuantumAmplitudeEstimationAlgorithmData:
    estimation_error: float = 0.1
    failure_tolerance: float = 0.1

@dataclass
class StatePreparationOracleData:
    failure_tolerance: float = 0.001
@dataclass
class MarkedSubspaceOracleData:
    failure_tolerance: float = 0.001

@dataclass
class LinearSystemBlockEncodingData:
    failure_tolerance: float = 0.001
    sub_normaliation: float = 0.001
    condition_number: float = 0.001

@dataclass
class CarlemanBlockEncodingData:
    failure_tolerance: float = 0.1
    kappa_P: float =0.1
    mu_P_A: float = 0.1  
    A_stable: float = 0.1

@dataclass
class LBMDragEstimationData:
    failure_tolerance: float = 0.1
    relative_estimation_error: float = 0.1
    estimation_error: float = 0.1
    estimated_drag_force: float = 0.1
    evolution_time: float = 0.1
    mu_P_A: float = 0.1
    kappa_P: float = 0.1
    norm_inhomogeneous_term_vector: float = 0.1
    norm_x_t: float = 0.1
    A_stable: bool = False
    # Intialize subroutines as generic routines with task name
    solve_quantum_ode: SubroutineModel =  SubroutineModel("solve_quantum_ode") 
    mark_drag_vector: SubroutineModel = SubroutineModel("mark_drag_vector")
    number_of_spatial_grid_points:float = 0.1,
    number_of_velocity_grid_points:float = 0.1
    x_length_in_meters: float =0.1
    y_length_in_meters: float = 0.1
    z_length_in_meters: float = 0.1
    sphere_radius_in_meters: float = 0.1
    time_discretization_in_seconds: float =0.1


@dataclass
class LBMDragCoefficientsReflectionData:
    failure_tolerance: float = 0.1
    number_of_spatial_grid_points: float = 0.1
    number_of_velocity_grid_points: float = 0.1
    x_length_in_meters: float = 0.1
    y_length_in_meters: float = 0.1
    z_length_in_meters: float = 0.1
    sphere_radius_in_meters: float = 0.1
    time_discretization_in_seconds: float = 0.1


@dataclass
class SphereBoundaryOracleData:
    failure_tolerance: float = None
    radius: float = None
    grid_spacing: float = None

@dataclass
class GenericLinearSystemBlockEncodingData:
        failure_tolerance: float = 0.1
        kappa_P: float = 0.1
        mu_P_A: float = 0.1
        A_stable: float = 0.01 

@dataclass
class LBMLinearTermBlockEncodingData:
    failure_tolerance: float = 0.1
    number_of_spatial_grid_points: float = 0.1
    number_of_velocity_grid_points: float = 0.1

@dataclass
class LBMQuadraticTermBlockEncodingData:
    failure_tolerance: float = 0.1
    number_of_spatial_grid_points: float = 0.1
    number_of_velocity_grid_points: float = 0.1

@dataclass
class LBMCubicTermBlockEncodingData:
    failure_tolerance: float = 0.1
    number_of_spatial_grid_points: float = 0.1
    number_of_velocity_grid_points: float = 0.1

@dataclass
class TaylorQLSAData:
    failure_tolerance: float = 0.1
@dataclass
class TaylorQuantumODESolverData:
    evolution_time: float = 0.1
    mu_P_A: float = 0.1
    kappa_P: float = 0.1
    failure_tolerance: float = 0.1
    norm_inhomogeneous_term_vector: float = 0.1
    norm_x_t: float = 0.1
    A_stable: bool = False
    solve_linear_system: SubroutineModel = SubroutineModel(
            "solve_linear_system"
        )
    ode_matrix_block_encoding: SubroutineModel = GenericLinearSystemBlockEncoding(
            "ode_matrix_block_encoding"
        )
    prepare_inhomogeneous_term_vector: SubroutineModel = SubroutineModel(
            "prepare_inhomogeneous_term_vector"
        )
    prepare_initial_vector: SubroutineModel = SubroutineModel(
            "prepare_initial_vector"
        )

@dataclass
class ODEHistoryBlockEncodingData:
    failure_tolerance: float = 0.1
    evolution_time: float = 0.1
    epsilon_td: float = 0.1
    norm_inhomogeneous_term_vector: float =0.1
    norm_x_t: float = 0.1
@dataclass
class ODEHistoryBVectorData:
    failure_tolerance: float =0.1

@dataclass
class GidneyAdderData:
    failure_tolerance: float = None
    number_of_bits: float = None
@dataclass
class GidneyComparator:
    failure_tolerance: float = None
    number_of_bits: float = None

@dataclass
class GidneyMultiplier:
    failure_tolerance: float = None,
    number_of_bits_total: float = None
    number_of_bits_above_decimal_place: float = None
@dataclass
class GidneySqaureRoot:
    failure_tolerance: float = None
    number_of_bits: float = None