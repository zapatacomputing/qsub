
from dataclasses import dataclass
from subroutine_model import SubroutineModel
@dataclass
class ObliviousAmplitudeAmplificationData:
      input_state_squared_overlap:float = 0.5
      failure_tolerance: float = 0.001
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
    kappa_P: float =0
    mu_P_A: float = 0   
    A_stable: float = 0


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
class LBMDragReflectionData:
    failure_tolerance: float = 0

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