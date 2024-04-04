
from dataclasses import dataclass

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