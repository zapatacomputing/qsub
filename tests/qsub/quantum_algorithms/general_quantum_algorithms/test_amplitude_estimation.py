import numpy as np
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
    SubroutineModel
)
from qsub.data_classes import QuantumAmplitudeEstimationData
import pytest
from dataclasses import dataclass, is_dataclass

@pytest.fixture()
def state_preparation_oracle():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()

    return MockSubroutineModel(task_name="state_prep")

@pytest.fixture()
def mark_subspace():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()

    return MockSubroutineModel(task_name="mark_space")

@pytest.fixture()
def mock_data_class():
    data_instance = QuantumAmplitudeEstimationData()
    data_instance.amplitude=0.6
    data_instance.failure_tolerance=0.1
    data_instance.estimation_error=0.001
    return data_instance


# Test populating requirements for subroutines in QuantumAmplitudeEstimation
def test_quantum_amplitude_estimation_populate_requirements_for_subroutines(mock_data_class, 
        state_preparation_oracle, 
        mark_subspace
    ):
    qae = QuantumAmplitudeEstimation(state_preparation_oracle, mark_subspace)
    qae.set_requirements(mock_data_class)
    print(qae.requirements)
    qae.populate_requirements_for_subroutines()
    assert qae.state_preparation_oracle.requirements["failure_tolerance"] is not None
    assert qae.mark_subspace.requirements["failure_tolerance"] is not None
