import pytest
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
    compute_number_of_grover_iterates_for_obl_amp,
    SubroutineModel,
)
from typing import Optional
from dataclasses import dataclass


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
    @dataclass
    class SubroutineModelData:
        input_state_squared_overlap: float = 0.5
        failure_tolerance: float = 0.03
    return SubroutineModelData()

def test_compute_number_of_grover_iterates_for_obl_amp():
    # Test with typical parameters
    result = compute_number_of_grover_iterates_for_obl_amp(0.01, 0.2)
    assert isinstance(result, float)


def test_obl_amp_populate_requirements_for_subroutines(mock_data_class, state_preparation_oracle, mark_subspace):

    # Test populate_requirements_for_subroutines method
    obl_amp = ObliviousAmplitudeAmplification(state_preparation_oracle, mark_subspace)
    obl_amp.set_requirements(mock_data_class)
    obl_amp.populate_requirements_for_subroutines()
    # Assertions to check the correct population of requirements
    assert obl_amp.state_preparation_oracle.number_of_times_called > 0
    assert obl_amp.mark_subspace.number_of_times_called > 0
