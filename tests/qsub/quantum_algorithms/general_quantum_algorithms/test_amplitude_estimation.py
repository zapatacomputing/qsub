import numpy as np
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
    SubroutineModel,
    compute_number_of_grover_iterates_for_quantum_amp_est,
    compute_number_of_grover_iterates_for_iterative_amp_est
)
import pytest
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
        estimation_error: float = 0.01
        failure_tolerance: float = 0.1
    return SubroutineModelData()

# Test the function for computing the number of Grover iterates
# TODO: This test uses a function compute_number_of_grover_iterates_for_amp_est which
# is not defined anywhere in the codebase. Maybe it was removed or renamed?
# Amara: changed it to the one imported TODO: make sure this is the right
# substitution
# def test_compute_number_of_grover_iterates_for_amp_est():
#     failure_tolerance = 0.1
#     estimation_error = 0.01
#     expected_iterates = np.log(1 / failure_tolerance) / estimation_error
#     actual_iterates = compute_number_of_grover_iterates_for_iterative_amp_est(
#         failure_tolerance, estimation_error
#     )
#     assert actual_iterates == expected_iterates


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
