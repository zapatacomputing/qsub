import pytest
from qsub.subroutine_model import SubroutineModel
from qsub.quantum_algorithms.gaussian_filtering_gsee import (
    GF_LD_GSEE,
    get_gf_ld_gsee_max_evolution_time,
    get_gf_ld_gsee_num_circuit_repetitions,
    _get_sigma,
    _get_epsilon_1,
)
from qsub.data_classes import GF_LD_GSEEData

@pytest.fixture()
def mock_data_class():
    data_instance = GF_LD_GSEEData()
    data_instance.alpha=0.1
    data_instance.failure_tolerance=0.1
    data_instance.energy_gap=0.1
    data_instance.precision=0.1
    data_instance.square_overlap=0.1
    data_instance.hamiltonian= None
    return data_instance

@pytest.fixture()
def c_time_evolution():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()
    return MockSubroutineModel(task_name='c_time_evolution')

# Test population of requirements for subroutines
def test_populate_requirements_for_subroutines(c_time_evolution, mock_data_class):
    gaussian_filter = GF_LD_GSEE(c_time_evolution=c_time_evolution)
    gaussian_filter.set_requirements(mock_data_class)
    print(gaussian_filter.requirements)
    gaussian_filter.populate_requirements_for_subroutines()

# Test _get_sigma
def test_get_sigma():
    sigma = _get_sigma(0.5, 1.0, 0.8, 0.01)
    assert isinstance(sigma, float)
    # Optionally, you can compute the expected value manually and check against it


# Test _get_epsilon_1
def test_get_epsilon_1():
    sigma = _get_sigma(0.5, 1.0, 0.8, 0.01)
    epsilon_1 = _get_epsilon_1(0.01, 0.8, sigma)
    assert isinstance(epsilon_1, float)


# Test get_gf_ld_gsee_max_evolution_time
def test_get_gf_ld_gsee_max_evolution_time():
    max_time = get_gf_ld_gsee_max_evolution_time(0.5, 1.0, 0.8, 0.01)
    assert isinstance(max_time, float)


# Test get_gf_ld_gsee_num_circuit_repetitions
def test_get_gf_ld_gsee_num_circuit_repetitions():
    reps = get_gf_ld_gsee_num_circuit_repetitions(0.5, 1.0, 0.8, 0.01, 0.1)
    assert isinstance(reps, float)
