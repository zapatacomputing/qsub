import pytest
from src.qsub.subroutine_model import SubroutineModel
from src.qsub.quantum_algorithms.gaussian_filtering_gsee import (
    GF_LD_GSEE,
    get_gf_ld_gsee_max_evolution_time,
    get_gf_ld_gsee_num_circuit_repetitions,
    _get_sigma,
    _get_epsilon_1,
)


# Test initialization of GF_LD_GSEE
def test_initialization():
    obj = GF_LD_GSEE()
    assert obj.task_name == "ground_state_energy_estimation"
    assert isinstance(obj.c_time_evolution, SubroutineModel)


# Test setting of requirements
def test_set_requirements():
    obj = GF_LD_GSEE()
    obj.set_requirements(0.5, 1.0, 0.8, 0.01, 0.1, None)
    assert obj.requirements["alpha"] == 0.5
    # Add further assertions for other parameters


# Test population of requirements for subroutines
def test_populate_requirements_for_subroutines():
    obj = GF_LD_GSEE()
    obj.set_requirements(0.5, 1.0, 0.8, 0.01, 0.1, None)
    obj.populate_requirements_for_subroutines()
    # Check that the c_time_evolution requirements have been set correctly.
    # This might need the real computation logic from your program to validate against.


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
