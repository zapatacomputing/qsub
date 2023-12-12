import pytest
from src.qsub.subroutine_model import SubroutineModel
from src.qsub.quantum_algorithms.general_quantum_algorithms.quantum_linear_systems_algorithms import (
    TaylorQLSA,
    get_taylor_qlsa_num_block_encoding_calls,
)


def test_get_taylor_qlsa_num_block_encoding_calls_normal():
    # Test with normal parameters
    result = get_taylor_qlsa_num_block_encoding_calls(0.1, 1.0, 4.0)
    assert isinstance(result, float)


def test_get_taylor_qlsa_num_block_encoding_calls_invalid_failure_probability():
    # Test with invalid failure_probability
    with pytest.raises(ValueError):
        get_taylor_qlsa_num_block_encoding_calls(0.25, 1.0, 4.0)


def test_get_taylor_qlsa_num_block_encoding_calls_invalid_condition_number():
    # Test with invalid condition_number
    with pytest.raises(ValueError):
        get_taylor_qlsa_num_block_encoding_calls(0.1, 1.0, 2.0)


def test_taylor_qlsa_init_with_block_encoding():
    # Test initialization with linear_system_block_encoding
    block_encoding = SubroutineModel("test")
    taylor_qlsa = TaylorQLSA(linear_system_block_encoding=block_encoding)
    assert taylor_qlsa.linear_system_block_encoding.task_name == "test"


def test_taylor_qlsa_init_without_block_encoding():
    # Test initialization without linear_system_block_encoding
    taylor_qlsa = TaylorQLSA()
    assert (
        taylor_qlsa.linear_system_block_encoding.task_name
        == "linear_system_block_encoding"
    )


def test_taylor_qlsa_set_requirements():
    # Test set_requirements method
    taylor_qlsa = TaylorQLSA()
    taylor_qlsa.set_requirements(
        failure_tolerance=0.1, subnormalization=1.0, condition_number=4.0
    )
    assert taylor_qlsa.requirements == {
        "failure_tolerance": 0.1,
        "subnormalization": 1.0,
        "condition_number": 4.0,
    }


def test_taylor_qlsa_populate_requirements_for_subroutines():
    # Test populate_requirements_for_subroutines method
    taylor_qlsa = TaylorQLSA()
    taylor_qlsa.set_requirements(
        failure_tolerance=0.1, subnormalization=1.0, condition_number=4.0
    )
    taylor_qlsa.populate_requirements_for_subroutines()
    # Assertions to check the correct population of requirements
    assert taylor_qlsa.linear_system_block_encoding.number_of_times_called > 0
