import pytest
from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericLinearSystemBlockEncoding
from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    TaylorQLSA,
    get_taylor_qlsa_num_block_encoding_calls,
)
from dataclasses import dataclass

@pytest.fixture()
def linear_system_block_encoding():
    class MockSubroutineModel(GenericLinearSystemBlockEncoding):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()

    return MockSubroutineModel(task_name="linear_system_block_encoding")

@pytest.fixture()
def prepare_b_vector():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()

    return MockSubroutineModel(task_name="test")

@pytest.fixture()
def mock_data_class():
    @dataclass
    class SubroutineModelData:
        failure_tolerance: float = 0.1
        subnormalization: float = 1
        condition_number: float = 4
    return SubroutineModelData()

def test_get_taylor_qlsa_num_block_encoding_calls_normal():
    # Test with normal parameters
    result = get_taylor_qlsa_num_block_encoding_calls(0.1, 1.0, 4.0)
    assert isinstance(result, tuple) and all(isinstance(item, float) for item in result)


def test_get_taylor_qlsa_num_block_encoding_calls_invalid_failure_probability():
    # Test with invalid failure_probability
    with pytest.raises(ValueError):
        get_taylor_qlsa_num_block_encoding_calls(0.25, 1.0, 4.0)


def test_get_taylor_qlsa_num_block_encoding_calls_invalid_condition_number():
    # Test with invalid condition_number
    with pytest.raises(ValueError):
        get_taylor_qlsa_num_block_encoding_calls(0.1, 1.0, 2.0)


def test_taylor_qlsa_init_with_block_encoding(linear_system_block_encoding, prepare_b_vector):
    # Test initialization with linear_system_block_encoding
    taylor_qlsa = TaylorQLSA(linear_system_block_encoding, prepare_b_vector)
    assert taylor_qlsa.linear_system_block_encoding.task_name == "linear_system_block_encoding"


def test_taylor_qlsa_populate_requirements_for_subroutines(mock_data_class,
        linear_system_block_encoding, 
        prepare_b_vector
        ):
    # Test populate_requirements_for_subroutines method
    taylor_qlsa = TaylorQLSA(linear_system_block_encoding, prepare_b_vector)
    taylor_qlsa.set_requirements(mock_data_class)
    taylor_qlsa.populate_requirements_for_subroutines()
