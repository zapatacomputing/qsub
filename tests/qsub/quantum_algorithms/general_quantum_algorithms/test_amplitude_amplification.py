import pytest
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
    compute_number_of_grover_iterates_for_obl_amp,
    SubroutineModel,
)


def test_compute_number_of_grover_iterates_for_obl_amp():
    # Test with typical parameters
    result = compute_number_of_grover_iterates_for_obl_amp(0.01, 0.2)
    assert isinstance(result, float)


def test_obl_amp_initialization():
    # Test initialization without optional parameters
    obl_amp = ObliviousAmplitudeAmplification()
    assert isinstance(obl_amp.state_preparation_oracle, SubroutineModel)
    assert isinstance(obl_amp.mark_subspace, SubroutineModel)


def test_obl_amp_initialization_with_parameters():
    # Test initialization with optional parameters
    state_prep_oracle = SubroutineModel("state_prep")
    mark_subspace = SubroutineModel("mark_space")
    obl_amp = ObliviousAmplitudeAmplification(
        state_preparation_oracle=state_prep_oracle, mark_subspace=mark_subspace
    )
    assert obl_amp.state_preparation_oracle.task_name == "state_prep"
    assert obl_amp.mark_subspace.task_name == "mark_space"


def test_obl_amp_set_requirements():
    # Test set_requirements method
    obl_amp = ObliviousAmplitudeAmplification()
    obl_amp.set_requirements(input_state_squared_overlap=0.2, failure_tolerance=0.01)
    assert obl_amp.requirements["input_state_squared_overlap"] == 0.2
    assert obl_amp.requirements["failure_tolerance"] == 0.01


def test_obl_amp_populate_requirements_for_subroutines():
    # Test populate_requirements_for_subroutines method
    obl_amp = ObliviousAmplitudeAmplification()
    obl_amp.set_requirements(input_state_squared_overlap=0.2, failure_tolerance=0.01)
    obl_amp.populate_requirements_for_subroutines()
    # Assertions to check the correct population of requirements
    assert obl_amp.state_preparation_oracle.number_of_times_called > 0
    assert obl_amp.mark_subspace.number_of_times_called > 0
