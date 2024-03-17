import numpy as np
from src.qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    CoherentQuantumAmplitudeEstimation,
    compute_number_of_grover_iterates_for_amp_est,
    QuantumAmplitudeEstimation,
    SubroutineModel,
)


# Test the function for computing the number of Grover iterates
# TODO: This test uses a function compute_number_of_grover_iterates_for_amp_est which
# is not defined anywhere in the codebase. Maybe it was removed or renamed?
def test_compute_number_of_grover_iterates_for_amp_est():
    failure_tolerance = 0.1
    estimation_error = 0.01
    expected_iterates = np.log(1 / failure_tolerance) / estimation_error
    actual_iterates = compute_number_of_grover_iterates_for_amp_est(
        failure_tolerance, estimation_error
    )
    assert actual_iterates == expected_iterates


# Test the initialization of CoherentQuantumAmplitudeEstimation
def test_quantum_amplitude_estimation_initialization():
    qae = CoherentQuantumAmplitudeEstimation()
    assert qae.task_name == "estimate_amplitude"
    assert isinstance(qae.state_preparation_oracle, SubroutineModel)
    assert isinstance(qae.mark_subspace, SubroutineModel)


# Test setting requirements in CoherentQuantumAmplitudeEstimation
def test_quantum_amplitude_estimation_set_requirements():
    qae = CoherentQuantumAmplitudeEstimation()
    estimation_error = 0.01
    failure_tolerance = 0.1
    qae.set_requirements(
        estimation_error=estimation_error, failure_tolerance=failure_tolerance
    )
    assert qae.requirements["estimation_error"] == estimation_error
    assert qae.requirements["failure_tolerance"] == failure_tolerance


# Test populating requirements for subroutines in CoherentQuantumAmplitudeEstimation
def test_quantum_amplitude_estimation_populate_requirements_for_subroutines():
    qae = CoherentQuantumAmplitudeEstimation()
    estimation_error = 0.01
    failure_tolerance = 0.1
    qae.set_requirements(
        estimation_error=estimation_error, failure_tolerance=failure_tolerance
    )
    qae.populate_requirements_for_subroutines()
    assert qae.state_preparation_oracle.requirements["failure_tolerance"] is not None
    assert qae.mark_subspace.requirements["failure_tolerance"] is not None
