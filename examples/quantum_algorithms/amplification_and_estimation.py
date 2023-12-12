import numpy as np
from qsub.subroutine_model import SubroutineModel


from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    ObliviousAmplitudeAmplification,
    compute_number_of_grover_iterates_for_obl_amp,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    QuantumAmplitudeEstimation,
    compute_number_of_grover_iterates_for_amp_est,
)

def test_quantum_amplitude_estimation():
    amp_est = QuantumAmplitudeEstimation()

    estimation_error = 0.001
    failure_tolerance = 0.01

    amp_est.set_requirements(
        estimation_error=estimation_error,
        failure_tolerance=failure_tolerance,
    )

    # Run the profile for this subroutine
    amp_est.run_profile()
    amp_est.print_profile()

    return print(amp_est.count_subroutines())


test_quantum_amplitude_estimation()