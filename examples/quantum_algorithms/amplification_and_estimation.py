from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    CoherentQuantumAmplitudeEstimation,
)


def example_quantum_amplitude_estimation():
    amp_est = CoherentQuantumAmplitudeEstimation()

    estimation_error = 0.001
    failure_tolerance = 0.01

    amp_est.set_requirements(
        estimation_error=estimation_error,
        failure_tolerance=failure_tolerance,
    )

    # Run the profile for this subroutine
    amp_est.run_profile()
    amp_est.print_profile()

    print("qubits =", amp_est.count_qubits())

    return print(amp_est.count_subroutines())


example_quantum_amplitude_estimation()
