from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    CoherentQuantumAmplitudeEstimation,
)
from qsub.quantum_algorithms.general_quantum_algorithms.time_evolution import (
    QSPTimeEvolution,
)
from qsub.utils import consume_fraction_of_error_budget

from qsub.quantum_gates.multi_control_gates import GidneyMultiControlledZGate
from qsub.subroutine_model import SubroutineModel
from qsub.generic_block_encoding import GenericBlockEncoding
from typing import Optional
from qsub.subroutine_model import SubroutineModel
from qsub.quantum_algorithms.quantum_simulation.correlation_functions import (
    CorrelationFunctionUnitary,
    DynamicCorrelationFunctionEstimation,
)
from sympy import symbols, Max


# Subroutine that uses amplitude estimation to estimate the dynamic correlation function at several times


def example_fermi_hubbard_correlation_function():

    # Uses quantum amplitude estimation to estimate
    # the dynamic correlation function for a specific time of the Fermi-Hubbard model
    # <0|U_I' U(-t) A U(t) B U_I|0>
    # Re(x) -> QAE -> <0|<+| c-(U_I' U(-t) A U(t) B U_I)|+>|0> = 0.5(1+Re(x))
    # Im(x) -> QAE -> <0|<+i| c-(U_I' U(-t) A U(t) B U_I)|+i>|0> = 0.5(1+Im(x))

    evolution_time = 3.0
    number_of_qubits = 100
    estimation_error = 0.001
    failure_tolerance = 0.01
    amplitude = 0.5

    time_evolution = QSPTimeEvolution()

    correlation_function_unitary = CorrelationFunctionUnitary(
        time_evolution=time_evolution
    )
    correlation_function_unitary.set_requirements(
        evolution_time=evolution_time,
    )
    mark_all_zero = GidneyMultiControlledZGate()
    mark_all_zero.set_requirements(
        number_of_controls=number_of_qubits - 1,
    )

    amp_est_real = CoherentQuantumAmplitudeEstimation(
        state_preparation_oracle=correlation_function_unitary,
        mark_subspace=mark_all_zero,
    )
    amp_est_imag = CoherentQuantumAmplitudeEstimation(
        state_preparation_oracle=correlation_function_unitary,
        mark_subspace=mark_all_zero,
    )
    amp_est_real.set_requirements(amplitude=0.5)
    amp_est_imag.set_requirements(amplitude=0.5)

    corr_est = DynamicCorrelationFunctionEstimation(
        estimate_real_part_of_correlation_function=amp_est_real,
        estimate_imaginary_part_of_correlation_function=amp_est_imag,
    )

    corr_est.set_requirements(
        estimation_error=estimation_error,
        failure_tolerance=failure_tolerance,
        evolution_time=evolution_time,
    )

    # Run the profile for this subroutine
    corr_est.run_profile()
    corr_est.print_profile()

    print("qubits =", corr_est.count_qubits())

    graph = corr_est.display_hierarchy()
    graph.view(cleanup=True)

    return print(corr_est.count_subroutines())


example_fermi_hubbard_correlation_function()
