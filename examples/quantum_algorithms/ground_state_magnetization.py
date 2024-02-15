from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_estimation import (
    CoherentQuantumAmplitudeEstimation,
)
from qsub.quantum_algorithms.quantum_simulation.ground_state_preparation.qsp_gsp import (
    LinTongGroundStatePreparation,
)


# Initialize the subroutines
ground_state = LinTongGroundStatePreparation()
ground_state.set_requirements(hamiltonian_gap=0.1, initial_state_overlap=0.9)
amplitude_estimation = CoherentQuantumAmplitudeEstimation(
    state_preparation_oracle=ground_state
)

amplitude_estimation.set_requirements(
    estimation_error=0.001,
    failure_tolerance=0.01,
)

# Run the profile for this subroutine
amplitude_estimation.run_profile()
amplitude_estimation.print_profile()

graph = amplitude_estimation.display_hierarchy()
graph.view(cleanup=True)
