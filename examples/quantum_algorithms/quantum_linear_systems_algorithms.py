import numpy as np
import matplotlib.pyplot as plt
from qsub.utils import consume_fraction_of_error_budget


from qsub.quantum_algorithms.general_quantum_algorithms.linear_systems import (
    get_taylor_qlsa_num_block_encoding_calls,
)
from qsub.quantum_algorithms.general_quantum_algorithms.amplitude_amplification import (
    compute_number_of_grover_iterates_for_obl_amp,
)


def make_psi_quantum_plot():
    # Define the range for kappa
    kappa_values = np.linspace(
        100, 10000000, 50
    )  # Kappa starts from sqrt(12) as per the theorem
    epsilon = 1e-10  # Given epsilon value
    alpha = 1  # Given alpha value

    # Compute Q* for each kappa
    Q_star_values = [
        get_taylor_qlsa_num_block_encoding_calls(epsilon, alpha, kappa)[0] / kappa
        for kappa in kappa_values
    ]

    # Generate the plot
    plt.figure(figsize=(12, 6))
    plt.plot(kappa_values, Q_star_values, label="Our QLSA Algorithm", color="blue")

    plt.title("Query Count of Our QLSA Algorithm in Units of Kappa")
    plt.xlabel("Kappa")
    plt.ylabel("Query Count per kappa")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()


make_psi_quantum_plot()
