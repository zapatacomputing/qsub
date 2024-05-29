import numpy as np
from typing import Optional
from ...subroutine_model import SubroutineModel
import warnings
from qsub.utils import consume_fraction_of_error_budget
from data_classes import StatePreparationOracleData, MarkedSubspaceOracleData

class QuantumAmplitudeEstimation(SubroutineModel):
    def __init__(
        self,
        state_preparation_oracle: SubroutineModel,
        mark_subspace: SubroutineModel,
        task_name="estimate_amplitude",

    ):
        super().__init__(task_name)
        assert isinstance(state_preparation_oracle,SubroutineModel)
        assert isinstance(mark_subspace, SubroutineModel)
        self.state_preparation_oracle = state_preparation_oracle
        self.mark_subspace = mark_subspace


    def populate_requirements_for_subroutines(self):
        # Populate requirements for state_preparation_oracle and mark_subspace

        # Allocate failure tolerance
        # consumed_failure_tolerance = allocation * self.requirements["failure_tolerance"]
        # remaining_failure_tolerance = (
        #     self.requirements["failure_tolerance"] - consumed_failure_tolerance
        # )
        remaining_failure_tolerance = self.requirements["failure_tolerance"]

        (consumed_failure_tolerance, remaining_failure_tolerance) = (
            consume_fraction_of_error_budget(0.5, remaining_failure_tolerance)
        )

        # Set subroutine error budget allocation
        subroutine_error_budget_allocation = [0.5, 0.5]

        # Compute number of Grover iterates needed
        # If amplitude note provided, use worst-case value of 0.5
        if "amplitude" not in self.requirements:
            warnings.warn(
                "No amplitude provided. Using worst-case value of 0.5 for amplitude."
            )
            self.set_requirements(amplitude=0.5)

        number_of_grover_iterates = (
            compute_number_of_grover_iterates_for_coherent_quantum_amp_est(
                consumed_failure_tolerance,
                self.requirements["estimation_error"],
                amplitude=self.requirements["amplitude"],
            )
        )

        # Set number of times called to number of Grover iterates
        self.state_preparation_oracle.number_of_times_called = (
            2 * number_of_grover_iterates + 1
        )
        self.mark_subspace.number_of_times_called = 2 * number_of_grover_iterates

        StatePreparationOracleData.failure_tolerance= (subroutine_error_budget_allocation[0]/ 
            self.state_preparation_oracle.number_of_times_called)* remaining_failure_tolerance
        MarkedSubspaceOracleData.failure_tolerance = (subroutine_error_budget_allocation[1]
            / self.mark_subspace.number_of_times_called) * remaining_failure_tolerance

        self.state_preparation_oracle.set_requirements(StatePreparationOracleData)

        self.mark_subspace.set_requirements(MarkedSubspaceOracleData)

    def count_qubits(self):
        return self.state_preparation_oracle.count_qubits()


def compute_number_of_grover_iterates_for_coherent_quantum_amp_est(
    failure_tolerance, estimation_error, amplitude=0.5
):
    # Compute number of Grover iterates needed for standard amplitude amplification
    # from https://arxiv.org/abs/quant-ph/0005055 combined with Appendix C of https://arxiv.org/abs/quant-ph/9708016
    number_of_grover_iterates = (
        np.pi * np.sqrt(amplitude * (1 - amplitude)) / (2 * estimation_error)
    ) * (1 / (2 * failure_tolerance) + 1 / 2)

    return number_of_grover_iterates


class IterativeQuantumAmplitudeEstimationAlgorithm(SubroutineModel):
    """
    Subroutine model for the iterative quantum amplitude estimation algorithm
    as described in https://arxiv.org/abs/1912.05559.
    The algorithm is a variant of the quantum amplitude estimation algorithm
    where there quantum fourier transform is not needed, each circuit is run
    multiple times, and the number of Grover iterates used per circuit increases
    in each iteration.
    """

    def __init__(
        self,
        state_preparation_oracle: SubroutineModel = None,
        mark_subspace: SubroutineModel = None,
        task_name="estimate_amplitude",
    ):
        super().__init__(task_name)
        self.state_preparation_oracle = state_preparation_oracle
        self.mark_subspace = mark_subspace

    def populate_requirements_for_subroutines(self):
        # Populate requirements for state_preparation_oracle and mark_subspace

        # Allocate failure tolerance
        allocation = 0.5
        consumed_failure_tolerance = allocation * self.requirements["failure_tolerance"]
        remaining_failure_tolerance = (
            self.requirements["failure_tolerance"] - consumed_failure_tolerance
        )

        # Set subroutine error budget allocation
        subroutine_error_budget_allocation = [0.5, 0.5]

        # Compute number of Grover iterates needed
        number_of_grover_iterates = (
            compute_number_of_grover_iterates_per_circuit_for_iterative_amp_est(
                consumed_failure_tolerance, self.requirements["estimation_error"]
            )
        )

        # Set number of times called to number of Grover iterates
        self.state_preparation_oracle.number_of_times_called = (
            2 * number_of_grover_iterates
        )
        self.mark_subspace.number_of_times_called = number_of_grover_iterates

        StatePreparationOracleData.failure_tolerance=(subroutine_error_budget_allocation[0]
                    / self.state_preparation_oracle.number_of_times_called)* remaining_failure_tolerance
        MarkedSubspaceOracleData.failure_tolerance = (subroutine_error_budget_allocation[1]
                    / self.mark_subspace.number_of_times_called)* remaining_failure_tolerance

        self.state_preparation_oracle.set_requirements(StatePreparationOracleData)

        self.mark_subspace.set_requirements(MarkedSubspaceOracleData)

    def count_qubits(self):
        return self.state_preparation_oracle.count_qubits()


def compute_number_of_grover_iterates_per_circuit_for_iterative_amp_est(
    failure_tolerance, estimation_error
):
    # Compute number of Grover iterates needed for iterative amplitude estimation
    # From https://arxiv.org/abs/1912.05559
    number_of_grover_iterates = np.pi / (8 * estimation_error)

    return number_of_grover_iterates


def compute_number_of_samples_for_iterative_amp_est(
    failure_tolerance, estimation_error
):
    # Compute number of qae circuit samples needed for iterative amplitude estimation
    # From https://arxiv.org/abs/1912.05559
    number_of_samples = (32 / (1 - 2 * np.sin(np.pi / 14)) ** 2) * np.log(
        (2 * np.log2((np.pi / (4 * estimation_error)))) / failure_tolerance
    )

    return number_of_samples
