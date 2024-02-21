import numpy as np
from typing import Optional
from ...subroutine_model import SubroutineModel
import warnings
from qsub.utils import create_data_class_from_dict


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
        allocation = 0.5
        consumed_failure_tolerance = allocation * self.requirements["failure_tolerance"]
        remaining_failure_tolerance = (
            self.requirements["failure_tolerance"] - consumed_failure_tolerance
        )

        # Set subroutine error budget allocation
        subroutine_error_budget_allocation = [0.5, 0.5]

        # Compute number of Grover iterates needed
        number_of_grover_iterates = (
            compute_number_of_grover_iterates_for_quantum_amp_est(
                consumed_failure_tolerance, self.requirements["estimation_error"]
            )
        )

        # Set number of times called to number of Grover iterates
        self.state_preparation_oracle.number_of_times_called = (
            2 * number_of_grover_iterates + 1
        )
        self.mark_subspace.number_of_times_called = 2 * number_of_grover_iterates

        state_preparation_oracle_requirements_dict= {"failure_tolerance":subroutine_error_budget_allocation[0]
            / self.state_preparation_oracle.number_of_times_called
            * remaining_failure_tolerance}

        mark_subspace_requirements_dict = {"failure_tolerance":subroutine_error_budget_allocation[1]
            / self.mark_subspace.number_of_times_called
            * remaining_failure_tolerance, }

        self.state_preparation_oracle.set_requirements(create_data_class_from_dict(state_preparation_oracle_requirements_dict))

        self.mark_subspace.set_requirements(create_data_class_from_dict(mark_subspace_requirements_dict))

    def count_qubits(self):
        return self.state_preparation_oracle.count_qubits()


def compute_number_of_grover_iterates_for_quantum_amp_est(estimation_error, amplitude):
    # Compute number of Grover iterates needed for oblivious amplitude amplification
    # From https://arxiv.org/abs/quant-ph/0005055
    number_of_grover_iterates = (
        np.pi * np.sqrt(amplitude * (1 - amplitude)) / (2 * estimation_error)
    )

    return number_of_grover_iterates


class IterativeQuantumAmplitudeEstimation(SubroutineModel):
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
            compute_number_of_grover_iterates_for_iterative_amp_est(
                consumed_failure_tolerance, self.requirements["estimation_error"]
            )
        )

        # Set number of times called to number of Grover iterates
        self.state_preparation_oracle.number_of_times_called = number_of_grover_iterates
        self.mark_subspace.number_of_times_called = number_of_grover_iterates

        state_preparation_oracle_requirements_dict= {"failure_tolerance":subroutine_error_budget_allocation[0]
                    / self.state_preparation_oracle.number_of_times_called
                    * remaining_failure_tolerance}

        mark_subspace_requirements_dict = {"failure_tolerance":subroutine_error_budget_allocation[1]
                    / self.mark_subspace.number_of_times_called
                    * remaining_failure_tolerance, }

        self.state_preparation_oracle.set_requirements(create_data_class_from_dict(state_preparation_oracle_requirements_dict))

        self.mark_subspace.set_requirements(create_data_class_from_dict(mark_subspace_requirements_dict))

    def count_qubits(self):
        return self.state_preparation_oracle.count_qubits()


def compute_number_of_grover_iterates_for_iterative_amp_est(
    estimation_error, amplitude
):
    # Compute number of Grover iterates needed for oblivious amplitude amplification
    # From https://arxiv.org/abs/quant-ph/0005055
    number_of_grover_iterates = (
        np.pi * np.sqrt(amplitude * (1 - amplitude)) / (2 * estimation_error)
    )

    return number_of_grover_iterates
