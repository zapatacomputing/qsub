from typing import Optional, Union
from dataclasses import dataclass, asdict, is_dataclass
from abc import ABC, abstractmethod


@dataclass
class SubtaskRequirements(ABC):
    failure_tolerance: float = None
    number_of_times_called: Optional[Union[float, int]] = None


class Subtask(ABC):
    """A Subtask is simply an object that holds SubtaskRequirements."""

    def __init__(
        self,
        requirements: SubtaskRequirements = None,
    ):
        self.requirements = requirements

    def set_subtask_requirements(self, requirements: SubtaskRequirements):
        self.requirements = requirements


class SubroutineModel(Subtask):
    """A SubroutineModel is a subtask endowed with the ability to model the servicing of
    that subtask by assigning requirements to its own (child) subtasks. These child subtasks
    are stored as attributes of the SubroutineModel. The key method of the SubroutineModel, which
    distinguishes it from a Subtask, is the assign_requirements_to_subtasks() method. This method
    uses the requirements of the SubroutineModel to assign requirements to its child subtasks.
    """

    def recursively_assign_requirements_to_all_subtasks(self):
        self.assign_requirements_to_subtasks()
        # If any of the subtasks are SubroutineModels, run their profiles as well.
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.recursively_assign_requirements_to_all_subtasks()

    def assign_requirements_to_subtasks(self):
        # Allocate failure tolerance
        failure_budget_allocation = self.allocate_failure_tolerance_budget()

        # Generate requirements for boring_task_1 and boring_task_2
        self.generate_requirements_for_subtasks(failure_budget_allocation)

    @abstractmethod
    def allocate_failure_tolerance_budget(self):
        pass

    # TODO: update name of this function
    @abstractmethod
    def generate_requirements_for_subtasks(self, failure_budget_allocation):
        pass


@dataclass
class FunTaskRequirements(SubtaskRequirements):
    fun_requirement: float = None
    task_name: str = "fun_task"


class FunSubroutineModel(SubroutineModel):
    def __init__(
        self,
        requirements: FunTaskRequirements = None,
        boring_task_1: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
        boring_task_2: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__(requirements=requirements)
        self.boring_task_1 = boring_task_1
        self.boring_task_2 = boring_task_2

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.5,
            "fraction_to_boring_task_1": 0.25,
            "fraction_to_boring_task_2": 0.25,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "boring_task_1_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_boring_task_1"
            ]
            * self.requirements.failure_tolerance,
            "boring_task_2_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_boring_task_2"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded
        assert (
            sum(failure_budget_allocation.values())
            <= self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def generate_requirements_for_subtasks(self, failure_budget_allocation):
        # Generate requirements for boring_task_1 and boring_task_2
        boring_task_1_requirements = BoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "boring_task_1_failure_tolerance"
            ]
        )
        boring_task_2_requirements = BoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "boring_task_2_failure_tolerance"
            ]
        )
        self.boring_task_1.set_subtask_requirements(boring_task_1_requirements)
        self.boring_task_2.set_subtask_requirements(boring_task_2_requirements)


@dataclass
class BoringTaskRequirements(SubtaskRequirements):
    boring_requirement: float = None
    task_name: str = "boring_task"


class BoringSubroutineModelA(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.6,
            "fraction_to_more_boring_task": 0.4,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "more_boring_task_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_more_boring_task"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded
        assert (
            sum(failure_budget_allocation.values())
            <= self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def generate_requirements_for_subtasks(self, failure_budget_allocation):
        # Generate requirements for more_boring_task
        more_boring_task_requirements = MoreBoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "more_boring_task_failure_tolerance"
            ]
        )
        self.more_boring_task.set_subtask_requirements(more_boring_task_requirements)


class BoringSubroutineModelB(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = Subtask(),
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {
            "fraction_consumed": 0.8,
            "fraction_to_more_boring_task": 0.2,
        }

        failure_budget_allocation = {
            "consumed_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_consumed"
            ]
            * self.requirements.failure_tolerance,
            "more_boring_task_failure_tolerance": fractional_failure_budget_allocation[
                "fraction_to_more_boring_task"
            ]
            * self.requirements.failure_tolerance,
        }
        # Check that budget is not exceeded
        assert (
            sum(failure_budget_allocation.values())
            <= self.requirements.failure_tolerance
        )

        return failure_budget_allocation

    def generate_requirements_for_subtasks(self, failure_budget_allocation):
        # Generate requirements for more_boring_task
        more_boring_task_requirements = MoreBoringTaskRequirements(
            failure_tolerance=failure_budget_allocation[
                "more_boring_task_failure_tolerance"
            ]
        )
        self.more_boring_task.set_subtask_requirements(more_boring_task_requirements)


@dataclass
class MoreBoringTaskRequirements(SubtaskRequirements):
    most_boring_requirement: float = None
    task_name: str = "more_boring_task"


fun_requirements = FunTaskRequirements(fun_requirement=0.1, failure_tolerance=0.01)

fun = FunSubroutineModel(requirements=fun_requirements)
fun.boring_task_1 = BoringSubroutineModelA()
fun.boring_task_2 = BoringSubroutineModelB()


fun.recursively_assign_requirements_to_all_subtasks()

print("fun requirements: ", asdict(fun.requirements))
