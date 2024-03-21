from typing import Optional, Union
from dataclasses import dataclass, asdict, is_dataclass
from abc import ABC, abstractmethod


@dataclass
class SubtaskRequirements(ABC):
    failure_tolerance: float = None
    number_of_times_called: Optional[Union[float, int]] = None


class Subtask(ABC):
    """ A Subtask is simply an object that holds SubtaskRequirements.
    """
    def __init__(
        self,
        requirements: SubtaskRequirements = None,
    ):
        self.requirements = requirements

    def set_subtask_requirements(self,requirements:SubtaskRequirements):
        self.requirements = requirements

class SubroutineModel(Subtask):
    """ A SubroutineModel is a subtask endowed with the ability to model the servicing of 
        that subtask by assigning requirements to its own (child) subtasks. These child subtasks 
        are stored as attributes of the SubroutineModel. The key method of the SubroutineModel, which 
        distinguishes it from a Subtask, is the assign_requirements_to_subtasks() method. This method
        uses the requirements of the SubroutineModel to assign requirements to its child subtasks.
    """
    def __init__(
        self,
    ):
        super().__init__()

    def recursively_assign_requirements_to_all_subtasks(self):
        self.assign_requirements_to_subtasks()
        # If any of the subtasks are SubroutineModels, run their profiles as well.
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.recursively_assign_requirements_to_all_subtasks()

    @abstractmethod
    def assign_requirements_to_subtasks(self):
        pass



@dataclass
class FunTaskRequirements(SubtaskRequirements):
    fun_requirement: float = None
    task_name: str = "fun_task"


class FunSubroutineModel(SubroutineModel):
    def __init__(
        self,
        boring_task_1: Optional[Union[Subtask, SubroutineModel]] = None,
        boring_task_2: Optional[Union[Subtask, SubroutineModel]] = None,
    ):
        super().__init__()
        self.boring_task_1 = boring_task_1
        self.boring_task_2 = boring_task_2

    def assign_requirements_to_subtasks(self):
        # Allocate failure tolerance
        failure_budget_allocation = self.allocate_failure_tolerance_budget()
        
        # Generate requirements for boring_task_1 and boring_task_2
        subtask_requirements = self.generate_requirements_for_subroutines(failure_budget_allocation)

        # Assign requirements to boring_task_1 and boring_task_2
        self.set_subtask_requirements(subtask_requirements)
        
    def allocate_failure_tolerance_budget(self):
        # Allocate failure tolerance
        fractional_failure_budget_allocation = {"fraction_consumed": 0.5,
                                                "fraction_to_boring_task_1": 0.25,
                                                "fraction_to_boring_task_2": 0.25}
        fraction_of_consumed_failure_tolerance_fraction = 0.5

        # Check that budget is not exceeded
        

        return 

@dataclass
class BoringTaskRequirements(SubtaskRequirements):
    boring_requirement: float = None
    task_name: str = "boring_task"


class BoringSubroutineModelA(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = None,
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def populate_requirements_for_subroutines(self):

        # Consume failure rate
        remaining_failure_tolerance = self.requirements.failure_tolerance / 2

        self.more_boring_task.requirements = MoreBoringTaskRequirements(
            failure_tolerance=remaining_failure_tolerance
        )

        return


class BoringSubroutineModelB(SubroutineModel):
    def __init__(
        self,
        more_boring_task: Optional[Union[Subtask, SubroutineModel]] = None,
    ):
        super().__init__()
        self.more_boring_task = more_boring_task

    def populate_requirements_for_subroutines(self):
        return super().populate_requirements_for_subroutines()


@dataclass
class MoreBoringTaskRequirements(SubtaskRequirements):
    most_boring_requirement: float = None
    task_name: str = "more_boring_task"


fun.subroutine = FunSubroutineModel()
fun.boring_task_1. = BoringSubroutineModelA()
fun.boring_task_2 = BoringSubroutineModelB()

fun_requirements = FunTaskRequirements(fun_requirement=0.1, failure_tolerance=0.01)

fun.run_profile()
