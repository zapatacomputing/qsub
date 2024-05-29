import numpy as np
from typing import Optional
from .subroutine_model import SubroutineModel
from sympy import symbols
from dataclasses import dataclass
 
class GenericLinearSystemBlockEncoding(SubroutineModel):
    def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
        self.task_name = task_name
        self.requirements = requirements or {}
        self.number_of_times_called = None
        for attr, value in kwargs.items():
            if isinstance(value, SubroutineModel):
                setattr(self, attr, value)

    def populate_requirements_for_subroutines(self):
        pass

    def get_condition_number(self):
        # For a generic linear system block encoding object, a symbol with the task name is returned
        # for the condition number
        return symbols(f"{self.task_name}_condition_number")

    def get_subnormalization(self):
        # For a generic block encoding object, a symbol with the task name is returned
        # for the subnormalization
        return symbols(f"{self.task_name}_subnormalization")
