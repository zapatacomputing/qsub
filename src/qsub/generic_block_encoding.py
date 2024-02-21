import numpy as np
from typing import Optional
from .subroutine_model import SubroutineModel
from sympy import symbols
from dataclasses import dataclass

@dataclass
class GenericLinearSystemBlockEncodingData:
        failure_tolerance: float = 0.1
        kappa_P: float = 0.1,
        mu_P_A: float = 0.1
        A_stable: float = 0.01  

class GenericBlockEncoding(SubroutineModel):
    def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
        self.task_name = task_name
        self.requirements = requirements or {}
        self.number_of_times_called = None
        for attr, value in kwargs.items():
            if isinstance(value, SubroutineModel):
                setattr(self, attr, value)

    def set_requirements(self, *args, **kwargs):
        if args:
            raise TypeError(
                "The set_requirements method expects keyword arguments of the form argument=value."
            )
        self.requirements = kwargs

    def populate_requirements_for_subroutines(self):
        pass

    def get_subnormalization(self):
        # For a generic block encoding object, a symbol with the task name is returned
        # for the number of qubits
        return symbols(f"{self.task_name}_subnorm")


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

    def get_subnormalization(self):
        # For a generic block encoding object, a symbol with the task name is returned
        # for the number of qubits
        return symbols(f"{self.task_name}_condition_number")
