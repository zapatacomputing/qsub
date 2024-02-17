from typing import Optional
from sympy import symbols
from qsub.subroutine_model import SubroutineModel

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

    def set_requirements(
        self,
        failure_tolerance: float = None,
        kappa_P: float = None,
        mu_P_A: float = None,
        A_stable: float = None,
    ):
        args = locals()
        # Clean up the args dictionary before setting requirements
        args.pop("self")
        args = {
            k: v for k, v in args.items() if v is not None and not k.startswith("__")
        }
        # Initialize the requirements attribute if it doesn't exist
        if not hasattr(self, "requirements"):
            self.requirements = {}

        # Update the requirements with new values
        self.requirements.update(args)

        # Call the parent class's set_requirements method with the updated requirements
        super().set_requirements(**self.requirements)

    def populate_requirements_for_subroutines(self):
        pass

    def get_subnormalization(self):
        # For a generic block encoding object, a symbol with the task name is returned
        # for the number of qubits
        return symbols(f"{self.task_name}_condition_number")
