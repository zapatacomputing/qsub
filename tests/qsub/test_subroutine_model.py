import pytest
from dataclasses import dataclass, asdict
from qsub.subroutine_model import SubroutineModel
from typing import Optional


@pytest.fixture()
def subroutinemodel():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
            super().__init__(task_name, requirements, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()

    return MockSubroutineModel(task_name="TestTask")

@pytest.fixture()
def child_subroutine_1():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
            super().__init__(task_name, requirements, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()
    return MockSubroutineModel(task_name="Task1")

@pytest.fixture()
def child_subroutine_2():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
            super().__init__(task_name, requirements, **kwargs)
        def populate_requirements_for_subroutines(self):
            return super().populate_requirements_for_subroutines()
    return MockSubroutineModel(task_name="Task2")

@pytest.fixture()
def mock_data_class(child_subroutine_1, child_subroutine_2):
    @dataclass
    class SubroutineModelData:
        field_1: float = 0
        field_2: float = 1
        subroutine_1: SubroutineModel = child_subroutine_1
        subroutine_2: SubroutineModel = child_subroutine_2
    return SubroutineModelData()
      
def test_initializing_subroutine(subroutinemodel, mock_data_class):
    sub = subroutinemodel
    sub.set_requirements(mock_data_class)
    assert sub.task_name == "TestTask"
    assert sub.requirements != {}
    assert sub.number_of_times_called is None
    assert 'field_1' in sub.requirements.keys()
    assert 'field_2' in sub.requirements.keys()
    assert 'subroutine_1' in sub.requirements.keys()
    assert isinstance(sub.requirements["subroutine_1"], SubroutineModel)

def test_count_subroutines(subroutinemodel, mock_data_class):
    sub = subroutinemodel
    sub.set_requirements(mock_data_class)
    counts = sub.count_subroutines()
    print("number of counts: ", counts)
    assert len(counts) == len({"TestTask": 1, " Task1": 1, " Task2": 1})
    assert counts == {'TestTask': 1, 'Task1': 1, 'Task2': 1}


# def test_print_profile(capsys):
#     sub = SubroutineModel(task_name="Main", requirements={"req1": "val1"})
#     sub.print_profile()

#     captured = capsys.readouterr()
#     assert "Subroutine: SubroutineModel (Task: Main)" in captured.out
#     assert "Requirements:" in captured.out
#     assert "req1: val1" in captured.out
