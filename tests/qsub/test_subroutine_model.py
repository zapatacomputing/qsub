import pytest
from dataclasses import dataclass
from qsub.subroutine_model import SubroutineModel
from typing import Optional


@pytest.fixture()
def subroutinemodel():
    class MockSubroutineModel(SubroutineModel):
        def __init__(self, task_name: str, **kwargs):
            super().__init__(task_name, **kwargs)
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
def mock_data_class():
    @dataclass
    class SubroutineModelData:
        field_1: float = 0
        field_2: float = 1
        failure_tolerance: float = 0.01
    return SubroutineModelData()
      
def test_initializing_subroutine(subroutinemodel, mock_data_class):
    sub = subroutinemodel
    sub.set_requirements(mock_data_class)
    assert sub.task_name == "TestTask"
    assert sub.requirements != {}
    assert sub.number_of_times_called is None
    assert 'field_1' in sub.requirements.keys()
    assert 'field_2' in sub.requirements.keys()

def test_count_subroutines(subroutinemodel, child_subroutine_1 , child_subroutine_2):
    sub = subroutinemodel
    sub.sub1 = child_subroutine_1
    sub.sub2 = child_subroutine_2
    counts = sub.count_subroutines()
    print("number of counts: ", counts)
    assert len(counts) == len({"TestTask": 1, " Task1": 1, " Task2": 1})
    assert counts == {'TestTask': 1, 'Task1': 1, 'Task2': 1}


def test_print_profile(capsys,subroutinemodel, mock_data_class):
    sub = subroutinemodel
    sub.set_requirements(mock_data_class)
    sub.print_profile()

    captured = capsys.readouterr()
    print(captured)
    assert "Subroutine: MockSubroutineModel (Task: TestTask)" in captured.out
    assert "Requirements:" in captured.out
    assert "field_1: 0" in captured.out
