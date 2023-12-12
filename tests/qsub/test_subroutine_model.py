import pytest

from src.qsub.subroutine_model import SubroutineModel

# Why does the following not work?
# from qsub.subroutine_model import (
#     SubroutineModel,
# )  # Adjust the import to where your class is located


def test_init():
    sub = SubroutineModel(task_name="TestTask")
    assert sub.task_name == "TestTask"
    assert sub.requirements == {}
    assert sub.number_of_times_called is None

    sub_with_reqs = SubroutineModel(task_name="TestTask", requirements={"req1": "val1"})
    assert sub_with_reqs.requirements == {"req1": "val1"}


def test_set_requirements():
    sub = SubroutineModel(task_name="TestTask")
    sub.set_requirements(req1="val1")
    assert sub.requirements == {"req1": "val1"}

    with pytest.raises(TypeError):
        sub.set_requirements("invalid_arg")


def test_count_subroutines():
    sub = SubroutineModel(task_name="Main")
    sub.sub1 = SubroutineModel(task_name="Sub1")
    sub.sub2 = SubroutineModel(task_name="Sub2")

    counts = sub.count_subroutines()
    assert counts == {"Main": 1, "Sub1": 1, "Sub2": 1}


def test_print_profile(capsys):
    sub = SubroutineModel(task_name="Main", requirements={"req1": "val1"})
    sub.print_profile()

    captured = capsys.readouterr()
    assert "Subroutine: SubroutineModel (Task: Main)" in captured.out
    assert "Requirements:" in captured.out
    assert "req1: val1" in captured.out
