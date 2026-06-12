from semcom.utils.project import get_project_name

def test_project_name() -> None:
    assert get_project_name() == "semcom-playground"