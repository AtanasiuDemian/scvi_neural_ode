import shutil

import pytest


@pytest.fixture(scope="session")
def save_path(tmp_path_factory):
    dir = tmp_path_factory.mktemp("temp_data", numbered=False)
    path = str(dir)
    shutil.copytree("tests/test_models", path, dirs_exist_ok=True)
    yield path + "/"
    shutil.rmtree(str(tmp_path_factory.getbasetemp()))
