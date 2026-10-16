import pytest

import hierophant as hp


@pytest.fixture(scope="session")
def diabetes():
    return hp.datasets.load("diabetes")


@pytest.fixture(scope="session")
def cancer():
    return hp.datasets.load("breast_cancer")
