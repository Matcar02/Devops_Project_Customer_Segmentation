import pytest
import pandas as pd
import os

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", required=True,
                     help="File path for the CSV file")
    parser.addoption("--expected_columns", action="store",
                     help="Comma-separated list of expected column names")

@pytest.fixture(scope='session')
def data(request):
    csv_path = request.config.getoption("--csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at the provided path: {csv_path}")

    df = pd.read_csv(csv_path)
    return df

@pytest.fixture(scope='session')
def expected_columns(request):
    expected_columns = request.config.getoption("--expected_columns")
    if not expected_columns:
        pytest.fail("You must provide --expected_columns")
    return expected_columns.split(',')
