"""
Dummy conftest.py for imputegaps.

If you don't know what this is for, just leave it empty.
Read more about conftest.py under:
- https://docs.pytest.org/en/stable/fixture.html
- https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import yaml

DEFAULT_SETTINGS = {
    "general": {
        "imputation": {"imputation_methods": ["pick", "dict"]},
    }
}


# import pytest
@pytest.fixture(scope="session")
def input_settings():
    """
    Input settings for the tests.

    This fixture returns a YAML string containing the default settings for the tests.
    """
    return yaml.dump(DEFAULT_SETTINGS)
