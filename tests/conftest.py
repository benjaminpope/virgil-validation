import warnings

import jax
import pytest

warnings.filterwarnings("ignore", category=SyntaxWarning)


def pytest_configure(config):
    config.addinivalue_line("markers", "x64: run virgil in float64")
    config.addinivalue_line("markers", "float32: run virgil in float32")
    config.addinivalue_line("markers", "slow: long simulation")
    config.addinivalue_line(
        "markers", "external: needs another package (PMOIRED, CANDID)"
    )


@pytest.fixture(autouse=True)
def _precision(request):
    """float64 for tests marked x64 (and any unmarked test that simulates
    with dLux); float32 only where asked."""
    if request.node.get_closest_marker("float32"):
        with jax.enable_x64(False):
            yield
    else:
        with jax.enable_x64(True):
            yield
