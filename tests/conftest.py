import warnings

import jax
import pytest

warnings.filterwarnings("ignore", category=SyntaxWarning)

pytest_plugins = ["evidence.plugin"]


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
