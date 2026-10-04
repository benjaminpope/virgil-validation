import warnings

import jax
import pytest

warnings.filterwarnings("ignore", category=SyntaxWarning)

# float64 from import time, so that module- and session-scoped fixtures (built
# before the function-scoped context below) are in float64 too; tests marked
# float32 switch it off locally.
jax.config.update("jax_enable_x64", True)


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
