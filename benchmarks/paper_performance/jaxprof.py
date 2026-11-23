"""Opt-in JAX profiling helpers (jax-profiling skill). Enable by setting JAX_PROFILE_DIR; otherwise a no-op."""
import contextlib
import os
import socket
import time

import jax


def profile_dir(tag="run"):
    root = os.environ.get("JAX_PROFILE_DIR")
    if not root:
        return None
    job = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M%S"))
    return os.path.join(root, f"{tag}_{job}", f"proc{os.environ.get('SLURM_PROCID', '0')}")


@contextlib.contextmanager
def maybe_profile(tag="run", python_tracer=False, host_level=2, perfetto=False):
    d = profile_dir(tag)
    if d is None:
        yield None
        return
    opts = jax.profiler.ProfileOptions()
    opts.python_tracer_level = int(python_tracer)
    opts.host_tracer_level = host_level
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "env.txt"), "w") as f:
        f.write(f"host={socket.gethostname()}\njax={jax.__version__}\ndevices={jax.devices()}\n"
                f"XLA_FLAGS={os.environ.get('XLA_FLAGS', '')}\n"
                f"JAX_DEFAULT_MATMUL_PRECISION={os.environ.get('JAX_DEFAULT_MATMUL_PRECISION', '')}\n")
    with jax.profiler.trace(d, create_perfetto_trace=perfetto, profiler_options=opts):
        yield d


def step(i, name="step"):
    return jax.profiler.StepTraceAnnotation(name, step_num=i)
