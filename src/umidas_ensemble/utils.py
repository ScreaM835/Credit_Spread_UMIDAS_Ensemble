from __future__ import annotations

from contextlib import contextmanager

try:
    from threadpoolctl import threadpool_limits  # type: ignore

    _HAVE_TPC = True
except Exception:
    threadpool_limits = None
    _HAVE_TPC = False


@contextmanager
def cpu_guard(n_threads: int = 2):
    """Limit BLAS/OpenMP thread usage within the context.

    If ``threadpoolctl`` is not installed, this is a no-op.
    """
    if _HAVE_TPC and threadpool_limits is not None:
        with threadpool_limits(int(n_threads)):
            yield
    else:
        yield
