"""Parallel evaluation plumbing for DEAP and QDax trainers.

This module provides a unified interface over the following execution backends:

* a ``_SerialExecutor`` that mimics :class:`concurrent.futures.Executor`
  for the ``num_workers <= 1`` case, so trainer call sites stay
  single-path.
* :class:`concurrent.futures.ProcessPoolExecutor` for real
  multi-process parallelism.
* optional SCOOP and Dask backends — both are
  soft-imported; importing this module never pulls them in.

All executors implement the same minimal protocol:

* ``__enter__`` / ``__exit__`` — context-managed lifetime so trainers
  can always wrap the pool in ``with make_executor(...) as ex:``.
* ``map(fn, iterable) -> Iterator`` — **ordered** map; the callers rely
  on result order matching input order to zip fitnesses back onto
  individuals.

Design notes
------------

* The worker-side initialisation pattern used throughout the project
  is **lazy module-global caching** inside the worker function itself
  (see :func:`deap_pettingzoo.integration.deap_integration._worker_evaluate`
  and
  :func:`deap_pettingzoo.contrib.qdax.scoring_numpy._numpy_rollout`).
  This is needed because SCOOP does not expose a per-worker
  ``initializer`` hook. Using the same pattern for
  ``ProcessPoolExecutor`` costs nothing and keeps the code paths
  unified.
* Payloads crossing the process boundary are **pure numpy arrays +
  small dataclasses / primitives**. We never ship a live ``Env``,
  ``PettingZooWrapper``, ``Evaluator``, or ``jax`` object; each is
  rebuilt on the worker from picklable metadata. This keeps the
  default pickler happy without pulling in ``cloudpickle`` / ``dill``.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple


__all__ = [
    "resolve_num_workers",
    "make_executor",
    "parallel_map",
    "SUPPORTED_BACKENDS",
]


SUPPORTED_BACKENDS: Tuple[str, ...] = ("process", "scoop", "dask")


def resolve_num_workers(requested: Optional[int]) -> int:
    """Normalise a ``num_workers`` request.

    Args:
        requested: ``None``, ``0``, or any ``int``.

    Returns:
        * ``1`` for ``None``, ``0``, or ``1`` (serial).
        * Otherwise the requested value, capped at
          ``os.cpu_count()`` (with a :class:`UserWarning` when
          capped).

    Raises:
        ValueError: if ``requested`` is negative.
    """
    if requested is None:
        return 1
    n = int(requested)
    if n < 0:
        raise ValueError(
            f"num_workers must be >= 0 (or None), got {requested!r}."
        )
    cpu = os.cpu_count() or 1
    if n == 0:
        return 1
    if n == 1:
        return 1
    if n > cpu:
        warnings.warn(
            f"num_workers={n} exceeds os.cpu_count()={cpu}; capping to {cpu}.",
            UserWarning,
            stacklevel=2,
        )
        return cpu
    return n


class _SerialExecutor:
    """Drop-in serial executor.

    Implements just enough of :class:`concurrent.futures.Executor` for
    the callers in this project (``map`` + context manager).
    """

    def __enter__(self) -> "_SerialExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
    ) -> Iterator[Any]:
        return map(fn, *iterables)

    def shutdown(self, wait: bool = True) -> None:  # pragma: no cover
        return None


class _ScoopExecutor:
    """SCOOP-backed executor.

    Worker pool is owned by the ``python -m scoop`` launcher, so
    ``num_workers`` is advisory. ``__enter__`` / ``__exit__`` are no-ops.

    Raises:
        ImportError: on construction if ``scoop`` is not installed.
    """

    def __init__(self, num_workers: int) -> None:
        try:
            from scoop import futures as _scoop_futures  # noqa: F401
        except ImportError as e:  # pragma: no cover - soft import
            raise ImportError(
                "parallel_backend='scoop' requires the 'scoop' package. "
                "Install with: pip install 'deap-pettingzoo[scoop]' "
                "(or pip install scoop). Launch scripts with "
                "`python -m scoop your_script.py`."
            ) from e
        self._num_workers = num_workers  # advisory only

    def __enter__(self) -> "_ScoopExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
    ) -> Iterator[Any]:
        from scoop import futures as _scoop_futures

        # scoop.futures.map is ordered and eager-iterable.
        return iter(_scoop_futures.map(fn, *iterables))


class _DaskExecutor:
    """Dask distributed executor.

    Accepts an injected ``dask.distributed.Client`` (lifetime not owned
    by us) or builds a local one (closed on ``__exit__``).
    """

    def __init__(
        self,
        num_workers: int,
        *,
        client: Optional[Any] = None,
    ) -> None:
        try:
            from dask.distributed import Client  # noqa: F401
        except ImportError as e:  # pragma: no cover - soft import
            raise ImportError(
                "parallel_backend='dask' requires the 'dask[distributed]' "
                "package. Install with: pip install "
                "'deap-pettingzoo[dask]' (or pip install dask[distributed])."
            ) from e

        self._owned = client is None
        if client is None:
            from dask.distributed import Client

            # One thread per worker: the project's worker-side caches
            # (``_WORKER_CACHE`` and friends) and the rebuilt envs are not
            # thread-safe — they assume one task at a time per process, like
            # ``ProcessPoolExecutor`` and SCOOP. Dask's multi-threaded default
            # would let concurrent tasks corrupt a shared cached env. Inject
            # your own ``Client`` if you need a different layout.
            client = Client(
                n_workers=max(2, num_workers),
                threads_per_worker=1,
                processes=True,
                dashboard_address=None,
            )
        self._client = client

    def __enter__(self) -> "_DaskExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owned:
            try:
                self._client.close()
            finally:
                self._client = None

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
    ) -> Iterator[Any]:
        # Materialise to preserve ordering semantics; client.map + gather returns results in submission order.
        materialised = [list(it) for it in iterables]
        futures = self._client.map(fn, *materialised, pure=False)
        results = self._client.gather(futures)
        return iter(results)


def make_executor(
    num_workers: Optional[int],
    *,
    backend: str = "process",
    initializer: Optional[Callable[..., None]] = None,
    initargs: Tuple[Any, ...] = (),
    dask_client: Optional[Any] = None,
) -> Any:
    """Build a context-managed executor.

    Args:
        num_workers: Requested worker count. Normalised via
            :func:`resolve_num_workers`; ``None``/``0``/``1`` yield a
            :class:`_SerialExecutor`.
        backend: One of ``"process"``, ``"scoop"``, ``"dask"``.
        initializer: Optional per-worker init callable
            (``ProcessPoolExecutor`` only; ignored by SCOOP/Dask,
            which have no such hook — worker-side code must self-init
            lazily).
        initargs: Args forwarded to ``initializer``.
        dask_client: Optional pre-built
            ``dask.distributed.Client`` (Dask backend only).

    Returns:
        A context-managed executor with an ordered ``.map(fn, iter)``
        method.

    Raises:
        ValueError: if ``backend`` is not in :data:`SUPPORTED_BACKENDS`.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown parallel_backend {backend!r}. "
            f"Expected one of {SUPPORTED_BACKENDS}."
        )

    n = resolve_num_workers(num_workers)
    # Serial fast-path applies regardless of backend choice.
    if n <= 1:
        return _SerialExecutor()

    if backend == "process":
        return ProcessPoolExecutor(
            max_workers=n,
            initializer=initializer,
            initargs=initargs,
        )

    if backend == "scoop":
        return _ScoopExecutor(n)

    if backend == "dask":
        return _DaskExecutor(n, client=dask_client)

    raise AssertionError("unreachable")  # pragma: no cover


def parallel_map(
    fn: Callable[..., Any],
    items: Iterable[Any],
    *,
    executor: Any,
) -> list:
    """Ordered map over ``items`` using ``executor``.

    Materialises the iterator to a list because callers zip the
    results back onto populations.
    """
    return list(executor.map(fn, items))
