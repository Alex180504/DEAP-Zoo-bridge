"""Parallel evaluation plumbing for DEAP and QDax trainers.

This module provides a unified interface over the following execution backends:

* a ``_SerialExecutor`` that mimics :class:`concurrent.futures.Executor`
  for the ``num_workers <= 1`` case, so trainer call sites stay
  single-path.
* :class:`concurrent.futures.ProcessPoolExecutor` for real
  multi-process parallelism.
* an optional Dask backend — soft-imported; importing this module
  never pulls it in. Supports both an auto-built ``LocalCluster``
  (single machine) and **remote clusters** reached through a scheduler
  address / scheduler file (HPC).

All executors implement the same minimal protocol:

* ``__enter__`` / ``__exit__`` — context-managed lifetime so trainers
  can always wrap the pool in ``with make_executor(...) as ex:``.
* ``map(fn, iterable) -> Iterator`` — **ordered** map; the callers rely
  on result order matching input order to zip fitnesses back onto
  individuals.

Remote Dask clusters
--------------------

The remote path is driven by one knob — a *scheduler address* — resolved
with the following precedence (centralised in :func:`make_executor`):

1. an injected ``dask_client`` (power users; lifetime stays theirs),
2. the ``scheduler_address`` argument,
3. the :data:`DASK_SCHEDULER_ENV_VAR` (``DEAP_DASK_SCHEDULER``)
   environment variable — the Slurm ergonomic: the sbatch script exports
   it and unmodified driver scripts just work,
4. an auto-built local cluster (previous behavior).

Address semantics: a value containing ``://`` (e.g.
``tcp://10.0.0.1:8786``) is connected to directly; anything else is
treated as a **scheduler-file path** (the canonical dask-on-Slurm
pattern over a shared filesystem).

When a remote client/address is in play, *remote intent wins*: the
``num_workers <= 1`` serial fast-path is skipped and ``num_workers`` is
not capped to the driver's CPU count — it becomes advisory metadata
(the cluster decides the real worker count).

Code shipped to remote workers must be importable there: the library
version must match the driver's (checked, warn-only, on connect) and
worker functions must not live in the driver script's ``__main__``
module (rejected with a clear error).

Design notes
------------

* The worker-side initialisation pattern used throughout the project
  is **lazy module-global caching** inside the worker function itself
  (see :func:`deap_pettingzoo.integration.deap_integration._worker_evaluate`
  and
  :func:`deap_pettingzoo.contrib.qdax.scoring_numpy._numpy_rollout`).
  This is needed because Dask does not expose a per-worker
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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "resolve_num_workers",
    "resolve_scheduler_address",
    "make_executor",
    "parallel_map",
    "SUPPORTED_BACKENDS",
    "DASK_SCHEDULER_ENV_VAR",
    "DASK_CONNECT_TIMEOUT_ENV_VAR",
]


SUPPORTED_BACKENDS: Tuple[str, ...] = ("process", "dask")

#: Environment variable holding a Dask scheduler address (``tcp://...``)
#: or scheduler-file path. Exported by HPC launch scripts so unmodified
#: driver scripts pick up the cluster automatically.
DASK_SCHEDULER_ENV_VAR = "DEAP_DASK_SCHEDULER"

#: Environment variable overriding the remote-connect timeout (seconds)
#: used both for reaching the scheduler and for waiting on the first worker.
DASK_CONNECT_TIMEOUT_ENV_VAR = "DEAP_DASK_CONNECT_TIMEOUT"

#: Default remote-connect timeout in seconds.
_DEFAULT_CONNECT_TIMEOUT_S = 120.0


def resolve_num_workers(requested: Optional[int], *, cap_to_cpu: bool = True) -> int:
    """Normalise a ``num_workers`` request.

    Args:
        requested: ``None``, ``0``, or any ``int``.
        cap_to_cpu: Cap the result at ``os.cpu_count()``. Pass ``False``
            when the workers live on a remote cluster and the driver's
            CPU count is irrelevant (a 64-worker cluster run must not be
            silently shrunk to the login node's cores).

    Returns:
        * ``1`` for ``None``, ``0``, or ``1`` (serial).
        * Otherwise the requested value, capped at
          ``os.cpu_count()`` when ``cap_to_cpu`` is ``True`` (with a
          :class:`UserWarning` when capped).

    Raises:
        ValueError: if ``requested`` is negative.
    """
    if requested is None:
        return 1
    n = int(requested)
    if n < 0:
        raise ValueError(f"num_workers must be >= 0 (or None), got {requested!r}.")
    if n == 0:
        return 1
    if n == 1:
        return 1
    cpu = os.cpu_count() or 1
    if cap_to_cpu and n > cpu:
        warnings.warn(
            f"num_workers={n} exceeds os.cpu_count()={cpu}; capping to {cpu}.",
            UserWarning,
            stacklevel=2,
        )
        return cpu
    return n


def resolve_scheduler_address(
    scheduler_address: Optional[str] = None,
) -> Optional[str]:
    """Resolve the effective remote Dask scheduler address.

    Precedence: the explicit ``scheduler_address`` argument, then the
    :data:`DASK_SCHEDULER_ENV_VAR` environment variable. Empty /
    whitespace-only values count as unset.

    Returns:
        The address (``tcp://...`` URL or scheduler-file path), or
        ``None`` when no remote scheduler is configured.
    """
    if scheduler_address is not None and str(scheduler_address).strip():
        return str(scheduler_address).strip()
    env_value = os.environ.get(DASK_SCHEDULER_ENV_VAR, "").strip()
    return env_value or None


def _dask_address_kwargs(scheduler_address: str) -> Dict[str, Any]:
    """Map a scheduler address onto ``dask.distributed.Client`` kwargs.

    A value containing ``://`` is a direct scheduler URL; anything else
    is treated as a scheduler-file path (the canonical dask-on-Slurm
    pattern over a shared filesystem).
    """
    if "://" in scheduler_address:
        return {"address": scheduler_address}
    return {"scheduler_file": scheduler_address}


def _connect_timeout_s() -> float:
    """Remote-connect timeout in seconds (env override, default 120)."""
    raw = os.environ.get(DASK_CONNECT_TIMEOUT_ENV_VAR, "").strip()
    try:
        return float(raw) if raw else _DEFAULT_CONNECT_TIMEOUT_S
    except ValueError:
        warnings.warn(
            f"Ignoring non-numeric {DASK_CONNECT_TIMEOUT_ENV_VAR}={raw!r}; "
            f"using the default of {_DEFAULT_CONNECT_TIMEOUT_S:.0f}s.",
            UserWarning,
            stacklevel=2,
        )
        return _DEFAULT_CONNECT_TIMEOUT_S


def _library_version() -> Optional[str]:
    """Installed ``deap-pettingzoo`` version, or ``None`` when not installed.

    Module-level so it can be shipped to Dask workers via ``client.run``
    for the driver-vs-worker version check.
    """
    try:
        from importlib.metadata import version

        return version("deap-pettingzoo")
    except Exception:  # noqa: BLE001
        # Absence is an answer, not an error.
        return None


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


class _DaskExecutor:
    """Dask distributed executor.

    Three construction modes, in precedence order:

    * an injected ``dask.distributed.Client`` (lifetime not owned by us;
      no connect checks — the caller opted into their own layout),
    * a remote ``scheduler_address`` (URL or scheduler-file path) — we
      own the resulting ``Client`` and close it on ``__exit__`` (the
      client only, never the remote cluster),
    * neither — an auto-built ``LocalCluster`` (closed on ``__exit__``).
    """

    def __init__(
        self,
        num_workers: int,
        *,
        client: Optional[Any] = None,
        scheduler_address: Optional[str] = None,
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
        self._remote = client is None and scheduler_address is not None
        if client is None:
            if scheduler_address is not None:
                client = self._connect_remote(scheduler_address)
            else:
                from dask.distributed import Client

                # One thread per worker: the project's worker-side caches
                # (``_WORKER_CACHE`` and friends) and the rebuilt envs are not
                # thread-safe — they assume one task at a time per process, like
                # ``ProcessPoolExecutor``. Dask's multi-threaded default would
                # let concurrent tasks corrupt a shared cached env. Inject your
                # own ``Client`` if you need a different layout.
                client = Client(
                    n_workers=max(2, num_workers),
                    threads_per_worker=1,
                    processes=True,
                    dashboard_address=None,
                )
        self._client = client

    def _connect_remote(self, scheduler_address: str) -> Any:
        """Connect to a remote scheduler and wait for at least one worker.

        Raises:
            RuntimeError: when the scheduler is unreachable or no worker
                joins within the connect timeout
                (:data:`DASK_CONNECT_TIMEOUT_ENV_VAR`, default 120s).
        """
        from dask.distributed import Client

        timeout_s = _connect_timeout_s()
        try:
            # ``set_as_default=False`` keeps the library side-effect free —
            # we always dispatch through ``self._client`` explicitly.
            client = Client(
                timeout=timeout_s,
                set_as_default=False,
                **_dask_address_kwargs(scheduler_address),
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the Dask scheduler at "
                f"{scheduler_address!r} within {timeout_s:.0f}s. Check that "
                "the scheduler is running and reachable (address/scheduler "
                f"file, set via the scheduler_address argument or the "
                f"{DASK_SCHEDULER_ENV_VAR} environment variable; timeout via "
                f"{DASK_CONNECT_TIMEOUT_ENV_VAR})."
            ) from e
        try:
            client.wait_for_workers(1, timeout=timeout_s)
        except Exception as e:
            client.close()
            raise RuntimeError(
                f"Connected to the Dask scheduler at {scheduler_address!r} "
                f"but no worker joined within {timeout_s:.0f}s. Check that "
                "the worker processes started (e.g. the srun worker step in "
                "the sbatch script) and can reach the scheduler. Increase "
                f"{DASK_CONNECT_TIMEOUT_ENV_VAR} for slow queues."
            ) from e
        # Surface worker-side ``logging`` records on the driver (worker
        # ``warnings``/stdout stay worker-local — they land in the job's
        # srun output on Slurm). Skipped for in-process (inproc://) clusters:
        # there the workers already share the driver's root logger, and
        # forwarding would echo every record back to itself forever.
        if not scheduler_address.startswith("inproc://"):
            try:
                client.forward_logging()
            except Exception:  # noqa: BLE001
                # Observability aid, never fatal.
                pass
        self._post_connect_checks(client)
        return client

    @staticmethod
    def _post_connect_checks(client: Any) -> None:
        """Warn-only sanity checks after a remote connect.

        (a) any worker running more than one thread (the worker-side
        caches are not thread-safe), and (b) a driver-vs-worker
        ``deap-pettingzoo`` version mismatch (catches stale installs;
        Dask's own handshake already covers dask/python/numpy skew).
        """
        try:
            workers = client.scheduler_info().get("workers", {})
            threaded = {
                addr: int(w.get("nthreads", 1))
                for addr, w in workers.items()
                if int(w.get("nthreads", 1)) > 1
            }
            if threaded:
                warnings.warn(
                    f"Dask workers with nthreads > 1 detected: {threaded}. "
                    "The worker-side evaluation caches are not thread-safe; "
                    "start workers with --nthreads 1 (one task at a time per "
                    "process).",
                    RuntimeWarning,
                    stacklevel=3,
                )
        except Exception:  # noqa: BLE001
            pass
        try:
            driver_version = _library_version()
            worker_versions = client.run(_library_version)
            mismatched = {
                addr: v for addr, v in worker_versions.items() if v != driver_version
            }
            if mismatched:
                warnings.warn(
                    f"deap-pettingzoo version mismatch: driver has "
                    f"{driver_version!r} but workers report {mismatched}. "
                    "Install the same library version on all worker nodes — "
                    "mixed versions can fail in obscure ways mid-run.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "_DaskExecutor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owned:
            try:
                # Closes our client (and the auto-built LocalCluster it owns
                # on the local path); a remote cluster is never torn down.
                self._client.close()
            finally:
                self._client = None

    def worker_info(self) -> List[Dict[str, Any]]:
        """Snapshot of the connected workers, one dict per worker.

        Each entry carries ``address``, ``host`` and ``nthreads`` from
        ``Client.scheduler_info()`` so trainers can record the observed
        worker layout in :class:`TrainingResult` metadata.
        """
        workers = self._client.scheduler_info().get("workers", {})
        return [
            {
                "address": str(addr),
                "host": w.get("host"),
                "nthreads": int(w.get("nthreads", 1)),
            }
            for addr, w in sorted(workers.items())
        ]

    def map(
        self,
        fn: Callable[..., Any],
        *iterables: Iterable[Any],
    ) -> Iterator[Any]:
        if self._remote and getattr(fn, "__module__", None) == "__main__":
            raise ValueError(
                f"Cannot dispatch {getattr(fn, '__qualname__', fn)!r} to "
                "remote Dask workers: it is defined in '__main__' (the "
                "driver script), which is not importable on worker nodes. "
                "Move it into an installed, importable module that is "
                "present on every worker."
            )
        # Materialise to preserve ordering semantics; client.map + gather returns results in submission order.
        materialised = [list(it) for it in iterables]
        # retries=2: evaluations are pure/idempotent, so resubmitting after a
        # worker death is safe. A deterministic failure still surfaces (with
        # the remote traceback) after the third attempt.
        futures = self._client.map(fn, *materialised, pure=False, retries=2)
        results = self._client.gather(futures)
        return iter(results)


def make_executor(
    num_workers: Optional[int],
    *,
    backend: str = "process",
    initializer: Optional[Callable[..., None]] = None,
    initargs: Tuple[Any, ...] = (),
    dask_client: Optional[Any] = None,
    scheduler_address: Optional[str] = None,
) -> Any:
    """Build a context-managed executor.

    Args:
        num_workers: Requested worker count. Normalised via
            :func:`resolve_num_workers`; ``None``/``0``/``1`` yield a
            :class:`_SerialExecutor` — *unless* a remote Dask
            client/address is in play (see below).
        backend: One of ``"process"``, ``"dask"``.
        initializer: Optional per-worker init callable
            (``ProcessPoolExecutor`` only; ignored by Dask, which has
            no such hook — worker-side code must self-init lazily).
        initargs: Args forwarded to ``initializer``.
        dask_client: Optional pre-built
            ``dask.distributed.Client`` (Dask backend only). Takes
            precedence over ``scheduler_address``.
        scheduler_address: Optional remote Dask scheduler — a
            ``tcp://host:port`` URL or a scheduler-file path (Dask
            backend only). Falls back to the
            :data:`DASK_SCHEDULER_ENV_VAR` environment variable when
            unset.

    Remote intent wins: when ``backend == "dask"`` and a client /
    address / env var is present, the executor always dispatches to the
    cluster — the ``num_workers <= 1`` serial fast-path is skipped and
    the value is not capped to the driver's CPU count (``num_workers``
    is advisory metadata there; the cluster decides the real count).

    Returns:
        A context-managed executor with an ordered ``.map(fn, iter)``
        method.

    Raises:
        ValueError: if ``backend`` is not in :data:`SUPPORTED_BACKENDS`,
            or ``scheduler_address`` is passed with a non-Dask backend.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown parallel_backend {backend!r}. "
            f"Expected one of {SUPPORTED_BACKENDS}."
        )
    if scheduler_address is not None and backend != "dask":
        raise ValueError(
            f"scheduler_address={scheduler_address!r} requires "
            f"backend='dask', got backend={backend!r}."
        )

    # The env-var fallback only applies to the dask backend, and an injected
    # client outranks any address.
    effective_address: Optional[str] = None
    if backend == "dask" and dask_client is None:
        effective_address = resolve_scheduler_address(scheduler_address)
    remote = backend == "dask" and (
        dask_client is not None or effective_address is not None
    )

    n = resolve_num_workers(num_workers, cap_to_cpu=not remote)
    # Serial fast-path — skipped when remote intent was made explicit.
    if n <= 1 and not remote:
        return _SerialExecutor()

    if backend == "process":
        return ProcessPoolExecutor(
            max_workers=n,
            initializer=initializer,
            initargs=initargs,
        )

    if backend == "dask":
        return _DaskExecutor(
            n, client=dask_client, scheduler_address=effective_address
        )

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
