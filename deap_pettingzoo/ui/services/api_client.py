"""HTTP client for the FastAPI experiment service.

Connects the Streamlit frontend to the backend via ``FASTAPI_URL``.
This module is **import-safe without Streamlit** — it only depends on
:mod:`httpx` (declared in the ``[ui]`` extra). Unit tests can mock
``httpx.Client`` or set ``FASTAPI_URL`` to a test server.

All methods raise typed exceptions so the UI can map them to
user-friendly banners / error displays:

* :class:`ApiUnavailable` — the backend is unreachable (connection refused,
  DNS failure, timeout). The Setup/Queue pages render a yellow banner
  asking the user to start the service.
* :class:`ApiValidationError` — the backend replied 422 (invalid experiment
  spec). The messages are rendered inline alongside the form fields.
* :class:`ApiTimeout` — the request took longer than ``DEFAULT_TIMEOUT``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


DEFAULT_FASTAPI_URL = "http://127.0.0.1:8100"
"""Default backend URL. Override via the ``FASTAPI_URL`` env var or by
passing ``base_url`` to :class:`ApiClient`."""

DEFAULT_TIMEOUT = 15.0
"""HTTP timeout (seconds) applied to every request."""


# ---------------------------------------------------------------------------
# Client errors (typed, so the UI can branch on them)
# ---------------------------------------------------------------------------


class ApiClientError(Exception):
    """Base for all API-client errors."""


class ApiUnavailable(ApiClientError):
    """The backend service is unreachable or timed out."""


class ApiTimeout(ApiClientError):
    """A request exceeded the configured timeout."""


class ApiValidationError(ApiClientError):
    """The backend replied 422 — experiment spec is invalid.

    Attributes:
        report: The parsed :class:`ValidationReport` dict from the server.
    """

    def __init__(self, report: Dict[str, Any]):
        self.report = report
        super().__init__(
            f"Validation failed with {len(report.get('issues', []))} issue(s)"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


@dataclass
class ApiClient:
    """Thin wrapper around ``httpx.Client`` with typed methods for every
    endpoint in the FastAPI service.

    Instantiate with ``base_url=None`` to read ``FASTAPI_URL`` from the
    environment. All methods are synchronous (the Streamlit execution model
    is single-threaded per-session; async wouldn't buy us anything here).

    Usage::

        client = ApiClient()
        catalog = client.get_catalog()
        algos  = client.get_algorithms()
        spec   = ExperimentSpecModel(...)
        report = client.validate(spec)      # raises ApiValidationError
        sub    = client.submit(spec)        # raises ApiValidationError
        progress = client.get_progress(submitted_id)
    """

    base_url: str = DEFAULT_FASTAPI_URL
    timeout: float = DEFAULT_TIMEOUT
    _client: Optional[Any] = None  # httpx.Client (lazy)

    def __post_init__(self) -> None:
        env_url = os.environ.get("FASTAPI_URL", "").strip()
        if env_url:
            self.base_url = env_url.rstrip("/")

    @property
    def client(self) -> Any:
        """Lazily-built :class:`httpx.Client`. Raises
        :class:`ApiUnavailable` if ``httpx`` is not installed."""
        if self._client is None:
            if httpx is None:
                raise ApiUnavailable(
                    "The 'httpx' package is required for the API client. "
                    "Install with: pip install deap-pettingzoo[ui]"
                )
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    # -- helpers -----------------------------------------------------------

    def _get(self, path: str, **kwargs) -> Dict[str, Any]:
        try:
            r = self.client.get(path, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiUnavailable(
                f"Cannot reach the backend at {self.base_url}. "
                f"Is the service running? ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiTimeout(
                f"Request to {path} timed out after {self.timeout}s"
            ) from exc
        return self._check(r)

    def _post(self, path: str, json_data=None, **kwargs) -> Dict[str, Any]:
        try:
            r = self.client.post(path, json=json_data, **kwargs)
        except httpx.ConnectError as exc:
            raise ApiUnavailable(
                f"Cannot reach the backend at {self.base_url}. ({exc})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ApiTimeout(
                f"Request to {path} timed out after {self.timeout}s"
            ) from exc
        return self._check(r)

    def _check(self, response: Any) -> Dict[str, Any]:
        status = response.status_code
        if status == 422:
            try:
                body = response.json()
            except Exception:
                body = {"ok": False, "issues": []}
            raise ApiValidationError(body)
        if status == 404:
            body = response.json() if getattr(response, "headers", {}).get("content-type") else {}
            raise ApiClientError(
                body.get("detail", f"Not found: {getattr(getattr(response, 'url', None), 'path', '?')}")
            )
        if isinstance(status, int) and status >= 400:
            raise ApiClientError(
                f"Backend returned {status}: "
                f"{getattr(response, 'text', '')[:300]}"
            )
        return response.json() if getattr(response, "text", None) else {}

    # -- catalog -----------------------------------------------------------

    def get_catalog(self) -> Dict[str, Any]:
        """Return the full catalog snapshot."""
        return self._get("/catalog")

    def get_algorithms(self) -> List[Dict[str, Any]]:
        return self._get("/catalog/algorithms")

    def get_algorithm(self, name: str) -> Dict[str, Any]:
        return self._get(f"/catalog/algorithms/{name}")

    def get_environments(self) -> List[Dict[str, Any]]:
        return self._get("/catalog/environments")

    def get_environment(self, name: str) -> Dict[str, Any]:
        return self._get(f"/catalog/environments/{name}")

    def get_coevolution_options(self) -> Dict[str, Any]:
        return self._get("/catalog/coevolution")

    def get_compatibility(
        self, env: str, algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        params = {"env": env}
        if algorithms:
            params["algorithms"] = ",".join(algorithms)
        return self._get("/catalog/compatibility", params=params)

    # -- experiments -------------------------------------------------------

    def validate(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Dry-run validation. Returns a :class:`ValidationReport`-shaped dict
        or raises :class:`ApiValidationError`."""
        return self._post("/experiments/validate", json_data=spec)

    def submit(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Submit an experiment spec. Returns ``{id, status, position}`` or
        raises :class:`ApiValidationError`."""
        return self._post("/experiments", json_data=spec)

    def list_experiments(
        self, status: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return self._get("/experiments", params=params)

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        return self._get(f"/experiments/{exp_id}")

    def cancel_experiment(self, exp_id: str) -> Dict[str, Any]:
        return self._post(f"/experiments/{exp_id}/cancel")

    def delete_experiment(self, exp_id: str) -> Dict[str, Any]:
        return self.client.delete(f"{self.base_url}/experiments/{exp_id}").json()

    def get_progress(self, exp_id: str) -> Dict[str, Any]:
        return self._get(f"/experiments/{exp_id}/progress")

    def get_stdout(
        self,
        exp_id: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        lines: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if offset is not None:
            params["offset"] = offset
        if limit is not None:
            params["limit"] = limit
        if lines is not None:
            params["lines"] = lines
        return self._get(f"/experiments/{exp_id}/stdout", params=params)

    # -- queue -------------------------------------------------------------

    def get_queue(self) -> Dict[str, Any]:
        return self._get("/queue")

    def pause_queue(self) -> Dict[str, Any]:
        return self._post("/queue/pause")

    def resume_queue(self) -> Dict[str, Any]:
        return self._post("/queue/resume")
