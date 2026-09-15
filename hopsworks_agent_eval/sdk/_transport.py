"""One HTTP door for the whole client.

Every resource handle talks through this: it knows the project's base URL, how
the caller is entitled to talk to it, and how a refusal is turned into an error
that says which rule refused and why -- the API's own message, never a bare
status. Paths are relative to ``/hopsworks-api/api/project/{id}``, so the same
transport reaches the evaluation API and the tracing API.
"""

from __future__ import annotations

import os
from typing import Any

from ..api import EvalApiError, _StaticAuth, hopsworks_session

TIMEOUT_S = 60


class AgentEvalsError(EvalApiError):
    """A refusal from the API, carrying the reason it gave and the status."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _message(response: Any) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(body, dict):
        return body.get("usrMsg") or body.get("errorMsg") or f"HTTP {response.status_code}"
    return f"HTTP {response.status_code}"


class Transport:
    def __init__(self, host: str, project_id: int, *, api_key: str | None = None,
                 verify: bool | str = True, session: Any = None):
        self.host = host.rstrip("/")
        self.project_id = int(project_id)
        self.base = f"{self.host}/hopsworks-api/api/project/{self.project_id}"
        if session is not None:
            self._session = session
        elif api_key:
            import requests

            self._session = requests.Session()
            self._session.auth = _StaticAuth("ApiKey " + api_key)
            self._session.verify = verify
        else:
            # whatever this container has: a job's JWT, a notebook's connected client
            self._session = hopsworks_session()
            if verify is not True:
                self._session.verify = verify

    def request(self, method: str, path: str, *, params: dict[str, Any] | None = None,
                json: Any = None) -> Any:
        clean = None
        if params:
            # None means "not given", never the string "None"
            clean = {k: v for k, v in params.items() if v is not None} or None
        response = self._session.request(method, self.base + path, params=clean, json=json, timeout=TIMEOUT_S)
        if response.status_code >= 400:
            raise AgentEvalsError(_message(response), response.status_code)
        if not getattr(response, "content", b""):
            return None
        return response.json()

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params or None)

    def post(self, path: str, json: Any = None, **params: Any) -> Any:
        return self.request("POST", path, params=params or None, json=json)

    def put(self, path: str, json: Any = None, **params: Any) -> Any:
        return self.request("PUT", path, params=params or None, json=json)

    def delete(self, path: str, **params: Any) -> Any:
        return self.request("DELETE", path, params=params or None)


def host_from_env() -> str:
    host = os.environ.get("HOPSWORKS_HOST") or os.environ.get("REST_ENDPOINT")
    if not host:
        raise AgentEvalsError("no host: pass host= or set HOPSWORKS_HOST")
    if not host.startswith("http"):
        host = "https://" + host
    return host
