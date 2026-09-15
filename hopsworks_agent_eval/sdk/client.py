"""The client: one object bound to a project, with a handle per kind of thing.

    from hopsworks_agent_eval.sdk import login

    evals = login()                       # inside Hopsworks; or login(host=..., project_id=..., api_key=...)
    suite = evals.suites.create("Refunds", checks=[check("llm_judge", "quality", criteria=[...])])
    evals.tasks.create("Refund order 42", expectations={"quality": "..."}).add_to(suite)
    suite.publish()
    run = evals.deployment(7).run(suite).wait()
    for trial in run.trials(): ...

Everything the UI does for agent evaluation and tracing is reachable here. The
package is standalone for now and shaped to slot into the hopsworks library
later, which is why the entry point mirrors ``hopsworks.login()``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, TypeVar

from ._transport import AgentEvalsError, Transport, host_from_env
from .evals import Evaluators, Jobs, Runs, Suites, Tasks
from .models import ApiModel, ReviewJob, Run, Suite, Task
from .tracing import Deployment

M = TypeVar("M", bound=ApiModel)


class AgentEvals:
    def __init__(self, host: str, project_id: int, *, api_key: str | None = None, verify: bool | str = True,
                 session: Any = None):
        self.http = Transport(host, project_id, api_key=api_key, verify=verify, session=session)
        self.project_id = int(project_id)
        self.suites = Suites(self)
        self.tasks = Tasks(self)
        self.evaluators = Evaluators(self)
        self.runs = Runs(self)
        self.jobs = Jobs(self)

    def __repr__(self) -> str:
        return f"AgentEvals(project={self.project_id}, host={self.http.host!r})"

    def deployment(self, deployment_id: int) -> Deployment:
        return Deployment(self, deployment_id)

    # models that act on themselves need the client they came from
    def _bind(self, model: M) -> M:
        model._client = self  # type: ignore[attr-defined]
        return model

    def _bind_all(self, models: Iterable[M]) -> list[M]:
        return [self._bind(m) for m in models]


def login(host: str | None = None, project_id: int | None = None, *, api_key: str | None = None,
          verify: bool | str = True) -> AgentEvals:
    """A client for one project.

    Inside Hopsworks -- a job, a notebook -- nothing needs passing: the host and
    the project come from the environment and the connected client, and the
    caller's own token is used. From outside, pass ``host``, ``project_id`` and
    an ``api_key`` with the serving scope (or set HOPSWORKS_HOST,
    HOPSWORKS_PROJECT_ID and HOPSWORKS_API_KEY).
    """
    host = host or host_from_env()
    api_key = api_key or os.environ.get("HOPSWORKS_API_KEY")
    if project_id is None:
        raw = os.environ.get("HOPSWORKS_PROJECT_ID")
        if raw:
            project_id = int(raw)
        else:
            try:
                import hopsworks  # noqa: PLC0415 -- only inside Hopsworks

                project_id = hopsworks.login().id
            except Exception as err:  # noqa: BLE001 -- the reason is the message
                raise AgentEvalsError(f"no project: pass project_id= or set HOPSWORKS_PROJECT_ID ({err})") from err
    return AgentEvals(host, project_id, api_key=api_key, verify=verify)


# ── methods on the models, calling back through the client they came from ──


def _client_of(model: ApiModel) -> AgentEvals:
    client = getattr(model, "_client", None)
    if client is None:
        raise AgentEvalsError("this object was not fetched through a client; use the client's methods instead")
    return client


def _suite_publish(self: Suite) -> Suite:
    return _client_of(self).suites.publish(self)


def _suite_tasks(self: Suite) -> list[Task]:
    return _client_of(self).suites.tasks(self)


def _suite_add_task(self: Suite, question: Any, expectations: dict[str, str] | None = None, **kwargs: Any) -> Task:
    """Author a task and join it to this suite with what it expects of each check."""
    client = _client_of(self)
    task = client.tasks.create(question, **kwargs)
    return client.tasks.add_to_suite(task, self, expectations=expectations)


def _suite_update(self: Suite, **changes: Any) -> Suite:
    return _client_of(self).suites.update(self, **changes)


def _suite_set_checks(self: Suite, checks: Any) -> Any:
    return _client_of(self).suites.set_checks(self, checks)


def _suite_delete(self: Suite, force: bool = False) -> None:
    _client_of(self).suites.delete(self, force=force)


def _suite_new_version(self: Suite) -> Suite:
    return _client_of(self).suites.new_version(self)


def _suite_import(self: Suite, tasks: Any) -> dict[str, int]:
    return _client_of(self).suites.import_tasks(self, tasks)


def _suite_run(self: Suite, deployment_id: int, n_trials: int = 1) -> Run:
    return _client_of(self).runs.start(self, deployment_id, n_trials=n_trials)


Suite.publish = _suite_publish  # type: ignore[attr-defined]
Suite.tasks = _suite_tasks  # type: ignore[attr-defined]
Suite.add_task = _suite_add_task  # type: ignore[attr-defined]
Suite.update = _suite_update  # type: ignore[attr-defined]
Suite.set_checks = _suite_set_checks  # type: ignore[attr-defined]
Suite.delete = _suite_delete  # type: ignore[attr-defined]
Suite.new_version = _suite_new_version  # type: ignore[attr-defined]
Suite.import_tasks = _suite_import  # type: ignore[attr-defined]
Suite.run = _suite_run  # type: ignore[attr-defined]


def _task_add_to(self: Task, suite: Suite | str, expectations: dict[str, str] | None = None,
                 version: int | None = None) -> Task:
    return _client_of(self).tasks.add_to_suite(self, suite, version=version, expectations=expectations)


def _task_leave(self: Task) -> Task:
    return _client_of(self).tasks.remove_from_suite(self)


def _task_confirm(self: Task, **kwargs: Any) -> Task:
    return _client_of(self).tasks.confirm_redaction(self, **kwargs)


def _task_regressions(self: Task, **kwargs: Any) -> Task:
    return _client_of(self).tasks.add_to_regressions(self, **kwargs)


def _task_delete(self: Task) -> None:
    _client_of(self).tasks.delete(self)


Task.add_to = _task_add_to  # type: ignore[attr-defined]
Task.leave_suite = _task_leave  # type: ignore[attr-defined]
Task.confirm_redaction = _task_confirm  # type: ignore[attr-defined]
Task.add_to_regressions = _task_regressions  # type: ignore[attr-defined]
Task.delete = _task_delete  # type: ignore[attr-defined]


def _run_refresh(self: Run) -> Run:
    return _client_of(self).runs.get(self.run_id)


def _run_wait(self: Run, **kwargs: Any) -> Run:
    return _client_of(self).runs.wait(self, **kwargs)


def _run_trials(self: Run) -> Any:
    return _client_of(self).runs.trials(self)


def _run_results(self: Run, trial: Any = None) -> Any:
    return _client_of(self).runs.results(self, trial)


def _run_metrics(self: Run) -> Any:
    return _client_of(self).runs.metrics(self)


def _run_review(self: Run, trial: Any, **kwargs: Any) -> None:
    _client_of(self).runs.review_trial(self, trial, **kwargs)


Run.refresh = _run_refresh  # type: ignore[attr-defined]
Run.wait = _run_wait  # type: ignore[attr-defined]
Run.trials = _run_trials  # type: ignore[attr-defined]
Run.results = _run_results  # type: ignore[attr-defined]
Run.metrics = _run_metrics  # type: ignore[attr-defined]
Run.review_trial = _run_review  # type: ignore[attr-defined]


def _review_analyse(self: ReviewJob, **kwargs: Any) -> Run:
    return _client_of(self).jobs.analyse(self, **kwargs)


def _review_update(self: ReviewJob, **settings: Any) -> ReviewJob:
    return _client_of(self).jobs.update_review_job(self, **settings)


def _review_delete(self: ReviewJob) -> None:
    _client_of(self).jobs.delete_review_job(self)


ReviewJob.analyse = _review_analyse  # type: ignore[attr-defined]
ReviewJob.update = _review_update  # type: ignore[attr-defined]
ReviewJob.delete = _review_delete  # type: ignore[attr-defined]
