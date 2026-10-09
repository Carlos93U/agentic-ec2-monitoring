"""Builds and runs the CrewAI crew, then parses whatever it produced.

Failure policy
--------------
Any crew failure (Bedrock ``AccessDenied``, throttling, timeout, malformed output)
is turned into :class:`CrewExecutionError` with the underlying detail attached, and
the handler degrades to the deterministic report. A Bedrock outage must never
destroy the monitoring diagnosis.

Note the deliberate asymmetry:

* **Bedrock/CrewAI failure** -> soft failure. The run continues, the report is
  rendered from :mod:`lambda_src.evaluator`, and the reason is logged.
* **Bad configuration** -> hard failure. Missing ``BEDROCK_MODEL_ID`` or an empty
  instance scope is a deployment bug and should fail loudly at invocation time.

CrewAI's structured-output features are intentionally not used; see the module
docstring of :mod:`lambda_src.tasks`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from lambda_src.agents import build_agents
from lambda_src.config import Settings
from lambda_src.llm_factory import describe_llm
from lambda_src.logging_utils import get_logger, log_event
from lambda_src.models import MonitoringPayload, Severity
from lambda_src.tasks import build_tasks

LOGGER = get_logger("crew")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CrewNarrative:
    """Structured output of the three agents (already parsed and normalised)."""

    monitoring: dict[str, Any] = field(default_factory=dict)
    diagnosis: dict[str, Any] = field(default_factory=dict)
    reporting: dict[str, Any] = field(default_factory=dict)
    agent_statuses: dict[str, Severity | None] = field(default_factory=dict)
    agent_narratives: dict[str, dict[str, str]] = field(default_factory=dict)
    parse_issues: tuple[str, ...] = ()
    usage: dict[str, Any] | None = None

    @property
    def overall_agent_status(self) -> Severity | None:
        """Status the Diagnosis agent proposed for the whole fleet, if any."""
        return Severity.parse(self.diagnosis.get("overall_status"))

    @property
    def summary(self) -> str:
        """Executive summary written by the Reporting agent."""
        return str(self.reporting.get("executive_summary") or "").strip()

    @property
    def priority_actions(self) -> tuple[str, ...]:
        """Ordered next steps written by the Reporting agent."""
        actions = self.reporting.get("priority_actions") or []
        if isinstance(actions, str):
            return (actions,)
        return tuple(str(action).strip() for action in actions if str(action).strip())

    @property
    def closing_note(self) -> str:
        return str(self.reporting.get("closing_note") or "").strip()

    @property
    def monitoring_summary(self) -> str:
        return str(self.monitoring.get("summary") or "").strip()


class CrewExecutionError(RuntimeError):
    """Raised when the crew cannot be built or executed."""


class CrewRunner:
    """Owns the crew lifecycle for a single invocation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_crew(self) -> Any:
        """Instantiate the CrewAI crew (agents + tasks, sequential)."""
        try:
            from crewai import Crew, Process
        except ModuleNotFoundError as exc:  # pragma: no cover - image always has crewai
            raise CrewExecutionError(
                "crewai is not installed in this runtime. The Lambda container image must install "
                "crewai[bedrock] (see lambda_src/Dockerfile)."
            ) from exc

        try:
            agents = build_agents(self._settings)
            monitoring_task, diagnosis_task, reporting_task = build_tasks(agents, self._settings)
        except Exception as exc:  # noqa: BLE001 - surfaced with context
            raise CrewExecutionError(f"Could not build the crew: {type(exc).__name__}: {exc}") from exc

        crew = Crew(
            agents=[agents["monitoring"], agents["diagnosis"], agents["reporting"]],
            tasks=[monitoring_task, diagnosis_task, reporting_task],
            process=Process.sequential,
            verbose=False,
            # No memory and no planning: both would add latency and cost to a run
            # that happens 288 times a day, and every run is stateless by design.
            memory=False,
            planning=False,
            cache=False,
        )
        log_event(
            LOGGER,
            logging.INFO,
            "crew_built",
            llm_shared=describe_llm(self._settings, reasoning=False),
            llm_diagnosis=describe_llm(self._settings, reasoning=True),
        )
        return crew

    def run(self, payload: MonitoringPayload, *, overall_status: Severity) -> CrewNarrative:
        """Run the crew and return its parsed narrative.

        Args:
            payload: The structured data the crew is allowed to see.
            overall_status: The deterministic overall status, injected into the
                Reporting prompt so the narrative cannot contradict the facts.

        Raises:
            CrewExecutionError: for any build, Bedrock or parsing failure.
        """
        crew = self.build_crew()

        log_event(LOGGER, logging.INFO, "crew_kickoff", instances=len(payload.observations))
        try:
            # Only these two are template variables. The Diagnosis and Reporting
            # tasks receive the previous task outputs through Task.context, not
            # through inputs (see lambda_src/tasks.py).
            crew.kickoff(
                inputs={
                    "payload_json": payload.to_json(),
                    "overall_status": overall_status.value,
                }
            )
        except Exception as exc:  # noqa: BLE001 - any provider failure is soft
            raise CrewExecutionError(
                f"CrewAI execution failed: {type(exc).__name__}: {exc}"
            ) from exc

        outputs = self._collect_outputs(crew)
        usage = self._collect_usage(crew)

        monitoring = outputs.get("monitoring") or {}
        diagnosis = outputs.get("diagnosis") or {}
        reporting = outputs.get("reporting") or {}

        agent_statuses: dict[str, Severity | None] = {}
        agent_narratives: dict[str, dict[str, str]] = {}
        for entry in diagnosis.get("instances", []) or []:
            if not isinstance(entry, dict):
                continue
            instance_id = str(entry.get("instance_id", ""))
            if not instance_id:
                continue
            agent_statuses[instance_id] = Severity.parse(entry.get("status"))
            agent_narratives[instance_id] = {
                "what_happened": str(entry.get("what_happened", "") or "").strip(),
                "affected_metric": str(entry.get("affected_metric", "") or "").strip(),
                "possible_cause": str(entry.get("possible_cause", "") or "").strip(),
                "recommendation": str(entry.get("recommendation", "") or "").strip(),
            }

        log_event(
            LOGGER,
            logging.INFO,
            "crew_finished",
            statuses={key: (value.value if value else None) for key, value in agent_statuses.items()},
            usage=usage,
        )

        return CrewNarrative(
            monitoring=monitoring,
            diagnosis=diagnosis,
            reporting=reporting,
            agent_statuses=agent_statuses,
            agent_narratives=agent_narratives,
            parse_issues=tuple(outputs.get("_issues", [])),
            usage=usage,
        )

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _collect_outputs(crew: Any) -> dict[str, Any]:
        """Extract and parse the three task outputs.

        Returns:
            ``{"monitoring": dict, "diagnosis": dict, "reporting": dict, "_issues": [...]}``.
        """
        parsed: dict[str, Any] = {"_issues": []}
        tasks = list(getattr(crew, "tasks", []))
        for index, name in enumerate(("monitoring", "diagnosis", "reporting")):
            # A crew that did not finish all three tasks must degrade to a
            # deterministic report, not raise KeyError further up.
            task = tasks[index] if index < len(tasks) else None
            if task is None:
                parsed["_issues"].append(f"{name}: task did not run")
                parsed[name] = {}
                continue
            raw = ""
            output = getattr(task, "output", None)
            if output is not None:
                raw = getattr(output, "raw", "") or getattr(output, "json_dict", "") or str(output)
            document = parse_json_object(raw)
            if document is None:
                parsed["_issues"].append(f"{name}: output was not valid JSON ({len(raw)} chars)")
                parsed[name] = {}
            else:
                parsed[name] = document
        return parsed

    @staticmethod
    def _collect_usage(crew: Any) -> dict[str, Any] | None:
        """Token usage reported by CrewAI, used to keep the lab cost visible."""
        usage = getattr(crew, "usage_metrics", None)
        if not usage:
            return None
        if isinstance(usage, dict):
            return usage
        for attribute in ("total_tokens", "prompt_tokens", "completion_tokens", "cached_prompt_tokens"):
            if hasattr(usage, attribute):
                return {attribute: getattr(usage, attribute) for attribute in dir(usage) if not attribute.startswith("_")}
        return None


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object out of a model answer.

    Strategy, in order: strict parse, ``json-repair``, fenced code block, outermost
    ``{...}`` slice. Returns ``None`` when nothing works, which lets the caller
    fall back to the deterministic report instead of crashing.
    """
    if not raw:
        return None
    candidates: list[str] = [raw.strip()]

    candidates.extend(match.strip() for match in _FENCE_RE.findall(raw))

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        if not candidate:
            continue
        for attempt in (_strict_json, _repair_json):
            document = attempt(candidate)
            if document is not None:
                return document
    return None


def _strict_json(candidate: str) -> dict[str, Any] | None:
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _repair_json(candidate: str) -> dict[str, Any] | None:
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:  # pragma: no cover - json-repair ships with crewai
        return None
    try:
        value = repair_json(candidate, return_objects=True)
    except Exception:  # noqa: BLE001 - repair is best effort
        return None
    return value if isinstance(value, dict) else None
