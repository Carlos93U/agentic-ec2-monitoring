"""Deterministic severities and the safety invariants of the lab.

Two ideas, both explained in README: Deterministic evaluator:

1. **The code decides the facts, the LLM decides the story.**
   Whether an instance is out of service and whether a CPU threshold was crossed
   are computed here from the collected payload. The Diagnosis agent may
   *escalate* a severity, but it can never downgrade one of the invariant
   findings below.

2. **The system still works when Bedrock does not.**
   Everything in this module is pure, testable code. If the crew fails, the report
   is still produced from these assessments and is still published, clearly marked
   as generated without the LLM.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from lambda_src.models import (
    OUT_OF_SERVICE_STATES,
    STATE_NOT_FOUND,
    STATE_PENDING,
    STATE_RUNNING,
    InstanceObservation,
    MonitoringPayload,
    Severity,
    Thresholds,
)

#: Findings that describe a fact that the LLM must not be able to soften.
INVARIANT_CODES = frozenset({"instance_out_of_service", "status_checks_failed"})


@dataclass(frozen=True)
class Finding:
    """One machine-derived observation about an instance."""

    code: str
    severity: Severity
    message: str
    evidence: Mapping[str, Any] | None = None

    @property
    def is_invariant(self) -> bool:
        """True when an LLM downgrade must be rejected."""
        return self.code in INVARIANT_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "evidence": dict(self.evidence or {}),
            "invariant": self.is_invariant,
        }


@dataclass(frozen=True)
class Assessment:
    """Deterministic verdict for a single instance."""

    instance_id: str
    name: str
    state: str
    severity: Severity
    findings: tuple[Finding, ...] = ()

    @property
    def is_out_of_service(self) -> bool:
        return any(finding.code == "instance_out_of_service" for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "state": self.state,
            "severity": self.severity.value,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class MergedAssessment:
    """Deterministic assessment after merging the Diagnosis agent's opinion."""

    assessment: Assessment
    agent_severity: Severity | None
    agent_narrative: Mapping[str, str]
    guard_applied: bool

    @property
    def instance_id(self) -> str:
        return self.assessment.instance_id

    @property
    def name(self) -> str:
        return self.assessment.name

    @property
    def state(self) -> str:
        return self.assessment.state

    @property
    def severity(self) -> Severity:
        return self.assessment.severity

    @property
    def findings(self) -> tuple[Finding, ...]:
        return self.assessment.findings

    @property
    def was_overridden(self) -> bool:
        """True when the agent tried to downgrade an invariant finding."""
        return self.guard_applied


def evaluate_instance(observation: InstanceObservation, thresholds: Thresholds) -> Assessment:
    """Compute the deterministic severity of a single instance.

    Rules, in order:

    * ``stopped`` / ``stopping`` / ``shutting-down`` / ``terminated`` -> CRITICAL,
      the instance is out of service (independent of what the metrics say).
    * ``pending`` -> WARNING, the instance is still starting up.
    * ``not-found`` -> WARNING, the monitoring scope no longer matches reality.
    * ``StatusCheckFailed`` total > 0 -> CRITICAL.
    * CPU **average** above the thresholds -> WARNING / CRITICAL.
    * Running with no datapoints at all -> WARNING (we cannot confirm health).
    """
    snapshot = observation.snapshot
    findings: list[Finding] = []
    state = snapshot.state

    if state in OUT_OF_SERVICE_STATES:
        findings.append(
            Finding(
                code="instance_out_of_service",
                severity=Severity.CRITICAL,
                message=(
                    f"Instance is currently {state.upper()} and is therefore out of service. "
                    "Any metric value still shown in the window belongs to the period before it went down "
                    "and must not be read as current activity."
                ),
                evidence={
                    "state": state,
                    "state_transition_reason": snapshot.state_transition_reason,
                    "launch_time": snapshot.launch_time.isoformat() if snapshot.launch_time else None,
                },
            )
        )
    elif state == STATE_PENDING:
        findings.append(
            Finding(
                code="instance_starting",
                severity=Severity.WARNING,
                message="Instance is pending: it has not finished starting yet, so no metrics exist yet.",
                evidence={"state": state},
            )
        )
    elif state == STATE_NOT_FOUND:
        findings.append(
            Finding(
                code="instance_not_found",
                severity=Severity.WARNING,
                message=(
                    "EC2 did not return this instance id. It may have been terminated outside the lab, "
                    "so the monitoring scope must be reviewed."
                ),
                evidence={"state": state},
            )
        )
    elif state != STATE_RUNNING:
        findings.append(
            Finding(
                code="instance_unknown_state",
                severity=Severity.WARNING,
                message=f"Unexpected instance state: {state!r}.",
                evidence={"state": state},
            )
        )

    status_checks = observation.metric("StatusCheckFailed")
    failed_checks = status_checks.total if status_checks is not None else None
    if status_checks is not None and status_checks.available and (failed_checks or 0) > 0:
        findings.append(
            Finding(
                code="status_checks_failed",
                severity=Severity.CRITICAL,
                message=(
                    f"{int(failed_checks or 0)} failed EC2 instance status checks in the window: the instance "
                    "is running but its underlying host checks are failing."
                ),
                evidence={"total_failed_checks": status_checks.total, "unit": status_checks.unit},
            )
        )

    cpu = observation.metric("CPUUtilization")
    if state == STATE_RUNNING:
        if cpu is None or not cpu.available:
            findings.append(
                Finding(
                    code="no_metric_data",
                    severity=Severity.WARNING,
                    message=(
                        "The instance is running but CloudWatch returned no CPU datapoints for the window. "
                        "Health cannot be confirmed from metrics."
                    ),
                    evidence={"reason": cpu.unavailable_reason if cpu else "metric_not_collected"},
                )
            )
        else:
            average = cpu.average or 0.0
            if average >= thresholds.cpu_critical_percent:
                findings.append(
                    Finding(
                        code="cpu_critical",
                        severity=Severity.CRITICAL,
                        message=(
                            f"Sustained CPU utilisation of {average:.1f}% over the window is at or above the "
                            f"critical threshold of {thresholds.cpu_critical_percent:.0f}%."
                        ),
                        evidence={
                            "cpu_average": average,
                            "cpu_maximum": cpu.maximum,
                            "threshold": thresholds.cpu_critical_percent,
                        },
                    )
                )
            elif average >= thresholds.cpu_warning_percent:
                findings.append(
                    Finding(
                        code="cpu_high",
                        severity=Severity.WARNING,
                        message=(
                            f"Sustained CPU utilisation of {average:.1f}% over the window is at or above the "
                            f"warning threshold of {thresholds.cpu_warning_percent:.0f}%."
                        ),
                        evidence={
                            "cpu_average": average,
                            "cpu_maximum": cpu.maximum,
                            "threshold": thresholds.cpu_warning_percent,
                        },
                    )
                )

    for network_metric in ("NetworkIn", "NetworkOut"):
        metric = observation.metric(network_metric)
        if metric is not None and not metric.available and state == STATE_RUNNING:
            findings.append(
                Finding(
                    code="metric_unavailable",
                    severity=Severity.WARNING,
                    message=f"No {network_metric} datapoints in the window (reason: {metric.unavailable_reason}).",
                    evidence={"metric": network_metric, "reason": metric.unavailable_reason},
                )
            )

    severity = Severity.worst([finding.severity for finding in findings]) if findings else Severity.HEALTHY
    return Assessment(
        instance_id=snapshot.instance_id,
        name=snapshot.name,
        state=snapshot.state,
        severity=severity,
        findings=tuple(findings),
    )


def evaluate_payload(payload: MonitoringPayload) -> tuple[Assessment, ...]:
    """Evaluate every observed instance of the payload."""
    return tuple(
        evaluate_instance(observation, payload.thresholds) for observation in payload.observations
    )


def overall_severity(assessments: Iterable[Assessment | MergedAssessment]) -> Severity:
    """Worst severity across instances (``HEALTHY`` when there are none).

    Accepts plain :class:`Assessment` objects and the :class:`MergedAssessment`
    wrapper alike; both expose ``.severity``.
    """
    return Severity.worst([assessment.severity for assessment in assessments])


def merge_agent_severity(
    assessment: Assessment,
    agent_severity: Severity | None,
) -> tuple[Severity, bool]:
    """Merge the agent's opinion with the deterministic verdict.

    Rules, in one line: **the agent may escalate, never de-escalate.**

    * A missing or unparsable agent severity keeps the deterministic verdict.
    * A *higher* agent severity is accepted, because the LLM can notice things
      thresholds do not cover (a suspicious network pattern, an instance that is
      running but serving nothing).
    * A *lower* agent severity is always rejected. When the verdict being defended
      includes an invariant finding, ``guard_applied`` is ``True`` so the report
      can say explicitly that the agent tried to soften a hard fact.

    Returns:
        ``(severity, guard_applied)``.
    """
    if agent_severity is None:
        return assessment.severity, False

    has_invariant = any(finding.is_invariant for finding in assessment.findings)
    if agent_severity.rank >= assessment.severity.rank:
        return agent_severity, False

    if has_invariant:
        return assessment.severity, True

    return assessment.severity, False


def merge_assessments(
    assessments: tuple[Assessment, ...] | list[Assessment],
    agent_severities: Mapping[str, Any],
    agent_narratives: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[MergedAssessment, ...]:
    """Attach the Diagnosis agent's opinion to each deterministic assessment.

    Deliberately pure: it takes plain mappings, not the CrewAI objects, so the
    safety rules can be unit-tested without the framework installed. The handler
    feeds it ``narrative.agent_statuses`` and ``narrative.agent_narratives``.

    Args:
        assessments: Deterministic verdicts, one per instance.
        agent_severities: ``{instance_id: "HEALTHY" | "WARNING" | ...}``. Unknown
            or missing values are treated as "the agent said nothing".
        agent_narratives: ``{instance_id: {"what_happened": ..., ...}}``.

    Returns:
        The assessments with the agent verdict and any guard applied.
    """
    narratives = agent_narratives or {}
    merged: list[MergedAssessment] = []
    for assessment in assessments:
        agent_severity = Severity.parse(agent_severities.get(assessment.instance_id))
        severity, guard = merge_agent_severity(assessment, agent_severity)
        # The escalated severity has to become the assessment of record, otherwise
        # the report and the overall status would ignore the agent's escalation.
        effective = assessment if severity is assessment.severity else replace(assessment, severity=severity)
        merged.append(
            MergedAssessment(
                assessment=effective,
                agent_severity=agent_severity,
                agent_narrative=dict(narratives.get(assessment.instance_id, {}) or {}),
                guard_applied=guard,
            )
        )
    return tuple(merged)
