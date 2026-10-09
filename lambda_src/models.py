"""Data contract shared by the collectors, the evaluator and the report.

These dataclasses are the *only* source of numbers in the system. The LLM receives
a JSON rendering of them and may only contribute qualitative text; the final
report prints the numeric fields straight from here. That is how the lab makes
"the agents never invent metrics" a structural guarantee instead of a prompt
request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    """Timezone-aware UTC now (single place so tests can reason about it)."""
    return datetime.now(UTC)


def iso(value: datetime | None) -> str | None:
    """Render a datetime the way the report and the agents expect: ``...Z``."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


class Severity(str, Enum):  # noqa: UP042 - see comment below
    """How bad it is. Order matters: :func:`max` compares by rank.

    ``str, Enum`` instead of ``StrEnum``: these values travel inside dataclasses
    that are serialised straight into the JSON payload and the report, and the
    ``str`` mixin keeps them JSON/format friendly on every path, including any
    f-string interpolation of a raw value.
    """

    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]

    def __lt__(self, other: Severity) -> bool:  # type: ignore[override]
        return self.rank < other.rank

    def __gt__(self, other: Severity) -> bool:  # type: ignore[override]
        return self.rank > other.rank

    @classmethod
    def parse(cls, raw: Any, default: Severity | None = None) -> Severity | None:
        """Parse an agent-supplied severity, tolerating casing and synonyms.

        Returns ``default`` (usually ``None``) when the value is unusable, so a
        hallucinated string can never crash the run.
        """
        if isinstance(raw, Severity):
            return raw
        if raw is None:
            return default
        text = str(raw).strip().upper()
        if text in _SEVERITY_RANK:
            return cls(text)
        synonyms = {
            "OK": cls.HEALTHY,
            "HEALTH": cls.HEALTHY,
            "NORMAL": cls.HEALTHY,
            "DEGRADED": cls.WARNING,
            "WARN": cls.WARNING,
            "FAIL": cls.CRITICAL,
            "FAILED": cls.CRITICAL,
            "ERROR": cls.CRITICAL,
            "OUTAGE": cls.CRITICAL,
        }
        return synonyms.get(text, default)

    @staticmethod
    def worst(severities: list[Severity]) -> Severity:
        """Return the most severe element of ``severities`` (``HEALTHY`` if empty)."""
        worst = Severity.HEALTHY
        for severity in severities:
            if severity.rank > worst.rank:
                worst = severity
        return worst


_SEVERITY_RANK: dict[Severity, int] = {
    Severity.HEALTHY: 0,
    Severity.WARNING: 1,
    Severity.CRITICAL: 2,
}

#: EC2 instance states, in the spelling the EC2 API uses.
STATE_PENDING = "pending"
STATE_RUNNING = "running"
STATE_SHUTTING_DOWN = "shutting-down"
STATE_TERMINATED = "terminated"
STATE_STOPPING = "stopping"
STATE_STOPPED = "stopped"

#: Synthetic state used when ``DescribeInstances`` does not return the instance.
STATE_NOT_FOUND = "not-found"

#: States that mean "this instance is not serving traffic".
OUT_OF_SERVICE_STATES = frozenset({STATE_STOPPED, STATE_STOPPING, STATE_SHUTTING_DOWN, STATE_TERMINATED})


@dataclass(frozen=True)
class Thresholds:
    """CPU thresholds (percent) evaluated against the window **average**."""

    cpu_warning_percent: float
    cpu_critical_percent: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_warning_percent": self.cpu_warning_percent,
            "cpu_critical_percent": self.cpu_critical_percent,
            "basis": "window average of CPUUtilization (percentage points)",
        }


@dataclass(frozen=True)
class MetricSummary:
    """Aggregated view of one CloudWatch metric for one instance in the window."""

    metric_name: str
    namespace: str = "AWS/EC2"
    unit: str | None = None
    average: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    total: float | None = None
    sample_count: int = 0
    period_seconds: int = 60
    first_sample_at: datetime | None = None
    last_sample_at: datetime | None = None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """True when CloudWatch returned at least one datapoint."""
        return self.sample_count > 0 and self.average is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "namespace": self.namespace,
            "unit": self.unit,
            "average": self.average,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "total": self.total,
            "sample_count": self.sample_count,
            "period_seconds": self.period_seconds,
            "first_sample_at": iso(self.first_sample_at),
            "last_sample_at": iso(self.last_sample_at),
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True)
class InstanceSnapshot:
    """Read-only view of an EC2 instance, straight from ``DescribeInstances``."""

    instance_id: str
    name: str
    state: str
    state_transition_reason: str | None = None
    instance_type: str | None = None
    availability_zone: str | None = None
    private_ip: str | None = None
    public_ip: str | None = None
    launch_time: datetime | None = None

    @property
    def is_out_of_service(self) -> bool:
        """True when the instance is not running, so it serves no traffic."""
        return self.state in OUT_OF_SERVICE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "name": self.name,
            "state": self.state,
            "state_transition_reason": self.state_transition_reason,
            "instance_type": self.instance_type,
            "availability_zone": self.availability_zone,
            "private_ip": self.private_ip,
            "public_ip": self.public_ip,
            "launch_time": iso(self.launch_time),
        }


@dataclass(frozen=True)
class InstanceObservation:
    """An instance state plus the metrics collected for it in the window."""

    snapshot: InstanceSnapshot
    metrics: dict[str, MetricSummary] = field(default_factory=dict)
    errors: tuple[str, ...] = ()

    def metric(self, name: str) -> MetricSummary | None:
        """Return the summary for ``name``, or ``None`` when it was not collected."""
        return self.metrics.get(name)

    @property
    def has_metrics(self) -> bool:
        return any(metric.available for metric in self.metrics.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance": self.snapshot.to_dict(),
            "data_available": self.has_metrics,
            "metrics": {name: metric.to_dict() for name, metric in sorted(self.metrics.items())},
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MonitoringPayload:
    """Everything the crew is allowed to know about the monitored instances."""

    generated_at: datetime
    window_start: datetime
    window_end: datetime
    region: str
    observations: tuple[InstanceObservation, ...]
    thresholds: Thresholds
    trigger: str = "unknown"
    errors: tuple[str, ...] = ()

    @property
    def window_minutes(self) -> int:
        return max(1, round((self.window_end - self.window_start).total_seconds() / 60))

    def observation(self, instance_id: str) -> InstanceObservation | None:
        for observation in self.observations:
            if observation.snapshot.instance_id == instance_id:
                return observation
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": iso(self.generated_at),
            "window": {
                "start": iso(self.window_start),
                "end": iso(self.window_end),
                "minutes": self.window_minutes,
            },
            "region": self.region,
            "trigger": self.trigger,
            "thresholds": self.thresholds.to_dict(),
            "instances_reviewed": len(self.observations),
            "instances": [observation.to_dict() for observation in self.observations],
            "collection_errors": list(self.errors),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise the payload for the CrewAI prompts."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
