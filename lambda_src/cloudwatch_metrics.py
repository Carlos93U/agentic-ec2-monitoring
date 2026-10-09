"""Read-only CloudWatch metric collection.

Design note: this module uses ``GetMetricStatistics`` (one call per instance and
metric) instead of ``GetMetricData`` (one batched call for everything).

* ``GetMetricStatistics`` returns ``Average``, ``Minimum``, ``Maximum``, ``Sum``,
  the ``Unit`` and the ``Timestamp`` in a **single** response, which is exactly
  the contract the lab asks for.
* ``GetMetricData`` returns a single ``Stat`` per query, so getting four
  statistics means four queries per metric (48 queries for this lab) and it still
  does not report the unit.

With ``Period=60`` (1-minute resolution, enabled by EC2 detailed monitoring) the
returned datapoints are one-minute buckets, which makes the window aggregates
trivially correct:

* average = mean of the bucket averages,
* minimum = smallest bucket minimum,
* maximum = largest bucket maximum,
* total   = sum of the bucket sums (this is the meaningful one for
  NetworkIn/NetworkOut/EBSReadOps/EBSWriteOps).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from lambda_src.logging_utils import get_logger, log_event
from lambda_src.models import MetricSummary

LOGGER = get_logger("cloudwatch_metrics")

#: Datapoints CloudWatch publishes with a delay; the window end is pulled back by
#: this amount so a run that fires exactly on the boundary still sees data.
PUBLICATION_LAG_SECONDS = 120


class CloudWatchReadError(RuntimeError):
    """Raised when metrics cannot be read at all."""


class CloudWatchClientProtocol(Protocol):
    """Structural type for the bit of the CloudWatch client this module uses."""

    def get_metric_statistics(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class MetricSpec:
    """One metric the lab analyses."""

    name: str
    statistics: tuple[str, ...] = ("Average", "Minimum", "Maximum", "Sum")
    namespace: str = "AWS/EC2"
    #: How the report should render the value (``percent``/``bytes``/``count``/``checks``).
    display: str = "count"
    #: Human label used in the report.
    label: str = ""
    #: Extra guidance injected into the agent prompts.
    hint: str = ""


#: The six metrics required by the lab specification.
DEFAULT_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        name="CPUUtilization",
        display="percent",
        label="CPU",
        hint="Whole-system CPU utilisation. The only metric compared against thresholds.",
    ),
    MetricSpec(
        name="NetworkIn",
        display="bytes",
        label="Network In",
        hint="Bytes received. No baseline in this lab: judge relative magnitude only.",
    ),
    MetricSpec(
        name="NetworkOut",
        display="bytes",
        label="Network Out",
        hint="Bytes sent. No baseline in this lab: judge relative magnitude only.",
    ),
    MetricSpec(
        name="EBSReadOps",
        display="count",
        label="Disk Read Ops",
        hint="Read operations completed against the attached EBS volumes. Spikes can indicate backups or batch jobs.",
    ),
    MetricSpec(
        name="EBSWriteOps",
        display="count",
        label="Disk Write Ops",
        hint="Write operations completed against the attached EBS volumes. Sustained writes can indicate log churn.",
    ),
    MetricSpec(
        name="StatusCheckFailed",
        display="checks",
        label="Status Checks",
        hint="Number of failed EC2 instance status checks in the window. Any value above zero is CRITICAL.",
    ),
)

#: Convenience lookup for tests and report rendering.
METRICS_BY_NAME: dict[str, MetricSpec] = {metric.name: metric for metric in DEFAULT_METRICS}


class CloudWatchMetricsCollector:
    """Collects the configured metrics for a set of instances in a time window."""

    def __init__(
        self,
        client: CloudWatchClientProtocol | None = None,
        *,
        region: str | None = None,
        period_seconds: int = 60,
        metrics: tuple[MetricSpec, ...] = DEFAULT_METRICS,
    ) -> None:
        self._client = client
        self._region = region
        self.period_seconds = period_seconds
        self.metrics = metrics

    @property
    def client(self) -> CloudWatchClientProtocol:
        """Lazily create the boto3 client (tests inject their own fake)."""
        if self._client is None:
            try:
                self._client = boto3.client("cloudwatch", region_name=self._region)
            except (BotoCoreError, ClientError) as exc:  # pragma: no cover - credential issues
                raise CloudWatchReadError(f"Could not create a CloudWatch client: {exc}") from exc
        return self._client

    def collect(
        self,
        instance_ids: tuple[str, ...] | list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> tuple[dict[str, dict[str, MetricSummary]], list[str]]:
        """Collect every configured metric for every instance.

        Args:
            instance_ids: Instances to query.
            window_start: Inclusive start of the analysis window.
            window_end: Exclusive end of the analysis window, pulled back by the
                CloudWatch publication lag.

        Returns:
            ``({instance_id: {metric_name: MetricSummary}}, errors)``. Errors are
            per metric, so one missing metric never hides the others.
        """
        effective_end = window_end - _seconds(PUBLICATION_LAG_SECONDS)
        results: dict[str, dict[str, MetricSummary]] = {}
        errors: list[str] = []

        for instance_id in instance_ids:
            per_metric: dict[str, MetricSummary] = {}
            for spec in self.metrics:
                try:
                    response = self.client.get_metric_statistics(
                        Namespace=spec.namespace,
                        MetricName=spec.name,
                        Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                        StartTime=window_start,
                        EndTime=effective_end,
                        Period=self.period_seconds,
                        Statistics=list(spec.statistics),
                    )
                except ClientError as exc:
                    code = str(exc.response.get("Error", {}).get("Code", ""))
                    message = str(exc.response.get("Error", {}).get("Message", exc))
                    errors.append(f"{instance_id}/{spec.name}: CloudWatch error {code} ({message})")
                    per_metric[spec.name] = _unavailable(spec, f"cloudwatch_error:{code}")
                    continue
                except BotoCoreError as exc:
                    errors.append(f"{instance_id}/{spec.name}: {exc}")
                    per_metric[spec.name] = _unavailable(spec, "client_error")
                    continue

                per_metric[spec.name] = summarise_datapoints(
                    spec, response, period_seconds=self.period_seconds
                )

            results[instance_id] = per_metric
            available = sum(1 for summary in per_metric.values() if summary.available)
            log_event(
                LOGGER,
                logging.INFO,
                "cloudwatch_metrics_collected",
                instance_id=instance_id,
                metrics_requested=len(self.metrics),
                metrics_available=available,
                window_start=window_start.isoformat(),
                window_end=effective_end.isoformat(),
                period_seconds=self.period_seconds,
            )

        return results, errors


def summarise_datapoints(
    spec: MetricSpec, response: dict[str, Any], *, period_seconds: int = 60
) -> MetricSummary:
    """Fold CloudWatch datapoints into a single :class:`MetricSummary`."""
    datapoints = response.get("Datapoints") or []
    if not datapoints:
        # CloudWatch returns an empty list (not an error) when the window contains
        # no data: a stopped instance, or a window that starts before the metric
        # existed. That is a valid, meaningful outcome.
        return _unavailable(spec, "no_datapoints")

    averages = _present([_float(point.get("Average")) for point in datapoints])
    minimums = _present([_float(point.get("Minimum")) for point in datapoints])
    maximums = _present([_float(point.get("Maximum")) for point in datapoints])
    sums = _present([_float(point.get("Sum")) for point in datapoints])

    timestamps = sorted(point["Timestamp"] for point in datapoints if point.get("Timestamp") is not None)
    unit = next((point.get("Unit") for point in datapoints if point.get("Unit")), None)

    if not averages and not maximums and not minimums and not sums:
        return _unavailable(spec, "empty_datapoints")

    return MetricSummary(
        metric_name=spec.name,
        namespace=spec.namespace,
        unit=unit,
        average=round(sum(averages) / len(averages), 4) if averages else None,
        minimum=round(min(minimums), 4) if minimums else None,
        maximum=round(max(maximums), 4) if maximums else None,
        total=round(sum(sums), 4) if sums else None,
        sample_count=len(datapoints),
        period_seconds=period_seconds,
        first_sample_at=timestamps[0] if timestamps else None,
        last_sample_at=timestamps[-1] if timestamps else None,
    )


def _present(values: list[float | None]) -> list[float]:
    """Drop the ``None`` entries produced by malformed or partial datapoints."""
    return [value for value in values if value is not None]


def _unavailable(spec: MetricSpec, reason: str) -> MetricSummary:
    return MetricSummary(metric_name=spec.name, namespace=spec.namespace, unavailable_reason=reason)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _seconds(count: int) -> timedelta:
    return timedelta(seconds=count)
