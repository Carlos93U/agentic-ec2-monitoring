"""Turns the EC2 + CloudWatch reads into the single structured payload the crew sees.

This module is the *only* place that talks to AWS on behalf of the agents. From
here on, CrewAI works with plain data: no AWS client, no credentials, no tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta

from lambda_src.cloudwatch_metrics import DEFAULT_METRICS, CloudWatchMetricsCollector
from lambda_src.config import Settings
from lambda_src.ec2_monitor import EC2Monitor
from lambda_src.logging_utils import get_logger, log_event
from lambda_src.models import InstanceObservation, MetricSummary, MonitoringPayload, utcnow

LOGGER = get_logger("collector")


class MonitoringCollector:
    """Collects instance state plus metrics for the configured window."""

    def __init__(
        self,
        settings: Settings,
        *,
        ec2_monitor: EC2Monitor | None = None,
        metrics_collector: CloudWatchMetricsCollector | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._settings = settings
        self._ec2 = ec2_monitor or EC2Monitor(region=settings.aws_region)
        self._metrics = metrics_collector or CloudWatchMetricsCollector(
            region=settings.aws_region,
            metrics=DEFAULT_METRICS,
        )
        self._clock = clock

    def collect(self, *, trigger: str = "unknown") -> MonitoringPayload:
        """Read everything the crew needs for one monitoring window.

        A failure of either AWS API is recorded as an error in the payload instead
        of raising: a partial diagnosis ("I could not read the metrics, but the
        instance is stopped") is far more useful than no diagnosis at all.
        """
        window_end = self._clock()
        window_start = window_end - timedelta(minutes=self._settings.monitoring_window_minutes)
        errors: list[str] = []

        log_event(
            LOGGER,
            logging.INFO,
            "collection_started",
            trigger=trigger,
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            window_minutes=self._settings.monitoring_window_minutes,
            instance_ids=list(self._settings.monitored_instance_ids),
        )

        try:
            ec2_result = self._ec2.describe(self._settings.monitored_instance_ids)
            snapshots = list(ec2_result.snapshots)
            errors.extend(ec2_result.errors)
        except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
            errors.append(f"EC2 state unavailable: {type(exc).__name__}: {exc}")
            log_event(LOGGER, logging.ERROR, "ec2_collection_failed", error=str(exc))
            snapshots = []

        metrics_by_instance: dict[str, dict[str, MetricSummary]] = {}
        if snapshots:
            try:
                metrics_by_instance, metric_errors = self._metrics.collect(
                    [snapshot.instance_id for snapshot in snapshots], window_start, window_end
                )
                errors.extend(metric_errors)
            except Exception as exc:  # noqa: BLE001 - reported, never silently ignored
                errors.append(f"CloudWatch metrics unavailable: {type(exc).__name__}: {exc}")
                log_event(LOGGER, logging.ERROR, "cloudwatch_collection_failed", error=str(exc))

        observations: list[InstanceObservation] = []
        for snapshot in snapshots:
            instance_errors = [error for error in errors if error.startswith(f"{snapshot.instance_id}/")]
            observations.append(
                InstanceObservation(
                    # The Name tag is authoritative; MONITORED_INSTANCE_NAMES (written
                    # by the stack) is the fallback so the report stays readable even
                    # if the tag is missing or the caller lacks iam:CreateTags rights.
                    snapshot=(
                        snapshot
                        if snapshot.name != snapshot.instance_id
                        else replace(
                            snapshot, name=self._settings.instance_name(snapshot.instance_id)
                        )
                    ),
                    metrics=dict(metrics_by_instance.get(snapshot.instance_id, {})),
                    errors=tuple(instance_errors),
                )
            )

        payload = MonitoringPayload(
            generated_at=window_end,
            window_start=window_start,
            window_end=window_end,
            region=self._settings.aws_region,
            observations=tuple(observations),
            thresholds=self._settings.thresholds,
            trigger=trigger,
            errors=tuple(errors),
        )

        log_event(
            LOGGER,
            logging.INFO,
            "collection_finished",
            instances_reviewed=len(observations),
            instances_with_metrics=sum(1 for item in observations if item.has_metrics),
            states={item.snapshot.instance_id: item.snapshot.state for item in observations},
            errors=len(errors),
        )
        return payload
