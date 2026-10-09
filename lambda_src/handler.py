"""Lambda entry point: collect -> evaluate -> analyse -> render -> publish.

The flow is deliberately split so that every step is testable on its own and so
that the failure of one step cannot hide the result of the previous ones:

.. code-block:: text

    collector.collect()      EC2 + CloudWatch   (read-only AWS calls)
      -> evaluator.evaluate_payload()          deterministic severities
      -> CrewRunner.run()                      Bedrock + CrewAI  (soft failure)
      -> evaluator.merge_assessments()         agent may escalate, never downgrade
      -> report.build_notification()           numbers from the payload
      -> SnsPublisher.publish()                email

Everything the crew produces is optional. Everything the collector produces is
mandatory. The handler therefore always returns the deterministic assessment, and
adds ``degraded: true`` plus the reason when the crew could not run.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from lambda_src.collector import MonitoringCollector
from lambda_src.config import Settings
from lambda_src.crew import CrewExecutionError, CrewNarrative, CrewRunner
from lambda_src.evaluator import evaluate_payload, merge_assessments, overall_severity
from lambda_src.logging_utils import (
    configure_logging,
    get_logger,
    log_event,
    reset_log_context,
    set_log_context,
)
from lambda_src.models import Severity
from lambda_src.report import build_notification
from lambda_src.sns_publisher import PublishError, SnsPublisher

LOGGER = get_logger("handler")


def _resolve_trigger(event: dict[str, Any] | None) -> str:
    """Describe what invoked the function, for the report footer and the logs."""
    if not event:
        return "manual invoke (no event)"
    detail = event.get("detail-type") or event.get("source") or event.get("detail")
    if isinstance(detail, dict):
        return str(detail)
    return str(detail or "unknown")


def lambda_handler(
    event: dict[str, Any] | None = None,
    context: Any = None,
    *,
    collector: MonitoringCollector | None = None,
    crew_runner: Any | None = None,
    publisher: Any | None = None,
) -> dict[str, Any]:
    """Run one monitoring cycle.

    Args:
        event: Invocation event. EventBridge Scheduler sends ``{}``; a manual
            invoke passes ``None``. Only the trigger description is read from it.
        context: Lambda context, used for the remaining-time warning.
        collector: Test seam. The AWS collector is built from settings by default.
        crew_runner: Test seam, any object with ``run(payload, overall_status=...)``.
        publisher: Test seam, any object with ``publish(notification)``.

    Returns:
        A JSON-serialisable dict with the deterministic assessment, the overall
        severity, whether the crew ran, and the SNS result.

    Raises:
        lambda_src.config.ConfigurationError: when the environment is invalid.
            This is a deployment bug and must fail loudly rather than silently
            skip a cycle.
    """
    started = time.monotonic()
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    # Bind the request id into the logging context so every line of this run can
    # be correlated in Logs Insights without threading it through each call.
    # Falls back to a synthetic id when invoked manually (no Lambda context).
    invocation_id = getattr(context, "aws_request_id", None) or f"local-{uuid4().hex[:12]}"
    context_token = set_log_context(invocation_id=invocation_id, trigger=_resolve_trigger(event))
    try:
        return _run(event, settings=settings, collector=collector, crew_runner=crew_runner,
                     publisher=publisher, context=context, started=started)
    finally:
        # Always reset, including on the exceptions the collector can raise, or
        # the id would leak into the next invocation of a warm container.
        reset_log_context(context_token)


def _run(
    event: dict[str, Any] | None,
    *,
    settings: Settings,
    collector: MonitoringCollector | None,
    crew_runner: Any | None,
    publisher: Any | None,
    context: Any | None,
    started: float,
) -> dict[str, Any]:
    """The monitoring cycle itself, split out so the log context can be unwound."""
    log_event(LOGGER, logging.INFO, "invocation_started", **settings.describe())

    collector = collector or MonitoringCollector(settings)
    payload = collector.collect(trigger=_resolve_trigger(event))
    assessments = evaluate_payload(payload)
    deterministic_overall = overall_severity(assessments)

    narrative: CrewNarrative | None = None
    degraded_reason: str | None = None
    try:
        runner = crew_runner or CrewRunner(settings)
        narrative = runner.run(payload, overall_status=deterministic_overall)
    except CrewExecutionError as exc:
        # Soft failure on purpose: a Bedrock outage must not silence monitoring.
        degraded_reason = str(exc)
        log_event(
            LOGGER,
            logging.ERROR,
            "crew_degraded",
            error=degraded_reason,
            fallback="deterministic report without LLM narrative",
        )

    merged = merge_assessments(
        assessments,
        narrative.agent_statuses if narrative else {},
        narrative.agent_narratives if narrative else {},
    )
    overall = overall_severity(merged)
    notification = build_notification(
        payload,
        merged,
        overall,
        narrative,
        degraded_reason=degraded_reason,
    )

    # ALWAYS_NOTIFY=false means "stay quiet while everything is fine", not
    # "do not run the analysis". The assessment above is always produced.
    should_notify = settings.always_notify or overall is not Severity.HEALTHY

    published: dict[str, Any] | None = None
    publish_error: str | None = None
    skipped_reason: str | None = None
    if should_notify:
        try:
            client = publisher or SnsPublisher(settings.sns_topic_arn, settings.aws_region)
            published = client.publish(notification)
        except PublishError as exc:
            publish_error = str(exc)
            log_event(LOGGER, logging.ERROR, "publish_failed", error=publish_error)
    else:
        skipped_reason = (
            f"overall severity is {overall.value} and ALWAYS_NOTIFY is false; "
            "the notification was not published"
        )
        log_event(LOGGER, logging.INFO, "publish_skipped", reason=skipped_reason)

    duration_ms = int((time.monotonic() - started) * 1000)

    result: dict[str, Any] = {
        "overall_severity": overall.value,
        "deterministic_severity": deterministic_overall.value,
        "crew_available": narrative is not None,
        "degraded": narrative is None,
        "degraded_reason": degraded_reason,
        "should_notify": bool(should_notify),
        "instances": [
            {
                "instance_id": item.instance_id,
                "name": item.name,
                "state": item.state,
                "severity": item.severity.value,
                "agent_severity": item.agent_severity.value if item.agent_severity else None,
                "guard_applied": item.guard_applied,
                "findings": [finding.to_dict() for finding in item.findings],
            }
            for item in merged
        ],
        "window": {
            "start": payload.window_start.isoformat(),
            "end": payload.window_end.isoformat(),
            "minutes": payload.window_minutes,
        },
        "collection_errors": list(payload.errors),
        "notification": {
            "subject": notification.subject,
            "should_notify": bool(should_notify),
            "published": published is not None,
            "message_id": (published or {}).get("message_id"),
            "skipped_reason": skipped_reason,
            "error": publish_error,
        },
        "duration_ms": duration_ms,
    }

    if context is not None and getattr(context, "get_remaining_time_in_millis", None):
        remaining = context.get_remaining_time_in_millis() - duration_ms
        log_event(
            LOGGER,
            logging.WARNING if remaining < 15000 else logging.INFO,
            "lambda_time_remaining_ms",
            duration_ms=duration_ms,
            remaining_ms=remaining,
        )

    log_event(
        LOGGER,
        logging.INFO,
        "invocation_finished",
        overall=overall.value,
        crew_available=narrative is not None,
        published=published is not None,
        duration_ms=duration_ms,
    )
    return result

