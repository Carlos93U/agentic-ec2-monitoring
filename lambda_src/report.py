"""Renders the notification that actually reaches the inbox.

Design rule, and the reason this module exists as a separate one: **every number
in the message is formatted from the collected payload**, never from a model
answer. The crew only contributes prose (summary, cause, recommendation), and the
prose is clearly delimited so it cannot be mistaken for data.

Both a plain-text and an HTML body are produced; SNS uses the HTML one when it is
available because email clients render it far better than monospace ASCII art.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lambda_src.cloudwatch_metrics import METRICS_BY_NAME
from lambda_src.crew import CrewNarrative
from lambda_src.evaluator import Assessment, Finding, MergedAssessment
from lambda_src.models import MonitoringPayload, Severity

#: Short, emoji-free severity labels: email clients and log aggregators mangle
#: emoji, and some corporate gateways strip them entirely.
SEVERITY_LABEL: Mapping[Severity, str] = {
    Severity.HEALTHY: "HEALTHY",
    Severity.WARNING: "WARNING",
    Severity.CRITICAL: "CRITICAL",
}

#: One-line explanation per finding code, used when no agent narrative exists.
#: Keys must stay in sync with the codes emitted by lambda_src.evaluator.
FINDING_EXPLANATION: Mapping[str, str] = {
    "instance_out_of_service": "The instance is not running, so it serves no traffic.",
    "instance_starting": "The instance is still starting; a launch takes a couple of minutes.",
    "instance_not_found": "This instance id does not exist in this account and region.",
    "instance_unknown_state": "The instance reported a state this lab does not know how to classify.",
    "status_checks_failed": "One or more EC2 status checks are failing on the underlying host.",
    "no_metric_data": "CloudWatch returned no CPU datapoints for the window, so performance cannot be confirmed.",
    "cpu_high": "Average CPU over the window reached the warning threshold.",
    "cpu_critical": "Average CPU over the window reached the critical threshold.",
    "metric_unavailable": "A network metric returned no datapoints for the window.",
}

#: Metric rendering. Labels and display styles come from the MetricSpec table in
#: lambda_src.cloudwatch_metrics, so the report, the tests and the agent prompts
#: can never disagree about a metric.
def metric_label(metric_name: str) -> str:
    """Human label for a metric, falling back to the raw name."""
    spec = METRICS_BY_NAME.get(metric_name)
    return spec.label if spec and spec.label else metric_name


def metric_display(metric_name: str) -> str:
    """Rendering style for a metric: ``percent``/``bytes``/``count``/``checks``."""
    spec = METRICS_BY_NAME.get(metric_name)
    return spec.display if spec else "count"


def format_metric_value(metric_name: str, value: float | None, unit: str | None = None) -> str:
    """Format one metric value using its declared display style.

    Args:
        metric_name: CloudWatch metric name, used to pick the style.
        value: The value to render, or ``None`` for "no data".
        unit: Unit reported by CloudWatch, used only as a fallback.
    """
    if value is None:
        return "N/A"

    display = metric_display(metric_name)
    if display == "percent" or (unit == "Percent" and display == "count"):
        return f"{value:.1f}%"
    if display == "bytes" or unit == "Bytes":
        return f"{_human_bytes(value)}/s"
    if display == "checks":
        return f"{value:.0f} failed"
    return f"{value:.1f}"

HTML_TEMPLATE = """<html>
<body style="font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color:#1b1f23; font-size:14px;">
  <div style="border-left:4px solid {color}; padding:8px 12px; background:{background};">
    <span style="font-weight:700; font-size:15px;">{severity}</span>
    <span style="color:#57606a;">&nbsp;&middot;&nbsp;{scope}</span>
  </div>
  <p style="margin:16px 0 4px;"><strong>Summary</strong></p>
  <p style="margin:0 0 12px;">{summary}</p>
  {sections}
  {footer}
</body>
</html>
"""

SEVERITY_COLOR: Mapping[Severity, tuple[str, str]] = {
    Severity.HEALTHY: ("#2da44e", "#eaf7ee"),
    Severity.WARNING: ("#bf8700", "#fff8e5"),
    Severity.CRITICAL: ("#cf222e", "#fdecec"),
}


@dataclass(frozen=True)
class Notification:
    """Everything the publisher needs to send one email."""

    subject: str
    text_body: str
    html_body: str
    severity: Severity
    degraded: bool
    parse_issues: tuple[str, ...] = ()


def build_subject(severity: Severity, scope: str, degraded: bool) -> str:
    """Subject line: severity first, because that is what gets triaged."""
    prefix = SEVERITY_LABEL[severity]
    if degraded:
        prefix = f"{prefix} (degraded: no LLM)"
    return f"[{prefix}] {scope}"


def _human_bytes(value: float) -> str:
    """Render a bytes-per-second value with a readable magnitude."""
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable, kept for exhaustiveness


def build_notification(
    payload: MonitoringPayload,
    assessments: Sequence[MergedAssessment],
    overall: Severity,
    narrative: CrewNarrative | None,
    *,
    degraded_reason: str | None = None,
) -> Notification:
    """Assemble the notification for one invocation.

    Args:
        payload: The collected data (source of every number in the message).
        assessments: Deterministic assessments, already merged with the agent.
        overall: Fleet status after merging.
        narrative: Crew output, or ``None`` when the crew failed.
        degraded_reason: Why the narrative is missing, for the footer.

    Returns:
        A :class:`Notification` with subject, text and HTML bodies.
    """
    degraded = narrative is None
    scope = _scope(payload, assessments)
    subject = build_subject(overall, scope, degraded)

    summary = _summary(payload, assessments, overall, narrative, degraded, degraded_reason)
    text_sections = [_text_instance_section(payload, item) for item in assessments]
    html_sections = [_html_instance_section(payload, item) for item in assessments]
    footer = _footer(payload, narrative, degraded, degraded_reason)

    text_body = "\n".join(
        part
        for part in [
            SEVERITY_LABEL[overall] + f" | {scope}",
            "",
            "SUMMARY",
            summary,
            "",
            "INSTANCES",
            *text_sections,
            footer,
        ]
        if part is not None
    )

    color, background = SEVERITY_COLOR[overall]
    html_body = HTML_TEMPLATE.format(
        color=color,
        background=background,
        severity=html.escape(SEVERITY_LABEL[overall]),
        scope=html.escape(scope),
        summary=html.escape(summary).replace("\n", "<br>"),
        sections="".join(html_sections),
        footer=html.escape(footer) if footer else "",
    )

    return Notification(
        subject=subject,
        text_body=text_body,
        html_body=html_body,
        severity=overall,
        degraded=degraded,
        parse_issues=narrative.parse_issues if narrative else (),
    )


# -- scope / summary ----------------------------------------------------- #
def _scope(payload: MonitoringPayload, assessments: Sequence[MergedAssessment]) -> str:
    healthy = sum(1 for item in assessments if item.severity is Severity.HEALTHY)
    return (
        f"{len(assessments)} EC2 instance(s) in {payload.region} - "
        f"{healthy} healthy, {len(assessments) - healthy} needing attention"
    )


def _summary(
    payload: MonitoringPayload,
    assessments: Sequence[MergedAssessment],
    overall: Severity,
    narrative: CrewNarrative | None,
    degraded: bool,
    degraded_reason: str | None,
) -> str:
    """Executive summary: the model's text when available, facts otherwise."""
    if narrative and narrative.summary:
        return narrative.summary
    if narrative and narrative.monitoring_summary:
        return narrative.monitoring_summary
    if not degraded:
        return (
            f"Fleet status {SEVERITY_LABEL[overall]}. The agents did not return an executive summary; "
            "see the per-instance findings below."
        )
    counts = _severity_counts(assessments)
    breakdown = ", ".join(
        f"{count} {SEVERITY_LABEL[severity].lower()}" for severity, count in counts if count
    )
    return (
        f"Automated check of {len(assessments)} instance(s) over the last "
        f"{payload.window_minutes} minutes: {breakdown or 'no data'}. "
        "Generated without the LLM narrative because the analysis crew did not complete "
        f"({degraded_reason or 'unknown reason'}). The findings below are computed directly from "
        "EC2 and CloudWatch data."
    )


def _severity_counts(assessments: Sequence[MergedAssessment]) -> list[tuple[Severity, int]]:
    """Count instances per severity, worst first."""
    order = (Severity.CRITICAL, Severity.WARNING, Severity.HEALTHY)
    counts = dict.fromkeys(order, 0)
    for item in assessments:
        counts[item.severity] += 1
    return [(severity, counts[severity]) for severity in order]


# -- per instance --------------------------------------------------------- #
def _text_instance_section(payload: MonitoringPayload, item: MergedAssessment) -> str:
    observation = payload.observation(item.instance_id)
    snapshot = observation.snapshot if observation else None
    lines = [
        f"- {item.name} ({item.instance_id}) - {SEVERITY_LABEL[item.severity]}",
        f"  state: {item.state}",
    ]

    if observation:
        for name, metric in sorted(observation.metrics.items()):
            label = metric_label(name)
            if metric.available:
                lines.append(
                    f"  {label}: avg {format_metric_value(name, metric.average, metric.unit)}"
                    f" / max {format_metric_value(name, metric.maximum, metric.unit)}"
                    f" ({metric.sample_count} samples)"
                )
            else:
                reason = metric.unavailable_reason or "no datapoints"
                lines.append(f"  {label}: no data ({reason})")

    for finding in item.findings:
        lines.append(f"  ! {finding.message}")

    narrative = item.agent_narrative
    if narrative:
        for label, key in (
            ("Analysis", "what_happened"),
            ("Likely cause", "possible_cause"),
            ("Recommended action", "recommendation"),
        ):
            value = narrative.get(key)
            if value:
                lines.append(f"  {label}: {value}")
    elif item.severity is not Severity.HEALTHY:
        explanation = next(
            (FINDING_EXPLANATION[finding.code] for finding in item.findings if finding.code in FINDING_EXPLANATION),
            None,
        )
        if explanation:
            lines.append(f"  Recommended action: {explanation} Consider checking the instance before acting.")

    if item.was_overridden:
        lines.append("  Note: the AI agent tried to downgrade this finding and was overruled by the safety rules.")

    if snapshot and snapshot.is_out_of_service and snapshot.state_transition_reason:
        lines.append(f"  EC2 transition reason: {snapshot.state_transition_reason}")

    return "\n".join(lines)


def _html_instance_section(payload: MonitoringPayload, item: MergedAssessment) -> str:
    observation = payload.observation(item.instance_id)
    color, background = SEVERITY_COLOR[item.severity]
    rows = []

    rows.append(
        f"<tr><td colspan=2 style='padding:6px 0;'><strong>{html.escape(item.name)}</strong> "
        f"<span style='color:#57606a;'>({html.escape(item.instance_id)})</span></td></tr>"
    )
    rows.append(f"<tr><td style='padding:2px 12px 2px 0;color:#57606a;'>EC2 state</td><td>{html.escape(item.state)}</td></tr>")

    if observation:
        for name, metric in sorted(observation.metrics.items()):
            label = metric_label(name)
            if metric.available:
                value = (
                    f"avg {format_metric_value(name, metric.average, metric.unit)} / "
                    f"max {format_metric_value(name, metric.maximum, metric.unit)} "
                    f"({metric.sample_count} samples)"
                )
            else:
                value = f"no data ({metric.unavailable_reason or 'no datapoints'})"
            rows.append(
                f"<tr><td style='padding:2px 12px 2px 0;color:#57606a;'>{html.escape(label)}</td>"
                f"<td>{html.escape(value)}</td></tr>"
            )

    for finding in item.findings:
        rows.append(
            f"<tr><td colspan=2 style='padding:2px 0;'>{html.escape(finding.message)}</td></tr>"
        )

    for label, key in (
        ("Analysis", "what_happened"),
        ("Likely cause", "possible_cause"),
        ("Recommended action", "recommendation"),
    ):
        value = str(item.agent_narrative.get(key) or "")
        if value:
            rows.append(
                f"<tr><td style='padding:2px 12px 2px 0;color:#57606a;'>{label}</td><td>{html.escape(value)}</td></tr>"
            )

    if item.was_overridden:
        rows.append(
            "<tr><td colspan=2 style='padding:2px 0;color:#bf8700;'>The AI agent tried to downgrade this "
            "finding; the safety rules overruled it.</td></tr>"
        )

    return (
        f"<div style='border:1px solid #d0d7de;border-left:4px solid {color};"
        f"background:{background};border-radius:6px;padding:10px 12px;margin:12px 0;'>"
        f"<div style='font-weight:600;margin-bottom:6px;'>{html.escape(SEVERITY_LABEL[item.severity])}</div>"
        "<table style='border-collapse:collapse;width:100%;'>"
        + "".join(rows)
        + "</table></div>"
    )


# -- footer -------------------------------------------------------------- #
def _footer(
    payload: MonitoringPayload,
    narrative: CrewNarrative | None,
    degraded: bool,
    degraded_reason: str | None,
) -> str:
    """Provenance line: where the numbers come from and how they were analysed."""
    parts = [
        f"Data: EC2 DescribeInstances + CloudWatch GetMetricStatistics, {payload.window_minutes} min window "
        f"ending {payload.window_end:%Y-%m-%d %H:%M:%S} UTC. Trigger: {payload.trigger}.",
        f"CPU thresholds: warning {payload.thresholds.cpu_warning_percent:.0f}%, "
        f"critical {payload.thresholds.cpu_critical_percent:.0f}% (window average).",
    ]
    if degraded:
        parts.append(
            f"Analysis: crew unavailable ({degraded_reason or 'unknown reason'}). Severities and numbers are "
            "deterministic and complete; only the narrative is missing."
        )
    else:
        parts.append(
            "Analysis: CrewAI (Monitoring -> Diagnosis with extended thinking -> Reporting) on Amazon Bedrock. "
            "Severities and numbers are computed by the collector, not by the model."
        )
        if narrative and narrative.priority_actions:
            parts.append("Priority actions: " + "; ".join(narrative.priority_actions))
        if narrative and narrative.parse_issues:
            parts.append("Partial output: " + "; ".join(narrative.parse_issues))
    return "\n".join(parts)


def to_plain_dict(
    assessments: Iterable[Assessment],
) -> list[dict[str, Any]]:
    """Serialisable view of the assessments, used by the Lambda response."""
    return [assessment.to_dict() for assessment in assessments]


def findings_summary(findings: Iterable[Finding]) -> str:
    """One-line join of finding messages, handy for logs."""
    return "; ".join(finding.message for finding in findings)
