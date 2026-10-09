"""The three CrewAI tasks and the JSON contract each one must satisfy.

Sequential process, one task per agent:

1. ``monitoring_task``  payload                 -> anomaly list per instance
2. ``diagnosis_task``   monitoring output        -> severity + narrative per instance
3. ``reporting_task``   diagnosis output         -> executive summary

Chaining is done with ``Task(context=[previous_task])``, which is what CrewAI 1.15
supports: ``Crew._get_context()`` aggregates the raw outputs of the referenced
tasks and prepends them to the prompt as "This is the context you're working
with: ...". The only template variables are therefore the crew inputs
(``payload_json`` and ``overall_status``), which ``crew.kickoff(inputs=...)``
interpolates.

Why the tasks ask for JSON *text* instead of using CrewAI's ``output_json``:

``output_json``/``output_pydantic`` make the model emit a tool call carrying the
object. That depends on the provider's tool-calling support being perfect for the
selected Bedrock model. Here the JSON is requested in plain text and parsed
defensively in :mod:`lambda_src.crew` (strict parser, then ``json-repair``, then a
fenced/embedded block). The run therefore degrades gracefully on any model, which
matters more than schema purity in a lab that runs 288 times a day.
"""

from __future__ import annotations

from typing import Any

from lambda_src.agents import ANALYST_RULES
from lambda_src.config import Settings
from lambda_src.models import Thresholds

MONITORING_JSON_CONTRACT = """Return exactly this JSON shape:

{
  "summary": "<one sentence describing the overall situation>",
  "instances": [
    {
      "instance_id": "<id copied from the payload>",
      "state_assessment": "RUNNING | DEGRADED | OUT_OF_SERVICE | UNKNOWN",
      "anomalies": [
        {
          "type": "state | cpu | network | disk | status_checks | data_availability",
          "detail": "<qualitative description, no numbers>",
          "impact": "<what it means for the workload>"
        }
      ]
    }
  ],
  "observations_worth_reporting": ["<short bullets a human should read>"]
}

Include every instance from the payload, in the same order. Use an empty
"anomalies" array when nothing looks wrong."""

DIAGNOSIS_JSON_CONTRACT = """Return exactly this JSON shape:

{
  "overall_status": "HEALTHY | WARNING | CRITICAL",
  "instances": [
    {
      "instance_id": "<id copied from the payload>",
      "status": "HEALTHY | WARNING | CRITICAL",
      "what_happened": "<2 short sentences: what is happening>",
      "affected_metric": "<metric name or 'instance_state', or 'none'>",
      "possible_cause": "<most likely cause in one sentence>",
      "recommendation": "<what a human operator should consider doing; never an action you performed>"
    }
  ]
}

Severity rules you must follow:
- An instance whose EC2 state is stopped, stopping, shutting-down or terminated is
  CRITICAL and out of service, whatever its metrics show.
- StatusCheckFailed greater than zero is CRITICAL.
- CPU average at or above the critical threshold is CRITICAL; at or above the
  warning threshold is WARNING.
- Otherwise HEALTHY. You may escalate to WARNING when the evidence is genuinely
  suspicious even without a threshold, but never downgrade a rule above.
- The overall status is the worst status of any instance."""

REPORTING_JSON_CONTRACT = """Return exactly this JSON shape:

{
  "executive_summary": "<max 3 sentences: situation, business impact, urgency>",
  "priority_actions": ["<concrete next step for a human>", "..."],
  "closing_note": "<one sentence, for example why nothing needs to be done>"
}

Do not include metric values, percentages or instance identifiers in this text:
they are added by the report generator."""


def build_tasks(agents: dict[str, Any], settings: Settings) -> tuple[Any, Any, Any]:
    """Create the three tasks, wired to the agents and the configured thresholds.

    Args:
        agents: Mapping produced by :func:`lambda_src.agents.build_agents`.
        settings: Validated Lambda settings (used for the thresholds).

    Returns:
        ``(monitoring_task, diagnosis_task, reporting_task)`` in execution order.
    """
    from crewai import Task

    thresholds: Thresholds = settings.thresholds
    threshold_block = (
        "Configuration for this run:\n"
        f"- monitoring window: {settings.monitoring_window_minutes} minutes\n"
        f"- CPU WARNING threshold: {thresholds.cpu_warning_percent:.0f}% (window average)\n"
        f"- CPU CRITICAL threshold: {thresholds.cpu_critical_percent:.0f}% (window average)\n"
        f"- instances reviewed: {', '.join(settings.monitored_instance_ids)}\n"
    )

    monitoring_task = Task(
        description=(
            f"{ANALYST_RULES}\n\n"
            "You receive the monitoring payload collected by AWS Lambda for a set of EC2 instances.\n"
            f"{threshold_block}\n"
            "The payload looks like this (all times are UTC, all metrics come from Amazon CloudWatch):\n"
            "{payload_json}\n\n"
            f"1. For every instance, read its EC2 state and say whether it is serving traffic.\n"
            "2. Look for anomalous signals in the metrics: sustained CPU pressure, network or disk activity "
            "that stands out, failed status checks, or a window with no datapoints at all.\n"
            "3. Remember that a stopped instance reports no metrics; that is an availability problem, not a "
            "performance problem, and it is the most severe one.\n\n"
            f"{MONITORING_JSON_CONTRACT}"
        ),
        expected_output=(
            "A JSON object listing, per instance, its state assessment and every anomaly found. "
            "Qualitative only: no numbers."
        ),
        agent=agents["monitoring"],
    )

    diagnosis_task = Task(
        description=(
            f"{ANALYST_RULES}\n\n"
            "The raw monitoring payload is below. The output of the Monitoring agent is appended to this "
            "prompt as the context you are working with: use it as a hypothesis, but trust the payload for "
            "facts.\n"
            f"{threshold_block}\n"
            "Original monitoring payload:\n{payload_json}\n\n"
            "Classify every instance, explain the evidence, propose the most likely cause and recommend what "
            "a human operator should consider doing. Do not perform any action.\n\n"
            f"{DIAGNOSIS_JSON_CONTRACT}"
        ),
        expected_output=(
            "A JSON object with an overall status and, per instance, a status, an explanation, the evidence "
            "metric, a possible cause and a recommendation."
        ),
        agent=agents["diagnosis"],
        # CrewAI prepends the raw output of the tasks listed here to this prompt
        # ("This is the context you're working with: ..."). Verified against
        # crewai 1.15.23: Crew._get_context() -> Task.prompt_context ->
        # format_task_with_context().
        context=[monitoring_task],
    )

    reporting_task = Task(
        description=(
            f"{ANALYST_RULES}\n\n"
            "The raw monitoring payload is below and the conclusions of the Diagnosis agent are appended to "
            "this prompt as the context you are working with.\n\n"
            # No f-string on purpose: {overall_status} is a CrewAI input that
            # crew.kickoff(inputs=...) interpolates, like {payload_json} below.
            "Overall status decided by the deterministic evaluator: {overall_status}\n\n"
            "Monitoring payload:\n{payload_json}\n\n"
            "Write for someone who has five seconds: what the situation is, why it matters, and what to do "
            "next. Keep it operational and specific.\n\n"
            f"{REPORTING_JSON_CONTRACT}"
        ),
        expected_output="A JSON object with the executive summary, the priority actions and a closing note.",
        agent=agents["reporting"],
        context=[diagnosis_task],
    )

    return monitoring_task, diagnosis_task, reporting_task
