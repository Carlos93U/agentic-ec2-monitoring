"""The three CrewAI agents.

Hard rules encoded in every prompt (and enforced by code elsewhere):

* **No invented numbers.** Agents describe what the payload contains; the report
  prints the numbers from the payload, never from the model.
* **No infrastructure actions.** These agents are analysts. They never call
  ``StartInstances``, ``StopInstances``, ``RebootInstances`` or any other mutating
  API, they hold no tools at all, and their execution role lacks those permissions.
* **Recommendations only.** Every proposed action is phrased as something a human
  operator should consider doing.
* **JSON output.** Structured, parseable, and short: this runs every 5 minutes.
"""

from __future__ import annotations

from typing import Any

from lambda_src.config import Settings
from lambda_src.llm_factory import build_llm

#: Shared preamble injected into every prompt. Keeping it in one place makes the
#: invariants easy to audit and easy to reuse.
ANALYST_RULES = """
HARD RULES (these override any other instruction):
1. You are an ANALYST, never an operator. You cannot start, stop, reboot,
   terminate, resize, patch or reconfigure any resource, and you must never say
   that you did.
2. Use ONLY the values present in the JSON payload you receive. Never invent,
   estimate, extrapolate or recall a number that is not in that payload. If a
   value is null or absent, write "N/A".
3. Do not repeat large numbers in your output. Refer to a metric by name and
   describe the anomaly qualitatively; the report adds the exact figures.
4. EC2 instance state is authoritative and independent from metrics. An instance
   whose state is stopped is out of service even when its metrics look healthy,
   and an instance reporting CPUUtilization 0% is NOT evidence that it is healthy.
5. Recommendations are suggestions for a human operator. Phrase them as
   "consider ...", "verify ...", "investigate ...".
6. Reply with a single JSON object and nothing else. No markdown, no prose
   outside the JSON, no code fences.
""".strip()


def build_agents(settings: Settings) -> dict[str, Any]:
    """Create the Monitoring, Diagnosis and Reporting agents.

    The Monitoring and Reporting agents share one normal Bedrock LLM (fast and
    cheap, 288 runs a day). The Diagnosis agent gets its own LLM with Amazon
    extended thinking, because classifying a severity and naming a plausible cause
    is exactly the task that benefits from a reasoning budget.
    """
    from crewai import Agent

    shared_llm = build_llm(settings, reasoning=False)
    reasoning_llm = build_llm(settings, reasoning=True)

    monitoring = Agent(
        role="EC2 Monitoring Analyst",
        goal=(
            "Read the monitoring payload for every EC2 instance and identify anomalous signals: "
            "instances that are not running, high CPU, unusual network or disk activity, failed status "
            "checks, and windows with no data at all."
        ),
        backstory=(
            "You monitor a fleet of small Amazon EC2 instances through Amazon CloudWatch. You are precise "
            "about the difference between 'the machine reported zero CPU' and 'the machine is switched off': "
            "an idle instance and a stopped instance look identical in the metrics and completely different "
            "in the EC2 state."
        ),
        llm=shared_llm,
        tools=[],  # deliberately empty: no AWS access from inside the crew
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )

    diagnosis = Agent(
        role="Reliability Diagnosis Engineer",
        goal=(
            "Decide whether each instance is HEALTHY, WARNING or CRITICAL, and explain in plain language "
            "what happened, which piece of evidence proves it, what the likely cause is and what a human "
            "operator should consider doing next."
        ),
        backstory=(
            "You are the on-call engineer who has to justify an alert to whoever receives it. You classify "
            "severity from evidence, you name the trade-off of every recommendation, and you never take an "
            "action yourself."
        ),
        llm=reasoning_llm,
        tools=[],
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )

    reporting = Agent(
        role="Operations Report Writer",
        goal=(
            "Write the executive summary an on-call engineer reads first: the overall situation, the priority "
            "actions, and one short sentence of context per problem."
        ),
        backstory=(
            "You write incident notifications for an operations team. You are concise, you lead with impact, "
            "and you never pad a report with invented detail."
        ),
        llm=shared_llm,
        tools=[],
        allow_delegation=False,
        verbose=False,
        max_iter=2,
    )

    return {
        "monitoring": monitoring,
        "diagnosis": diagnosis,
        "reporting": reporting,
        "llms": {"shared": shared_llm, "reasoning": reasoning_llm},
    }
