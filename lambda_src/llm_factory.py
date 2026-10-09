"""CrewAI LLM objects backed by Amazon Bedrock.

CrewAI 1.15 has a **native** Bedrock provider that talks to the Converse API
through boto3, so the extra to install is ``crewai[bedrock]`` and LiteLLM is not
needed at all (it was quarantined on PyPI after a security incident).

Two LLM objects are created:

============================  ==========================================
Object                        Used by
============================  ==========================================
:func:`build_llm` (normal)    Monitoring agent, Reporting agent
:func:`build_llm` (reasoning) Diagnosis agent (Amazon Nova extended thinking)
============================  ==========================================

Amazon Nova extended thinking is requested through
``additionalModelRequestFields={"reasoningConfig": {...}}`` because that is where
the Converse API expects model-specific parameters. Two documented traps:

* ``maxReasoningEffort="high"`` makes Bedrock **reject** ``temperature``,
  ``topP`` and ``maxTokens``, and needs a very large token budget. This lab
  therefore never sends ``temperature`` in that mode.
* small ``maxTokens`` values fail with ``"maxTokens is insufficient"``; the
  factory raises the budget to 16k/32k/50k for low/medium/high.
"""

from __future__ import annotations

from typing import Any

from lambda_src.config import Settings

#: CrewAI provider prefix for the native Bedrock integration.
BEDROCK_PREFIX = "bedrock/"

#: Reasoning is only applied when the provider supports it.
_REASONING_CAPABLE_PREFIXES = ("nova-2", "nova-pro", "nova-lite", "nova-premier", "nova-sonic")


def bedrock_model_string(model_id: str) -> str:
    """Return the CrewAI model string for a Bedrock model ID.

    Accepts every shape a user might configure:

    * ``amazon.nova-2-lite-v1:0``            -> ``bedrock/amazon.nova-2-lite-v1:0``
    * ``bedrock/amazon.nova-2-lite-v1:0``    -> unchanged
    * ``us.anthropic.claude-haiku-4-5-...``  -> ``bedrock/us.anthropic...``
    """
    candidate = str(model_id).strip()
    if not candidate:
        raise ValueError("BEDROCK_MODEL_ID is empty")
    if candidate.startswith(BEDROCK_PREFIX):
        return candidate
    return f"{BEDROCK_PREFIX}{candidate}"


def supports_reasoning(model_id: str) -> bool:
    """Whether the configured model family supports Amazon extended thinking."""
    bare = str(model_id).removeprefix(BEDROCK_PREFIX)
    # Drop a cross-region prefix such as `us.` or `global.`.
    bare = bare.split(".", 1)[1] if "." in bare and bare.split(".", 1)[0] in {"us", "global", "eu", "apac"} else bare
    return any(family in bare for family in _REASONING_CAPABLE_PREFIXES)


def use_extended_thinking(settings: Settings, *, requested: bool) -> bool:
    """Whether extended thinking is actually sent for this role.

    Three conditions must hold: the operator enabled it globally
    (``BEDROCK_REASONING_ENABLED``), the caller wants it (Diagnosis only), and the
    configured model family supports it. Without the first check, flipping the
    env var at deploy time would have no effect on the Diagnosis agent.
    """
    return bool(requested and settings.bedrock_reasoning_enabled and supports_reasoning(settings.bedrock_model_id))


def reasoning_request_fields(enabled: bool, effort: str) -> dict[str, Any] | None:
    """Build the ``additionalModelRequestFields`` payload, or ``None``.

    Returns ``None`` when reasoning is disabled or the model family does not
    support it, so Anthropic deployments keep working unchanged.
    """
    if not enabled:
        return None
    return {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": effort}}


def build_llm(settings: Settings, *, reasoning: bool = False) -> Any:
    """Create a ``crewai.LLM`` bound to Bedrock.

    Args:
        settings: Validated Lambda settings.
        reasoning: Enable Amazon extended thinking (Diagnosis agent only).

    Returns:
        A ``crewai.LLM`` instance. The crewai import happens here, not at module
        import time, so the unit tests never need the framework installed.
    """
    from crewai import LLM  # imported lazily: heavy and optional at test time

    model_id = settings.bedrock_model_id
    use_reasoning = use_extended_thinking(settings, requested=reasoning)

    kwargs: dict[str, Any] = {
        "model": bedrock_model_string(model_id),
        "region_name": settings.aws_region,
        "max_tokens": settings.bedrock_max_tokens_reasoning if use_reasoning else settings.bedrock_max_tokens,
    }

    if use_reasoning:
        # 'high' effort forbids temperature/topP/maxTokens in Bedrock.
        if settings.bedrock_reasoning_effort != "high":
            kwargs["temperature"] = settings.bedrock_temperature
        fields = reasoning_request_fields(True, settings.bedrock_reasoning_effort)
        if fields:
            kwargs["additional_model_request_fields"] = fields
    else:
        kwargs["temperature"] = settings.bedrock_temperature

    if settings.llm_timeout_seconds:
        kwargs["timeout"] = settings.llm_timeout_seconds

    return LLM(**kwargs)


def describe_llm(settings: Settings, *, reasoning: bool) -> dict[str, Any]:
    """Loggable description of the LLM a role will use (no secrets involved)."""
    use_reasoning = use_extended_thinking(settings, requested=reasoning)
    return {
        "model": bedrock_model_string(settings.bedrock_model_id),
        "region": settings.aws_region,
        "temperature": None if (use_reasoning and settings.bedrock_reasoning_effort == "high") else settings.bedrock_temperature,
        "max_tokens": settings.bedrock_max_tokens_reasoning if use_reasoning else settings.bedrock_max_tokens,
        "reasoning": bool(reasoning_request_fields(use_reasoning, settings.bedrock_reasoning_effort)),
        "reasoning_effort": settings.bedrock_reasoning_effort if use_reasoning else None,
    }
