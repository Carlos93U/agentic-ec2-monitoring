"""Typed settings built from the environment variables set by the CDK stack.

Every value the Lambda function uses comes from here. Nothing is hardcoded in the
business logic, so changing the behaviour of the system is a ``cdk deploy``, not
a code change.

Variables (see ``.env.example`` and README: Configuration):

======================================  =========================================
Variable                                Meaning
======================================  =========================================
``BEDROCK_MODEL_ID``                    Model the crew calls (required)
``BEDROCK_REASONING_ENABLED``           Extended thinking on the Diagnosis agent
``BEDROCK_REASONING_EFFORT``            ``low`` | ``medium`` | ``high``
``ALWAYS_NOTIFY``                       Notify on every run vs. only on problems
``MONITORING_WINDOW_MINUTES``           Analysis window
``CPU_WARNING_THRESHOLD``               CPU % that triggers WARNING
``CPU_CRITICAL_THRESHOLD``              CPU % that triggers CRITICAL
``MONITORED_INSTANCE_IDS``              Comma-separated EC2 IDs to monitor
``MONITORED_INSTANCE_NAMES``            Display names for the report
``SNS_TOPIC_ARN``                       Where the report is published (required)
``AWS_REGION``                          Provided by the Lambda runtime
``LOG_LEVEL``                           Logging verbosity
======================================  =========================================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from lambda_src.models import Thresholds

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}
_REASONING_EFFORTS = ("low", "medium", "high")

#: Token budget for a normal Bedrock call. Small: the agents answer with short JSON.
DEFAULT_MAX_TOKENS = 2048
#: Token budget when extended thinking is on. Bedrock rejects small budgets with
#: "maxTokens is insufficient" (recommended minimums: low=15k, medium=30k, high=50k).
REASONING_MAX_TOKENS = {"low": 16000, "medium": 32000, "high": 51200}


class ConfigurationError(RuntimeError):
    """Raised when the Lambda environment is missing or contradicts configuration."""


def _as_bool(value: str | bool, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False
    raise ConfigurationError(f"{key} must be a boolean ('true'/'false'), got {value!r}")


def _as_int(value: str, *, key: str, minimum: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be an integer, got {value!r}") from exc
    if parsed < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}, got {parsed}")
    return parsed


def _as_float(value: str, *, key: str) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be a number, got {value!r}") from exc
    if not 0 <= parsed <= 100:
        raise ConfigurationError(f"{key} must be a percentage between 0 and 100, got {parsed}")
    return parsed


def _as_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Validated view of the Lambda environment."""

    aws_region: str
    bedrock_model_id: str
    bedrock_reasoning_enabled: bool
    bedrock_reasoning_effort: str
    always_notify: bool
    monitoring_window_minutes: int
    cpu_warning_threshold: float
    cpu_critical_threshold: float
    monitored_instance_ids: tuple[str, ...]
    monitored_instance_names: tuple[str, ...]
    sns_topic_arn: str
    log_level: str = "INFO"
    bedrock_temperature: float = 0.0
    llm_timeout_seconds: float | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` (defaults to :data:`os.environ`).

        Raises:
            ConfigurationError: when a required variable is missing or invalid.
        """
        source: Mapping[str, str] = os.environ if env is None else env

        def require(key: str) -> str:
            value = source.get(key, "").strip()
            if not value:
                raise ConfigurationError(
                    f"Required environment variable {key} is missing. "
                    "Deploy the stack with CDK or set the variable manually."
                )
            return value

        model_id = require("BEDROCK_MODEL_ID")
        topic_arn = require("SNS_TOPIC_ARN")
        instance_ids = _as_csv(source.get("MONITORED_INSTANCE_IDS", ""))
        if not instance_ids:
            raise ConfigurationError(
                "MONITORED_INSTANCE_IDS is empty: the function refuses to discover instances "
                "on its own so it can never look at resources outside the lab."
            )

        effort = (source.get("BEDROCK_REASONING_EFFORT", "low") or "low").strip().lower()
        if effort not in _REASONING_EFFORTS:
            raise ConfigurationError(
                f"BEDROCK_REASONING_EFFORT must be one of {', '.join(_REASONING_EFFORTS)}, got {effort!r}"
            )

        warning = _as_float(source.get("CPU_WARNING_THRESHOLD", "70"), key="CPU_WARNING_THRESHOLD")
        critical = _as_float(source.get("CPU_CRITICAL_THRESHOLD", "90"), key="CPU_CRITICAL_THRESHOLD")
        if critical <= warning:
            raise ConfigurationError(
                f"CPU_CRITICAL_THRESHOLD ({critical}) must be greater than CPU_WARNING_THRESHOLD ({warning})"
            )

        region = (source.get("AWS_REGION") or source.get("AWS_DEFAULT_REGION") or "us-east-1").strip()

        timeout_raw = source.get("BEDROCK_TIMEOUT_SECONDS", "").strip()
        llm_timeout = float(timeout_raw) if timeout_raw else None

        return cls(
            aws_region=region,
            bedrock_model_id=model_id,
            bedrock_reasoning_enabled=_as_bool(
                source.get("BEDROCK_REASONING_ENABLED", "false"), key="BEDROCK_REASONING_ENABLED"
            ),
            bedrock_reasoning_effort=effort,
            always_notify=_as_bool(source.get("ALWAYS_NOTIFY", "false"), key="ALWAYS_NOTIFY"),
            monitoring_window_minutes=_as_int(
                source.get("MONITORING_WINDOW_MINUTES", "10"), key="MONITORING_WINDOW_MINUTES", minimum=1
            ),
            cpu_warning_threshold=warning,
            cpu_critical_threshold=critical,
            monitored_instance_ids=instance_ids,
            monitored_instance_names=_as_csv(source.get("MONITORED_INSTANCE_NAMES", "")),
            sns_topic_arn=topic_arn,
            log_level=(source.get("LOG_LEVEL", "INFO") or "INFO").strip().upper(),
            llm_timeout_seconds=llm_timeout,
        )

    # -- derived values ----------------------------------------------------- #
    @property
    def thresholds(self) -> Thresholds:
        """CPU thresholds in the shape the evaluator expects."""
        return Thresholds(
            cpu_warning_percent=self.cpu_warning_threshold,
            cpu_critical_percent=self.cpu_critical_threshold,
        )

    @property
    def bedrock_max_tokens(self) -> int:
        """Token budget for a normal Bedrock call."""
        return DEFAULT_MAX_TOKENS

    @property
    def bedrock_max_tokens_reasoning(self) -> int:
        """Token budget for the reasoning-enabled Diagnosis agent."""
        return REASONING_MAX_TOKENS[self.bedrock_reasoning_effort]

    def instance_name(self, instance_id: str) -> str:
        """Map an instance ID to its ``Name`` tag when known.

        Two sources, in order of trust: the ``MONITORED_INSTANCE_NAMES`` entries
        written by the stack, and the tag discovered by the collector at runtime.
        ``MONITORED_INSTANCE_NAMES`` is positional (it mirrors the ID list), never
        a lookup by name, so an instance that appears twice is handled the same
        way as any other.
        """
        try:
            index = self.monitored_instance_ids.index(instance_id)
        except ValueError:
            return instance_id
        if index < len(self.monitored_instance_names):
            return self.monitored_instance_names[index]
        return instance_id

    def describe(self) -> dict[str, object]:
        """Configuration safe to log (contains no secrets by design)."""
        return {
            "region": self.aws_region,
            "bedrock_model_id": self.bedrock_model_id,
            "reasoning_enabled": self.bedrock_reasoning_enabled,
            "reasoning_effort": self.bedrock_reasoning_effort,
            "always_notify": self.always_notify,
            "monitoring_window_minutes": self.monitoring_window_minutes,
            "cpu_warning_threshold": self.cpu_warning_threshold,
            "cpu_critical_threshold": self.cpu_critical_threshold,
            "monitored_instance_ids": list(self.monitored_instance_ids),
            "sns_topic_arn": self.sns_topic_arn,
        }
