"""Centralised, validated configuration for the lab.

Every value the stack needs is resolved once, here, from (in order):

1. real environment variables (highest priority),
2. a local ``.env`` file (``python-dotenv``),
3. ``cdk.json`` context,
4. built-in defaults.

The same values are forwarded to the Lambda function as plain environment
variables, so a single ``cdk deploy`` is enough to change the behaviour of the
system.  Nothing in this module reaches out to AWS at import time, which keeps
``cdk synth`` fast and deterministic.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tag applied to every resource the lab can create. Used to identify and,
# if needed, clean up anything the stack leaves behind after `cdk destroy`.
LAB_TAG_KEY = "LabName"
LAB_TAG_VALUE = "agentic-ec2-monitoring"

#: Bedrock cross-region inference profiles always start with one of these.
INFERENCE_PROFILE_PREFIXES = ("us.", "global.", "eu.", "apac.", "us-gov.")

#: Reasoning efforts accepted by Amazon Nova extended thinking.
REASONING_EFFORTS = ("low", "medium", "high")

#: CloudWatch Logs retention values that ``aws_cdk.aws_logs.RetentionDays`` can
#: express, expressed in days so they can be configured as a plain number.
LOG_RETENTION_DAYS = (1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365)

#: Maps those day counts to the CDK enum member names.
LOG_RETENTION_MEMBER_NAMES = {
    1: "ONE_DAY",
    3: "THREE_DAYS",
    5: "FIVE_DAYS",
    7: "ONE_WEEK",
    14: "TWO_WEEKS",
    30: "ONE_MONTH",
    60: "TWO_MONTHS",
    90: "THREE_MONTHS",
    120: "FOUR_MONTHS",
    150: "FIVE_MONTHS",
    180: "SIX_MONTHS",
    365: "ONE_YEAR",
}


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid."""


# --------------------------------------------------------------------------- #
# Primitive parsers
# --------------------------------------------------------------------------- #
_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


def parse_bool(raw: Any, *, key: str) -> bool:
    """Parse a boolean coming from an env var, a context value or a default."""
    if isinstance(raw, bool):
        return raw
    value = str(raw).strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise ConfigError(f"{key} must be a boolean like 'true'/'false', got {raw!r}")


def parse_int(raw: Any, *, key: str, minimum: int = 1, maximum: int | None = None) -> int:
    """Parse and range-check an integer setting."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}")
    return value


def parse_float(raw: Any, *, key: str, minimum: float | None = None, maximum: float | None = None) -> float:
    """Parse and range-check a float setting."""
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} must be <= {maximum}, got {value}")
    return value


def parse_choice(raw: Any, *, key: str, allowed: tuple[str, ...]) -> str:
    """Parse a setting that must be one of a fixed set of values."""
    value = str(raw).strip().lower()
    if value not in allowed:
        raise ConfigError(f"{key} must be one of {', '.join(allowed)}, got {raw!r}")
    return value


def parse_email(raw: Any, *, key: str) -> str:
    """Very small e-mail sanity check: SNS rejects anything else at deploy time."""
    value = str(raw).strip()
    local, sep, domain = value.partition("@")
    if not sep or not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        raise ConfigError(f"{key} does not look like an e-mail address: {raw!r}")
    return value


def parse_schedule_expression(raw: Any, *, key: str) -> str:
    """Validate an EventBridge Scheduler ``rate()`` or ``cron()`` expression."""
    value = str(raw).strip()
    if not (value.startswith("rate(") or value.startswith("cron(")):
        raise ConfigError(f"{key} must start with 'rate(' or 'cron(', got {raw!r}")
    if not value.endswith(")"):
        raise ConfigError(f"{key} must end with ')', got {raw!r}")
    return value


# --------------------------------------------------------------------------- #
# Bedrock helpers
# --------------------------------------------------------------------------- #
def is_inference_profile(model_id: str) -> bool:
    """Return True for cross-region inference profile IDs (``us.anthropic...``)."""
    return model_id.startswith(INFERENCE_PROFILE_PREFIXES)


def bedrock_model_arns(model_id: str, region: str, *, account: str) -> list[str]:
    """Build the least-privilege ARNs a Lambda role needs to invoke ``model_id``.

    ``bedrock:InvokeModel`` *does* support resource-level permissions, so unlike
    the EC2/CloudWatch read APIs this policy can be scoped tightly.

    * ``amazon.nova-2-lite-v1:0``      -> ``foundation-model/amazon.nova-2-lite-v1:0``
    * ``us.amazon.nova-2-lite-v1:0``   -> ``inference-profile/us.amazon.nova-2-lite-v1:0``
      plus the underlying ``foundation-model/amazon.nova-2-lite-v1:0`` ARN, because
      AWS requires the profile *and* the model it routes to. The model ARN uses a
      ``*`` region because the profile may route to any region of its region set.
    * account-scoped profiles (``arn:aws:bedrock:<region>:<account>:inference-profile/...``)
      are supported as well, since a student may use one.

    The AWS IAM resource has *no* account segment (``::``) for foundation models,
    but the system-defined inference-profile ARN *does* carry the account ID, so
    it must be passed in explicitly.
    """
    if model_id.startswith("arn:"):
        return [model_id]

    if is_inference_profile(model_id):
        # System-defined profiles are scoped to the calling account at the IAM level.
        profile_arn = f"arn:aws:bedrock:{region}:{account}:inference-profile/{model_id}"
        # The `us.`/`global.`/... prefix is not part of the foundation-model ID.
        bare_id = model_id.split(".", 1)[1]
        # A cross-region profile routes the underlying model to *any* region in
        # its region set (the probe hit us-east-2 from a us-east-1 deployment),
        # and IAM authorizes InvokeModel against the routed region's model ARN.
        # A `*` region keeps the policy scoped to the exact model id.
        return [profile_arn, f"arn:aws:bedrock:*::foundation-model/{bare_id}"]

    return [f"arn:aws:bedrock:{region}::foundation-model/{model_id}"]


# --------------------------------------------------------------------------- #
# Lab configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabConfig:
    """Immutable, validated configuration for a single lab deployment."""

    stack_name: str = "agentic-ec2-monitoring"
    aws_region: str = "us-east-1"

    bedrock_model_id: str = "us.amazon.nova-2-lite-v1:0"
    bedrock_reasoning_enabled: bool = True
    bedrock_reasoning_effort: str = "low"

    always_notify: bool = False
    monitoring_window_minutes: int = 10
    cpu_warning_threshold: float = 70.0
    cpu_critical_threshold: float = 90.0

    notification_email: str = "you@example.com"
    schedule_expression: str = "rate(5 minutes)"

    ec2_instance_type: str = "t3.micro"
    ec2_instance_count: int = 2
    detailed_monitoring: bool = True

    lambda_timeout_seconds: int = 180
    lambda_memory_mb: int = 2048
    log_retention_days: int = 14

    #: Directory packaged as the Lambda container image.
    lambda_source_dir: Path = field(default=PROJECT_ROOT / "lambda_src")

    @property
    def log_group_name(self) -> str:
        """Deterministic log group name (lets IAM be scoped to it)."""
        return f"/aws/lambda/{self.stack_name}-monitoring"

    @property
    def instance_names(self) -> list[str]:
        """EC2 ``Name`` tags, e.g. ``agentic-monitoring-ec2-1``."""
        return [f"agentic-monitoring-ec2-{index}" for index in range(1, self.ec2_instance_count + 1)]

    @property
    def tags(self) -> dict[str, str]:
        """Tags applied to every lab resource."""
        return {
            LAB_TAG_KEY: LAB_TAG_VALUE,
            "Project": "agentic-ai-lab",
            "ManagedBy": "aws-cdk",
        }

    def validate(self) -> None:
        """Cross-field validation that individual parsers cannot express."""
        if self.cpu_critical_threshold <= self.cpu_warning_threshold:
            raise ConfigError(
                "CPU_CRITICAL_THRESHOLD must be greater than CPU_WARNING_THRESHOLD "
                f"(got {self.cpu_critical_threshold} <= {self.cpu_warning_threshold})"
            )
        if self.ec2_instance_count < 1:
            raise ConfigError("EC2_INSTANCE_COUNT must be >= 1")
        if self.lambda_timeout_seconds > 900:
            raise ConfigError("LAMBDA_TIMEOUT_SECONDS must be <= 900 (Lambda maximum)")
        if self.lambda_memory_mb < 128:
            raise ConfigError("LAMBDA_MEMORY_MB must be >= 128")
        if self.log_retention_days not in LOG_RETENTION_DAYS:
            raise ConfigError(
                f"LOG_RETENTION_DAYS must be one of {', '.join(str(day) for day in LOG_RETENTION_DAYS)}"
            )

    # -- loading ---------------------------------------------------------- #
    @classmethod
    def load(
        cls,
        context: Callable[[str], Any] | None = None,
        *,
        env: dict[str, str] | None = None,
    ) -> LabConfig:
        """Build the configuration from env vars, ``.env``, context and defaults.

        Args:
            context: Callable used to read ``cdk.json`` context (``app.node.try_get_context``).
            env: Environment mapping to read. Defaults to :data:`os.environ`.
        """
        load_dotenv_file()
        environ: dict[str, str] = dict(os.environ if env is None else env)
        resolve = context or (lambda _key: None)

        def get(env_key: str, context_key: str, default: Any) -> Any:
            if env_key in environ and str(environ[env_key]).strip() != "":
                return environ[env_key]
            from_context = resolve(context_key)
            if from_context not in (None, ""):
                return from_context
            return default

        config = cls(
            stack_name=str(get("STACK_NAME", "stackName", "agentic-ec2-monitoring")),
            aws_region=str(get("AWS_REGION", "awsRegion", "us-east-1")),
            bedrock_model_id=str(get("BEDROCK_MODEL_ID", "bedrockModelId", "us.amazon.nova-2-lite-v1:0")),
            bedrock_reasoning_enabled=parse_bool(
                get("BEDROCK_REASONING_ENABLED", "bedrockReasoningEnabled", True),
                key="BEDROCK_REASONING_ENABLED",
            ),
            bedrock_reasoning_effort=parse_choice(
                get("BEDROCK_REASONING_EFFORT", "bedrockReasoningEffort", "low"),
                key="BEDROCK_REASONING_EFFORT",
                allowed=REASONING_EFFORTS,
            ),
            always_notify=parse_bool(get("ALWAYS_NOTIFY", "alwaysNotify", False), key="ALWAYS_NOTIFY"),
            monitoring_window_minutes=parse_int(
                get("MONITORING_WINDOW_MINUTES", "monitoringWindowMinutes", 10),
                key="MONITORING_WINDOW_MINUTES",
                minimum=1,
                maximum=1440,
            ),
            cpu_warning_threshold=parse_float(
                get("CPU_WARNING_THRESHOLD", "cpuWarningThreshold", 70), key="CPU_WARNING_THRESHOLD", minimum=0, maximum=100
            ),
            cpu_critical_threshold=parse_float(
                get("CPU_CRITICAL_THRESHOLD", "cpuCriticalThreshold", 90), key="CPU_CRITICAL_THRESHOLD", minimum=0, maximum=100
            ),
            notification_email=parse_email(
                get("NOTIFICATION_EMAIL", "notificationEmail", "you@example.com"),
                key="NOTIFICATION_EMAIL",
            ),
            schedule_expression=parse_schedule_expression(
                get("SCHEDULE_EXPRESSION", "scheduleExpression", "rate(5 minutes)"),
                key="SCHEDULE_EXPRESSION",
            ),
            ec2_instance_type=str(get("EC2_INSTANCE_TYPE", "ec2InstanceType", "t3.micro")),
            ec2_instance_count=parse_int(
                get("EC2_INSTANCE_COUNT", "ec2InstanceCount", 2), key="EC2_INSTANCE_COUNT", minimum=1, maximum=20
            ),
            detailed_monitoring=parse_bool(
                get("DETAILED_MONITORING", "detailedMonitoring", True), key="DETAILED_MONITORING"
            ),
            lambda_timeout_seconds=parse_int(
                get("LAMBDA_TIMEOUT_SECONDS", "lambdaTimeoutSeconds", 180),
                key="LAMBDA_TIMEOUT_SECONDS",
                minimum=30,
                maximum=900,
            ),
            lambda_memory_mb=parse_int(
                get("LAMBDA_MEMORY_MB", "lambdaMemoryMb", 2048),
                key="LAMBDA_MEMORY_MB",
                minimum=128,
                maximum=10240,
            ),
            log_retention_days=parse_int(
                get("LOG_RETENTION_DAYS", "logRetentionDays", 14), key="LOG_RETENTION_DAYS", minimum=1
            ),
            lambda_source_dir=Path(str(get("LAMBDA_SOURCE_DIR", "lambdaSourceDir", str(PROJECT_ROOT / "lambda_src")))),
        )
        config.validate()
        return config


def load_dotenv_file() -> None:
    """Load ``.env`` into :data:`os.environ` without overriding real env vars."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:  # pragma: no cover - requirements always install it
        return
    load_dotenv(env_file, override=False)
