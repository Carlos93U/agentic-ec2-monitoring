#!/usr/bin/env python3
"""CDK application entry point.

Run it through the CDK CLI:

    cdk synth
    cdk diff
    cdk deploy

Configuration is resolved once by :class:`infrastructure.lab_config.LabConfig`
from environment variables, ``.env`` and ``cdk.json`` context, and handed to
the stack. See the Configuration section in README.md.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

VENV_PYTHON = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
_REEXEC_GUARD = "AGENTIC_LAB_REEXEC"


def _ensure_dependencies_are_importable() -> None:
    """Re-exec into the local virtualenv when aws_cdk is missing.

    The CDK CLI runs ``cdk.json``'s ``app`` command with whatever ``python3`` is
    first on PATH, which is usually the system interpreter that does not have
    aws-cdk-lib installed. Rather than hardcoding ``.venv/bin/python`` in cdk.json
    (which breaks for anyone who bootstraps differently), the app fixes it up for
    itself. ``AGENTIC_LAB_REEXEC`` prevents an exec loop if the venv is broken.
    """
    try:
        import aws_cdk  # noqa: F401
    except ImportError:
        pass
    else:
        return

    if os.environ.get(_REEXEC_GUARD) or not VENV_PYTHON.exists():
        sys.exit(
            "aws-cdk-lib is not importable and no local virtualenv was found.\n"
            "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
        )

    os.environ[_REEXEC_GUARD] = "1"
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


_ensure_dependencies_are_importable()

import aws_cdk as cdk  # noqa: E402

from infrastructure.lab_config import ConfigError, LabConfig  # noqa: E402
from infrastructure.monitoring_stack import MonitoringStack  # noqa: E402

app = cdk.App()

try:
    config = LabConfig.load(context=app.node.try_get_context)
except ConfigError as error:
    sys.exit(f"Configuration error: {error}")

stack = MonitoringStack(
    app,
    config.stack_name,
    config=config,
    env=cdk.Environment(
        # CDK_DEFAULT_ACCOUNT/REGION are injected by the CDK CLI.
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=os.environ.get("CDK_DEFAULT_REGION") or config.aws_region,
    ),
)

app.synth()
