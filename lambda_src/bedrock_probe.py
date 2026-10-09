"""Preflight check for Bedrock access, runnable locally and inside the image.

This is the single most common first failure of the lab: the stack deploys
perfectly, Lambda runs on schedule, and every invocation fails with
``AccessDeniedException`` because the account never enabled the model. This module
answers that question before it costs a debugging session, and it can be invoked
directly:

.. code-block:: bash

    # Locally, with the caller's credentials
    python -m lambda_src.bedrock_probe

    # With the exact model/region the Lambda function will use
    BEDROCK_MODEL_ID=us.amazon.nova-pro-v1:0 python -m lambda_src.bedrock_probe

Three distinct failures are reported separately, because they have three different
fixes:

===================================  ==========================================
Failure                              Fix
===================================  ==========================================
``MODEL_ACCESS``                     Enable the model in the Bedrock console
``MISSING_IAM``                      Add ``bedrock:InvokeModel`` to the principal
``MISSING_REGION``                   Use a region where the model is available
===================================  ==========================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_MODEL = "us.amazon.nova-2-lite-v1:0"
DEFAULT_REGION = "us-east-1"


@dataclass
class ProbeResult:
    """Outcome of a single probe."""

    ok: bool
    model_id: str
    region: str
    identity: str = ""
    failure: str | None = None
    detail: str = ""
    available: bool | None = None
    models_seen: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "model_id": self.model_id,
            "region": self.region,
            "identity": self.identity,
            "failure": self.failure,
            "detail": self.detail,
            "model_access_in_account": self.available,
            "models_listed": self.models_seen,
            "notes": self.notes,
        }


def call_identity(identity_client: Any) -> str:
    """Best-effort caller description for the report."""
    try:
        return identity_client.get_caller_identity().get("Arn", "unknown")
    except (BotoCoreError, ClientError):
        return "unknown"


def list_model_access(bedrock_client: Any, model_id: str) -> tuple[int, bool, bool]:
    """Look the configured model up in the catalogue.

    ``ListFoundationModels`` only ever returns **foundation models**
    (``foundation-model/...``); a cross-region inference profile such as
    ``us.amazon.nova-2-lite-v1:0`` is *not* in that list, even when it works
    perfectly. It also omits ``accessInferenceStatus`` for most entries, so
    filtering on that field reports zero access for a perfectly usable account.

    Returns:
        ``(models_listed, catalog_contains_model, catalog_check_reliable)``.
        The last flag is ``False`` when the model could not be found *and* it is
        an inference profile, which is expected rather than a problem.
    """
    summaries = bedrock_client.list_foundation_models().get("modelSummaries", [])
    identifiers = {summary.get("modelId") for summary in summaries}

    # A `us.`/`eu.`/`global.`/`apac.` id is an inference profile; strip the
    # geographic prefix to find the foundation model it routes to.
    bare = model_id.split(".", 1)[1] if "." in model_id else model_id
    found = model_id in identifiers or bare in identifiers
    is_inference_profile = model_id != bare
    return len(summaries), found, found or not is_inference_profile


def probe(
    model_id: str = DEFAULT_MODEL,
    region: str = DEFAULT_REGION,
    bedrock_client: Any | None = None,
    identity_client: Any | None = None,
) -> ProbeResult:
    """Send a minimal Converse call and classify the result.

    A one-word prompt is used on purpose: the point is authorization, not answer
    quality, and the cheapest possible call keeps the probe nearly free.
    """
    result = ProbeResult(ok=False, model_id=model_id, region=region)
    bedrock = bedrock_client or boto3.client("bedrock", region_name=region)
    runtime = boto3.client("bedrock-runtime", region_name=region)
    result.identity = call_identity(identity_client or boto3.client("sts", region_name=region))

    try:
        result.models_seen, in_catalogue, catalogue_reliable = list_model_access(bedrock, model_id)
    except (BotoCoreError, ClientError) as exc:
        result.notes.append(f"Could not list foundation models: {type(exc).__name__}: {exc}")
        in_catalogue, catalogue_reliable = False, False
    # Only the Converse call below is authoritative; the catalogue is a hint.
    result.available = None if in_catalogue else (False if catalogue_reliable else None)
    if in_catalogue:
        result.notes.append("Model found in the Bedrock catalogue.")
    elif not catalogue_reliable:
        result.notes.append(
            "Model is a cross-region inference profile, which ListFoundationModels does not "
            "return. The Converse call below is the real test."
        )

    try:
        runtime.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "ping"}]}],
            inferenceConfig={"maxTokens": 16, "temperature": 0},
        )
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        result.detail = exc.response.get("Error", {}).get("Message", str(exc))
        message = result.detail.lower()
        if code in {"AccessDeniedException", "UnauthorizedException"}:
            # Bedrock returns AccessDenied for two very different problems, and
            # they have different fixes, so read the message instead of guessing.
            if "access to the model" in message or "model id" in message:
                result.failure = "MODEL_ACCESS"
                result.notes.append(
                    "The account has no inference access to this model. Bedrock console > "
                    "Model access > enable it."
                )
            else:
                result.failure = "MISSING_IAM"
                result.notes.append(
                    "The principal lacks bedrock:InvokeModel for this model. The Lambda role gets "
                    "this from the CDK policy; a human caller may not."
                )
        elif code == "ValidationException":
            if "model identifier is invalid" in message or "inference profile" in message:
                result.failure = "MISSING_INFERENCE_PROFILE"
            elif "not supported" in message:
                result.failure = "MISSING_REGION"
            else:
                result.failure = code
        else:
            result.failure = code
        return result
    except BotoCoreError as exc:
        result.failure = type(exc).__name__
        result.detail = str(exc)
        return result

    result.ok = True
    return result


def crew_kickoff(model_id: str, region: str) -> ProbeResult:
    """Drive a real CrewAI crew against Bedrock, the way the Lambda does.

    The boto3 path in :func:`probe` can pass while the crew path still fails:
    CrewAI resolves its own Bedrock client, its own region and its own model
    kwargs, and any of those can disagree with what we set by hand. This runs a
    one-agent, one-task crew with the same factory the Lambda uses, so a failure
    here is a real prediction of a broken lab rather than a probe artefact.
    """
    result = ProbeResult(ok=False, model_id=model_id, region=region)
    result.identity = call_identity(boto3.client("sts", region_name=region))

    try:
        from crewai import Agent, Crew, Task
    except ImportError as error:
        result.failure = "CREWAI_MISSING"
        result.detail = f"crewai is not importable in this interpreter: {error}"
        return result

    # build_llm() takes validated Settings, and Settings insists on SNS and EC2
    # variables that have nothing to do with reaching the model. Inject inert
    # placeholders so this probe exercises the same factory the Lambda does
    # without pretending it has a deployed stack behind it.
    probe_env = {
        **os.environ,
        "BEDROCK_MODEL_ID": model_id,
        "AWS_REGION": region,
        "SNS_TOPIC_ARN": os.environ.get("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:000000000000:probe"),
        "MONITORED_INSTANCE_IDS": os.environ.get("MONITORED_INSTANCE_IDS", "i-00000000000000000"),
        "MONITORED_INSTANCE_NAMES": os.environ.get("MONITORED_INSTANCE_NAMES", "probe"),
    }

    try:
        from lambda_src.config import Settings
        from lambda_src.llm_factory import build_llm

        settings = Settings.from_env(probe_env)
        # reasoning=True exercises the extended-thinking branch as well, since
        # that is where model/effort mismatches surface.
        llm = build_llm(settings, reasoning=True)
    except Exception as error:  # noqa: BLE001 - surfaced verbatim, this is a diagnostic tool
        result.failure = "CREWAI_LLM_INIT"
        result.detail = f"{type(error).__name__}: {error}"
        return result

    agent = Agent(
        role="Reliability analyst",
        goal="Answer with a single short sentence proving the LLM answered.",
        backstory="You verify that a Bedrock-backed crew can complete a single task.",
        llm=llm,
        verbose=False,
    )
    task = Task(
        description="Reply with exactly the word: reachable",
        expected_output="The single word: reachable",
        agent=agent,
    )

    try:
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        output = str(crew.kickoff())
    except Exception as error:  # noqa: BLE001 - the message is the deliverable
        result.failure = "CREWAI_KICKOFF"
        result.detail = f"{type(error).__name__}: {error}"[:900]
        result.notes.append(
            "This is the integration the Lambda depends on; a failure here is a real "
            "blocker, not a probe artefact."
        )
        return result

    result.ok = True
    result.available = True
    result.detail = f"crew reply: {output.strip()[:200]}"
    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Exit code 0 means the model is usable."""
    parser = argparse.ArgumentParser(description="Check Bedrock access before deploying the lab")
    # Same variables the Lambda function reads, so the probe and the function can
    # never disagree about which model or region they target.
    parser.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL),
        help="Bedrock model ID to probe (default: $BEDROCK_MODEL_ID)",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION") or DEFAULT_REGION,
        help="AWS region to probe (default: $AWS_REGION)",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--local-only",
        action="store_true",
        help="only run the boto3 Converse probe (no CrewAI)",
    )
    modes.add_argument(
        "--crew-only",
        action="store_true",
        help="only run the CrewAI kickoff (needs crewai[bedrock] installed)",
    )
    args = parser.parse_args(argv)

    if args.crew_only:
        result = crew_kickoff(model_id=args.model_id, region=args.region)
        label = "CrewAI"
    else:
        result = probe(model_id=args.model_id, region=args.region)
        label = "boto3 Converse"
        if result.ok and not args.local_only:
            # Both paths must pass: proving only one is how a green check hides a
            # broken crew. The second failure is reported if it happens.
            crew_result = crew_kickoff(model_id=args.model_id, region=args.region)
            if not crew_result.ok:
                result = crew_result
                label = "CrewAI"

    print(json.dumps(result.to_dict(), indent=2, default=str))

    if result.ok:
        print(f"\nOK ({label}): the crew can run.")
        return 0

    print(f"\nFAILED ({result.failure}): {result.detail}", file=sys.stderr)
    fixes = {
        "MODEL_ACCESS": (
            "Open the Bedrock console in "
            f"{args.region}, choose Model access, and enable {args.model_id} for this account."
        ),
        "MISSING_IAM": (
            "The calling principal is not authorized to invoke this model. The Lambda role is "
            "granted this by CDK; a human caller may need bedrock:InvokeModel."
        ),
        "MISSING_INFERENCE_PROFILE": (
            f"{args.model_id} looks like an inference profile that does not exist in "
            f"{args.region}. Create it in the Bedrock console, or use the bare foundation "
            "model id (for example amazon.nova-2-lite-v1:0)."
        ),
        "MISSING_REGION": f"{args.model_id} is not offered in {args.region}. Pick another region.",
        "CREWAI_MISSING": (
            "Install the Lambda dependency set locally: "
            ".venv/bin/pip install -r lambda_src/requirements.txt"
        ),
        "CREWAI_LLM_INIT": (
            "The LLM factory refused this model. Check BEDROCK_REASONING_EFFORT and the "
            "model id in README: Configuration."
        ),
        "CREWAI_KICKOFF": (
            "CrewAI reached the LLM but the call failed. Check the region, the model id and "
            "the reasoning settings; crewai[bedrock] resolves its own client."
        ),
    }
    print(fixes.get(result.failure or "", "See README: Configuration."), file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
