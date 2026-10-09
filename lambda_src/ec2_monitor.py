"""Read-only EC2 state collection.

Only ``ec2:DescribeInstances`` is called. There is no code path in this module
(or anywhere else in the Lambda) that can start, stop, reboot or terminate an
instance -- and the execution role does not even have those permissions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from lambda_src.logging_utils import get_logger, log_event
from lambda_src.models import STATE_NOT_FOUND, InstanceSnapshot

LOGGER = get_logger("ec2_monitor")

#: EC2 error codes that mean "this ID cannot produce a state", not "the call failed".
_TERMINAL_ERROR_CODES = frozenset(
    {
        "InvalidInstanceID.NotFound",
        "InvalidInstanceID.Malformed",
        "InvalidInstanceID.Unavailable",
    }
)


class EC2ReadError(RuntimeError):
    """Raised when the EC2 API cannot be queried at all."""


class EC2ClientProtocol(Protocol):
    """Structural type for the bit of the EC2 client this module uses."""

    def describe_instances(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class EC2ReadResult:
    """Snapshots plus non-fatal errors, so one bad ID cannot lose the whole run."""

    snapshots: tuple[InstanceSnapshot, ...] = ()
    errors: tuple[str, ...] = field(default_factory=tuple)


class EC2Monitor:
    """Fetches the current state of the instances the lab is allowed to monitor."""

    def __init__(self, client: EC2ClientProtocol | None = None, *, region: str | None = None) -> None:
        self._client = client
        self._region = region

    @property
    def client(self) -> EC2ClientProtocol:
        """Lazily create the boto3 client (tests inject their own fake)."""
        if self._client is None:
            try:
                self._client = boto3.client("ec2", region_name=self._region)
            except (BotoCoreError, ClientError) as exc:  # pragma: no cover - credential issues
                raise EC2ReadError(f"Could not create an EC2 client: {exc}") from exc
        return self._client

    def describe(self, instance_ids: tuple[str, ...] | list[str]) -> EC2ReadResult:
        """Describe each instance individually.

        Describing one ID at a time is intentional: ``DescribeInstances`` fails the
        *whole* request when a single ID is invalid, and we would rather lose one
        instance than the entire run.

        Args:
            instance_ids: Instance IDs configured in ``MONITORED_INSTANCE_IDS``.

        Returns:
            Snapshots (one per instance, using ``not-found`` when AWS does not
            return it) and a list of human-readable, non-fatal errors.
        """
        snapshots: list[InstanceSnapshot] = []
        errors: list[str] = []

        for instance_id in instance_ids:
            try:
                response = self.client.describe_instances(InstanceIds=[instance_id])
            except ClientError as exc:
                code = str(exc.response.get("Error", {}).get("Code", ""))
                message = str(exc.response.get("Error", {}).get("Message", exc))
                if code in _TERMINAL_ERROR_CODES:
                    log_event(
                        LOGGER,
                        logging.WARNING,
                        "instance_not_describable",
                        instance_id=instance_id,
                        error_code=code,
                        error_message=message,
                    )
                    errors.append(f"{instance_id}: EC2 returned {code} ({message})")
                    snapshots.append(
                        InstanceSnapshot(instance_id=instance_id, name=instance_id, state=STATE_NOT_FOUND)
                    )
                    continue
                raise EC2ReadError(f"describe_instances failed for {instance_id} with {code}: {message}") from exc
            except BotoCoreError as exc:
                raise EC2ReadError(f"describe_instances failed for {instance_id}: {exc}") from exc

            snapshots.extend(self._parse_response(response, instance_id))
            errors.extend(self._log_response(response, instance_id))

        log_event(
            LOGGER,
            logging.INFO,
            "ec2_instances_described",
            requested=len(instance_ids),
            returned=len(snapshots),
            states={snapshot.instance_id: snapshot.state for snapshot in snapshots},
        )
        return EC2ReadResult(snapshots=tuple(snapshots), errors=tuple(errors))

    # -- parsing ----------------------------------------------------------- #
    def _parse_response(self, response: dict[str, Any], instance_id: str) -> list[InstanceSnapshot]:
        instances: list[dict[str, Any]] = []
        for reservation in response.get("Reservations", []) or []:
            instances.extend(reservation.get("Instances", []) or [])

        if not instances:
            return [InstanceSnapshot(instance_id=instance_id, name=instance_id, state=STATE_NOT_FOUND)]

        return [self._to_snapshot(instance) for instance in instances]

    def _log_response(self, response: dict[str, Any], instance_id: str) -> list[str]:
        errors: list[str] = []
        for reservation in response.get("Reservations", []) or []:
            for instance in reservation.get("Instances", []) or []:
                if not instance.get("State"):
                    errors.append(f"{instance_id}: EC2 returned an instance without state information")
        return errors

    @staticmethod
    def _to_snapshot(instance: dict[str, Any]) -> InstanceSnapshot:
        state = instance.get("State", {}) or {}
        name = instance_id = str(instance.get("InstanceId", ""))
        for tag in instance.get("Tags", []) or []:
            if tag.get("Key") == "Name" and tag.get("Value"):
                name = str(tag["Value"])
                break

        return InstanceSnapshot(
            instance_id=instance_id,
            name=name,
            state=str(state.get("Name", STATE_NOT_FOUND)),
            state_transition_reason=(state.get("TransitionReason") or None),
            instance_type=instance.get("InstanceType"),
            availability_zone=(instance.get("Placement") or {}).get("AvailabilityZone"),
            private_ip=instance.get("PrivateIpAddress"),
            public_ip=instance.get("PublicIpAddress"),
            launch_time=_parse_timestamp(instance.get("LaunchTime")),
        )


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse the ISO strings EC2 returns, tolerating None."""
    if not value:
        return None
    if hasattr(value, "tzinfo"):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
