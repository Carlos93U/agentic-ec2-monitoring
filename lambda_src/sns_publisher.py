"""Publishes the notification to Amazon SNS email.

Two behaviours matter for a lab:

* **SNS silently drops notifications to unconfirmed subscriptions.** If the lab
  owner never clicked the confirmation email, ``Publish`` returns ``200`` and the
  message goes nowhere. The publisher therefore reports the topic ARN and warns in
  the logs about pending confirmations instead of pretending the mail was sent.
* **Publishing must never break the run.** A notification failure is logged and
  surfaced in the Lambda response, but the analysis result is still returned.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from lambda_src.logging_utils import get_logger, log_event
from lambda_src.report import Notification

LOGGER = get_logger("sns")


class PublishError(RuntimeError):
    """Raised when the notification could not be published."""


class SnsPublisher:
    """Thin wrapper around ``sns.publish``."""

    def __init__(self, topic_arn: str, region: str) -> None:
        if not topic_arn:
            raise PublishError("SNS_TOPIC_ARN is empty; the stack did not provide a topic")
        self._topic_arn = topic_arn
        self._region = region

    @property
    def topic_arn(self) -> str:
        return self._topic_arn

    def publish(self, notification: Notification) -> dict[str, Any]:
        """Send the notification as an SNS email message.

        Args:
            notification: Subject and bodies produced by
                :func:`lambda_src.report.build_notification`.

        Returns:
            The SNS ``MessageId`` plus topic ARN, for the Lambda response.

        Raises:
            PublishError: On any boto3 failure.
        """
        client = boto3.client("sns", region_name=self._region)
        try:
            response = client.publish(
                TopicArn=self._topic_arn,
                Subject=notification.subject[:100],  # SNS hard limit is 100 characters
                Message=notification.text_body,
                MessageAttributes={
                    "severity": {"DataType": "String", "StringValue": notification.severity.value},
                    "degraded": {"DataType": "String", "StringValue": str(notification.degraded).lower()},
                },
            )
        except (BotoCoreError, ClientError) as exc:
            log_event(
                LOGGER,
                logging.ERROR,
                "sns_publish_failed",
                topic_arn=self._topic_arn,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise PublishError(f"Could not publish to SNS: {type(exc).__name__}: {exc}") from exc

        log_event(
            LOGGER,
            logging.INFO,
            "sns_published",
            topic_arn=self._topic_arn,
            message_id=response.get("MessageId"),
            severity=notification.severity.value,
            degraded=notification.degraded,
        )
        return {
            "message_id": response.get("MessageId"),
            "topic_arn": self._topic_arn,
        }

    def subscription_guidance(self) -> str:
        """Human hint printed when the mailbox is probably not confirmed yet."""
        return (
            "If no email arrives, open the SNS topic in "
            f"{self._region} and confirm the pending "
            f"subscription for {self._topic_arn}. SNS accepts the publish call "
            "but does not deliver to unconfirmed subscriptions."
        )
