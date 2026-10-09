"""SNS topic plus the e-mail subscription used by the Reporting agent.

The e-mail subscription is created in a **pending** state on purpose: AWS always
sends a confirmation link and only CDK-free manual action can flip it to
``Confirmed``. That is documented in README: After deploying.
"""

from __future__ import annotations

from aws_cdk import CfnOutput
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions
from constructs import Construct

from infrastructure.lab_config import LabConfig


class Notifications(Construct):
    """Owns the SNS topic and its e-mail subscription."""

    def __init__(self, scope: Construct, construct_id: str, *, config: LabConfig) -> None:
        super().__init__(scope, construct_id)

        self.topic = sns.Topic(
            self,
            "AlertsTopic",
            display_name="EC2 Agentic Monitoring",
            # Reject any publish that does not go over TLS. The Lambda function
            # uses HTTPS, so this is free, and it documents the security intent.
            # Tags come from cdk.Tags.of(self) at the stack level.
            enforce_ssl=True,
        )

        self.email_subscription = self.topic.add_subscription(
            subscriptions.EmailSubscription(config.notification_email)
        )

        CfnOutput(self, "TopicArn", value=self.topic.topic_arn, description="SNS topic the Lambda publishes to")
        CfnOutput(
            self,
            "NotificationEmail",
            value=config.notification_email,
            description="E-mail that must be confirmed in the AWS console",
        )
