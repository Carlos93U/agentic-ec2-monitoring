"""The single CloudFormation stack behind the lab.

Composition (each piece lives in its own construct so the stack file stays
readable):

* :class:`LabNetwork`             - VPC + public subnet + security group (no ingress)
* :class:`MonitoredInstances`     - 2x t3.micro, Amazon Linux 2023, IMDSv2, SSM operator access
* :class:`Notifications`          - SNS topic + e-mail subscription
* :class:`MonitoringLambda`       - container-image Lambda + least-privilege IAM role
* :class:`MonitoringScheduler`    - EventBridge Scheduler, rate(5 minutes)
"""

from __future__ import annotations

import aws_cdk as cdk
from constructs import Construct

from infrastructure.constructs.ec2_construct import MonitoredInstances
from infrastructure.constructs.lambda_construct import MonitoringLambda
from infrastructure.constructs.network_construct import LabNetwork
from infrastructure.constructs.scheduler_construct import MonitoringScheduler
from infrastructure.constructs.sns_construct import Notifications
from infrastructure.lab_config import LAB_TAG_KEY, LAB_TAG_VALUE, LabConfig


class MonitoringStack(cdk.Stack):
    """Agentic EC2 monitoring lab: EventBridge -> Lambda -> CrewAI -> Bedrock -> SNS."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: LabConfig,
        env: cdk.Environment | None = None,
    ) -> None:
        super().__init__(
            scope,
            construct_id,
            env=env,
            description=(
                "Educational lab: a CrewAI crew (Monitoring / Diagnosis / Reporting) reads EC2 state "
                "and CloudWatch metrics through a read-only Lambda function every 5 minutes and "
                "publishes a report to SNS."
            ),
        )
        self.config = config

        # Tag everything taggable so `cdk destroy` / cleanup scripts can find it.
        cdk.Tags.of(self).add(LAB_TAG_KEY, LAB_TAG_VALUE)
        cdk.Tags.of(self).add("Project", "agentic-ai-lab")

        network = LabNetwork(self, "Network")
        instances = MonitoredInstances(
            self,
            "Instances",
            config=config,
            vpc=network.vpc,
            subnets=network.public_subnets,
            security_group=network.security_group,
        )
        notifications = Notifications(self, "Notifications", config=config)
        handler = MonitoringLambda(
            self,
            "Handler",
            config=config,
            topic_arn=notifications.topic.topic_arn,
            instance_ids=instances.instance_ids,
            instance_names=instances.names,
        )
        MonitoringScheduler(self, "Scheduler", config=config, function=handler.function)

        # ------------------------------------------------------------------ #
        # Deployment summary
        # ------------------------------------------------------------------ #
        cdk.CfnOutput(self, "Region", value=self.region, description="Region hosting the lab")
        cdk.CfnOutput(self, "BedrockModelId", value=config.bedrock_model_id, description="Model the crew will call")
        cdk.CfnOutput(
            self,
            "Reasoning",
            value=(
                f"enabled={str(config.bedrock_reasoning_enabled).lower()} effort={config.bedrock_reasoning_effort}"
                if config.bedrock_reasoning_enabled
                else "disabled"
            ),
            description="Extended thinking configuration for the Diagnosis agent",
        )
        cdk.CfnOutput(
            self,
            "MonitoringWindow",
            value=f"{config.monitoring_window_minutes} minutes",
            description="Analysis window of every run",
        )
        cdk.CfnOutput(
            self,
            "AlwaysNotify",
            value=str(config.always_notify).lower(),
            description="true = e-mail on every run; false = only on WARNING/CRITICAL",
        )
        cdk.CfnOutput(self, "SecurityPosture", value="read-only EC2 (no Start/Stop/Reboot/Terminate anywhere)")

        cdk.CfnOutput(
            self,
            "NextStep",
            value=(
                "1) Confirm the SNS e-mail subscription (console: SNS > Subscriptions > Pending). "
                "2) Confirm Bedrock model access for " + config.bedrock_model_id + ". "
                "3) Wait one schedule period, or trigger the function manually: "
                "aws lambda invoke --function-name <HandlerFunctionName> --cli-binary-format "
                "raw-in-base64-out --payload '{\"source\":\"manual\"}' /tmp/monitoring-result.json"
            ),
            description="Manual steps CloudFormation cannot do for you",
        )
