"""The Lambda function that runs the CrewAI crew, and its IAM role.

Packaging decision (see README: Project layout)
--------------------------------------
The function is deployed as a **container image**, not as a zip:

* Lambda caps a zip at 250 MB unzipped. ``crewai[bedrock]`` pulls in
  chromadb, lancedb, tokenizers, pdfplumber, opentelemetry... which is far above
  that limit.
* Layers cannot help: in Lambda, layers and container images are mutually
  exclusive, and a layer shares the very same 250 MB budget.
* A container image may be up to 10 GB, is built from a ``Dockerfile`` you can
  read, and ``cdk deploy`` builds and publishes it to ECR for you.

IAM decision (see README: Security model)
--------------------------------
Exactly four capabilities, nothing else:

======================  ==========================================  ==========================
Capability              Actions                                     Resource scope
======================  ==========================================  ==========================
Observe EC2             ec2:DescribeInstances,                      ``*`` (see below)
                        ec2:DescribeInstanceStatus
Read metrics            cloudwatch:GetMetricData,                   ``*`` (see below)
                        cloudwatch:GetMetricStatistics,             |
                        cloudwatch:ListMetrics                      |
Invoke Bedrock          bedrock:InvokeModel,                       exact model/profile ARNs
                        bedrock:InvokeModelWithResponseStream      built from BEDROCK_MODEL_ID
Publish alerts          sns:Publish                                 exact topic ARN
Write own logs          logs:CreateLogGroup/CreateLogStream/       exact log group ARN
                        PutLogEvents
======================  ==========================================  ==========================

``ec2:Describe*`` and ``cloudwatch:GetMetric*`` are the only entries with
``Resource: "*"``, and that is *not* a ``"*"`` policy: AWS does not support
resource-level permissions for those read-only APIs at all. No statement uses
``Action: "*"``, and there is **no** ``ec2:StartInstances``,
``ec2:StopInstances``, ``ec2:RebootInstances`` or ``ec2:TerminateInstances``
anywhere in the stack.
"""

from __future__ import annotations

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_logs as logs,
)
from constructs import Construct

from infrastructure.lab_config import LOG_RETENTION_MEMBER_NAMES, LabConfig, bedrock_model_arns


class MonitoringLambda(Construct):
    """Owns the container-image Lambda function and its least-privilege role."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: LabConfig,
        topic_arn: str,
        instance_ids: list[str],
        instance_names: list[str],
    ) -> None:
        super().__init__(scope, construct_id)
        self._config = config

        self.log_group = logs.LogGroup(
            self,
            "LogGroup",
            log_group_name=config.log_group_name,
            retention=getattr(logs.RetentionDays, LOG_RETENTION_MEMBER_NAMES[config.log_retention_days]),
            # `cdk destroy` must delete the log group; see README: Cleanup.
            removal_policy=RemovalPolicy.DESTROY,
        )

        self.role = iam.Role(
            self,
            "Role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Read-only observability + Bedrock invoke + SNS publish. No EC2 mutation.",
        )

        # --- CloudWatch Logs -------------------------------------------------
        group_arn = self.log_group.log_group_arn
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="WriteOwnLogs",
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[group_arn, f"{group_arn}:*"],
            )
        )

        # --- EC2 (read-only, resource scope unsupported by AWS) ---------------
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadEc2StateOnly",
                actions=["ec2:DescribeInstances", "ec2:DescribeInstanceStatus"],
                resources=["*"],
            )
        )

        # --- CloudWatch metrics (read-only, resource scope unsupported) ------
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadCloudWatchMetrics",
                actions=["cloudwatch:GetMetricData", "cloudwatch:GetMetricStatistics", "cloudwatch:ListMetrics"],
                resources=["*"],
            )
        )

        # --- Amazon Bedrock (scoped to the configured model) -----------------
        model_arns = bedrock_model_arns(config.bedrock_model_id, config.aws_region, account=Stack.of(self).account)
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeConfiguredBedrockModel",
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=model_arns,
            )
        )

        # --- SNS publish (scoped to this stack's topic) ---------------------
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="PublishAlertsToLabTopic",
                actions=["sns:Publish"],
                resources=[topic_arn],
            )
        )

        self.function = lambda_.DockerImageFunction(
            self,
            "Handler",
            function_name=f"{config.stack_name}-monitoring",
            description="Runs the CrewAI monitoring crew over EC2 + CloudWatch data (read-only).",
            code=lambda_.DockerImageCode.from_image_asset(
                str(config.lambda_source_dir),
                # The Dockerfile in lambda_src/ installs crewai[bedrock] into a
                # python:3.12-slim virtualenv.
            ),
            role=self.role,
            log_group=self.log_group,
            architecture=lambda_.Architecture.X86_64,
            timeout=Duration.seconds(config.lambda_timeout_seconds),
            memory_size=config.lambda_memory_mb,
            # Lambda must never need a VPC: without it the function uses the
            # public AWS endpoints for Bedrock/SNS/CloudWatch/EC2, which removes
            # the NAT Gateway from the architecture entirely.
            environment={
                # --- Runtime: Lambda's filesystem is read-only except /tmp.
                # CrewAI/ChromaDB derive their storage paths from $HOME (appdirs),
                # so they must be pinned to /tmp or the crew dies on import with
                # OSError: Errno 30 (read-only file system).
                "HOME": "/tmp",
                "XDG_DATA_HOME": "/tmp",
                # --- Bedrock / LLM
                "BEDROCK_MODEL_ID": config.bedrock_model_id,
                "BEDROCK_REASONING_ENABLED": str(config.bedrock_reasoning_enabled).lower(),
                "BEDROCK_REASONING_EFFORT": config.bedrock_reasoning_effort,
                # --- Notification policy
                "ALWAYS_NOTIFY": str(config.always_notify).lower(),
                # --- Monitoring scope and thresholds
                "MONITORING_WINDOW_MINUTES": str(config.monitoring_window_minutes),
                "CPU_WARNING_THRESHOLD": str(config.cpu_warning_threshold),
                "CPU_CRITICAL_THRESHOLD": str(config.cpu_critical_threshold),
                "MONITORED_INSTANCE_IDS": ",".join(instance_ids),
                "MONITORED_INSTANCE_NAMES": ",".join(instance_names),
                # --- Outputs
                "SNS_TOPIC_ARN": topic_arn,
                "LAB_NAME": "agentic-ec2-monitoring",
                "LOG_LEVEL": "INFO",
            },
        )

        CfnOutput(self, "FunctionName", value=self.function.function_name, description="Lambda to invoke manually")
        CfnOutput(self, "FunctionArn", value=self.function.function_arn, description="Lambda ARN")
        CfnOutput(self, "LogGroupName", value=self.log_group.log_group_name, description="Where to read the logs")
        CfnOutput(
            self,
            "LogGroupConsoleUrl",
            value=(
                "https://us-east-1.console.aws.amazon.com/cloudwatch/home?region="
                f"{Stack.of(self).region}#logsV2:log-groups/log-group/{config.log_group_name}"
            ),
            description="AWS Console shortcut to the Lambda log group",
        )
