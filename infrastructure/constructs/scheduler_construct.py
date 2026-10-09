"""EventBridge Scheduler: the only thing that decides *when* the lab runs.

Why Scheduler and not a polling loop inside Lambda?

* Zero idle cost: between 18:10 and 18:15 nothing runs.
* No overlapping executions: a rate schedule cannot pile up invocations the way
  an in-Lambda ``while True: sleep()`` loop can.
* Retries and DLQ support belong to the service, not to our code.
* The period is configuration (``SCHEDULE_EXPRESSION``), not code.

Why ``time_window=TimeWindow.off()``?

A *flexible* time window lets EventBridge invoke the schedule anywhere inside a
window (up to 60 minutes). That is great in production and terrible in a lab:
you would not know when to look at your inbox. Turning it off makes runs
predictable, which matters when you are waiting for a specific test result.

Why ``aws_scheduler_targets.LambdaInvoke`` instead of a hand-built
``ScheduleTargetConfig``?

In aws-cdk-lib 2.272.0, ``Schedule.target`` expects an ``IScheduleTarget``
*interface*, not a ``ScheduleTargetConfig`` struct. ``LambdaInvoke`` is the L2
implementation of that interface; it also grants the invoking role
``lambda:InvokeFunction`` on the function, which removes a whole class of
forgetting-the-role mistakes.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_scheduler as scheduler
from aws_cdk import aws_scheduler_targets as targets
from constructs import Construct

from infrastructure.lab_config import LabConfig


class MonitoringScheduler(Construct):
    """Creates the IAM role and the schedule that invoke the Lambda function."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: LabConfig,
        function: lambda_.IFunction,
) -> None:
        super().__init__(scope, construct_id)

        # EventBridge Scheduler assumes this role and calls the Lambda function.
        # Note this is a *separate* role from the Lambda execution role.
        self.role = iam.Role(
            self,
            "SchedulerRole",
            assumed_by=iam.ServicePrincipal("scheduler.amazonaws.com"),
            description="Allows EventBridge Scheduler to invoke the monitoring Lambda function",
        )
        self.role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeMonitoringLambda",
                actions=["lambda:InvokeFunction"],
                resources=[function.function_arn],
            )
        )

        self.target = targets.LambdaInvoke(
            function,
            role=self.role,
            # The handler does not need this payload, but logging who started the
            # run makes CloudWatch Logs far easier to read.
            input=scheduler.ScheduleTargetInput.from_object(
                {
                    "source": "eventbridge-scheduler",
                    "schedule": config.schedule_expression,
                    "lab": "agentic-ec2-monitoring",
                }
            ),
        )

        self.schedule = scheduler.Schedule(
            self,
            "EveryFiveMinutes",
            schedule_name=f"{config.stack_name}-every-5-minutes",
            description=f"CrewAI EC2 monitoring run ({config.schedule_expression})",
            schedule=scheduler.ScheduleExpression.expression(config.schedule_expression),
            target=self.target,
            # Predictable runs: no flexible window, no surprise invocations.
            time_window=scheduler.TimeWindow.off(),
            enabled=True,
        )

        CfnOutput(
            self,
            "ScheduleName",
            value=self.schedule.schedule_name,
            description="EventBridge Scheduler schedule",
        )
        CfnOutput(
            self,
            "SchedulerConsoleUrl",
            value=(
                "https://us-east-1.console.aws.amazon.com/scheduler/home?region="
                f"{Stack.of(self).region}#/schedules/{self.schedule.schedule_name}"
            ),
            description="AWS Console shortcut to the schedule",
        )
