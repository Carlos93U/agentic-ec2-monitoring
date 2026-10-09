"""The two EC2 instances the lab monitors.

Design decisions
----------------
* ``t3.micro`` + Amazon Linux 2023: the smallest sensible footprint for a lab.
* ``Name`` tags are exactly ``agentic-monitoring-ec2-1`` / ``-2`` as required.
* IMDSv2 is enforced (``require_imdsv2=True``).
* ``detailed_monitoring=True`` gives 1-minute CloudWatch resolution. Without it
  EC2 publishes every 5 minutes, which makes a 5-10 minute analysis window
  unusable for the CPU scenario. It costs a few cents per instance per month.
* The instance profile grants **only** ``AmazonSSMManagedInstanceCore``. It is
  there so a human can open an operator session, not so monitoring can read
  anything: the monitoring path is CloudWatch + ``ec2:Describe*`` from Lambda.
* No key pair and no SSH ingress: there is literally no way to log in with SSH.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Stack
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct

from infrastructure.lab_config import LabConfig


class MonitoredInstances(Construct):
    """Creates ``ec2_instance_count`` instances tagged for the lab."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: LabConfig,
        vpc: ec2.IVpc,
        subnets: ec2.SubnetSelection,
        security_group: ec2.ISecurityGroup,
    ) -> None:
        super().__init__(scope, construct_id)
        self._config = config

        # Operator-only access. No mutating or monitoring permissions live here.
        self.instance_role = iam.Role(
            self,
            "OperatorRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description=(
                "Lets an operator open an SSM Session Manager session on the lab instances. "
                "It intentionally grants no EC2 control-plane permissions."
            ),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"),
            ],
        )

        # AL2023 ships the SSM Agent; make sure it is enabled and running so
        # `aws ssm start-session` works right after the first boot.
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            "#!/bin/bash",
            "set -euo pipefail",
            "# Monitoring never uses SSH. This only enables the operator channel.",
            "if command -v amazon-ssm-agent >/dev/null 2>&1; then",
            "  systemctl enable amazon-ssm-agent || true",
            "  systemctl start amazon-ssm-agent || true",
            "fi",
            "mkdir -p /var/log/lab",
            "echo \"agentic-monitoring lab instance\" > /var/log/lab/README.txt",
        )

        self.instances: list[ec2.Instance] = []
        for index, name in enumerate(config.instance_names, start=1):
            instance = ec2.Instance(
                self,
                f"Ec2{index}",
                # instance_name sets the Name tag; extra tags come from Tags.of() at
                # the stack level, because ec2.Instance has no `tags` prop.
                instance_name=name,
                instance_type=ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.MICRO),
                machine_image=ec2.MachineImage.latest_amazon_linux2023(),
                vpc=vpc,
                vpc_subnets=subnets,
                associate_public_ip_address=True,
                security_group=security_group,
                role=self.instance_role,
                require_imdsv2=True,
                detailed_monitoring=config.detailed_monitoring,
                user_data=user_data,
                block_devices=[
                    ec2.BlockDevice(
                        device_name="/dev/xvda",
                        volume=ec2.BlockDeviceVolume.ebs(
                            volume_size=8,
                            volume_type=ec2.EbsDeviceVolumeType.GP3,
                            encrypted=True,
                            delete_on_termination=True,
                        ),
                    )
                ],
            )
            self.instances.append(instance)

            CfnOutput(self, f"Ec2{index}Id", value=instance.instance_id, description=f"Instance ID of {name}")
            CfnOutput(
                self,
                f"Ec2{index}PrivateIp",
                value=instance.instance_private_ip,
                description=f"Private IP of {name}",
            )
            CfnOutput(
                self,
                f"Ec2{index}ConsoleUrl",
                value=(
                    "https://us-east-1.console.aws.amazon.com/ec2/home?region="
                    f"{Stack.of(self).region}#InstanceDetails:instanceId={instance.instance_id}"
                ),
                description=f"AWS Console shortcut for {name}",
            )

    @property
    def instance_ids(self) -> list[str]:
        """Instance IDs (CDK tokens)."""
        return [instance.instance_id for instance in self.instances]

    @property
    def names(self) -> list[str]:
        """The ``Name`` tags of the lab instances."""
        return list(self._config.instance_names)
