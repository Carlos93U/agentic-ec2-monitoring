"""Minimal VPC for the lab EC2 instances.

Why a hand-made VPC instead of the account's default VPC?

* Reproducibility: the lab behaves the same in every account.
* Cost: ``nat_gateways=0``. A NAT Gateway costs roughly USD 32/month plus data
  processing, which is absurd for a monitoring lab.
* Simplicity: exactly one public subnet is enough.

Why are the instances in a **public** subnet?

The instances are not monitored *through* the network; they are monitored through
CloudWatch, which needs no connectivity at all. The only reason they need outbound
internet is the SSM Agent, so that you can open an operator session
(``aws ssm start-session``) to launch the CPU-load scenario. A public subnet with
an Internet Gateway is the cheapest way to get that.

Production hardening for this VPC (not needed for the lab):

* put the instances in private subnets,
* add interface VPC endpoints for ``ssm``/``ssmmessages``/``ec2messages``,
  or add a NAT Gateway,
* keep the security group without any ingress rule (which is already the case).
"""

from __future__ import annotations

from aws_cdk import aws_ec2 as ec2
from constructs import Construct


class LabNetwork(Construct):
    """A single-AZ-friendly VPC with one public subnet and no NAT."""

    def __init__(self, scope: Construct, construct_id: str, *, max_azs: int = 2) -> None:
        super().__init__(scope, construct_id)

        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=max_azs,
            # No NAT Gateway: the lab must stay cheap. Instances reach the
            # internet through the Internet Gateway instead.
            nat_gateways=0,
            enable_dns_hostnames=True,
            enable_dns_support=True,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                )
            ],
        )

        self.security_group = ec2.SecurityGroup(
            self,
            "InstanceSecurityGroup",
            vpc=self.vpc,
            description="Lab instances: no inbound access at all, operator access via SSM only",
            # No ingress rule is declared on purpose: SSH/HTTP are never opened.
            allow_all_outbound=True,
        )
        self.node.add_metadata(
            "Warning", "No ingress rules: monitoring uses CloudWatch, access uses SSM."
        )

    @property
    def public_subnets(self) -> ec2.SubnetSelection:
        """Subnet selection used for the monitored instances."""
        return ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC)
