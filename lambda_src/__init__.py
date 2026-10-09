"""CrewAI EC2 monitoring Lambda source.

The package is split into small modules with a single responsibility each::

    handler.py             entry point / orchestration
    config.py              environment variables -> typed settings
    models.py              data contract shared by every module
    ec2_monitor.py         EC2 instance state (read-only)
    cloudwatch_metrics.py  CloudWatch metric collection (read-only)
    collector.py           EC2 + CloudWatch -> MonitoringPayload
    evaluator.py           deterministic severities and safety invariants
    llm_factory.py         CrewAI LLM objects (Bedrock, optional reasoning)
    agents.py / tasks.py   the three agents and their prompts
    crew.py                builds and runs the crew, parses its output
    report.py              renders the final report (numbers come from Lambda)
    sns_publisher.py       notification policy and publishing
    logging_utils.py       structured logging
    bedrock_probe.py       pre-deploy check of the CrewAI/Bedrock integration

None of these modules grants, or is capable of granting, permission to change an
EC2 instance: they only ever call ``Describe*``/``GetMetric*`` APIs.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
