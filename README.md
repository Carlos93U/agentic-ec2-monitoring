# Agentic EC2 Monitoring

A **CrewAI multi-agent system on Amazon Bedrock** that watches an EC2 fleet and
e-mails a human operator an executive report every 5 minutes. Infrastructure is
defined entirely with the **AWS CDK (Python)**.

An EventBridge Scheduler wakes a Lambda function. A crew of three agents —
Monitoring, Diagnosis and Reporting — reads the instance state and CloudWatch
metrics, classifies each instance as `HEALTHY` / `WARNING` / `CRITICAL`, explains
what is happening and publishes a summary to SNS. The whole system is
deliberately **read-only**: it observes and recommends, it never changes the
infrastructure.

## How it works

```
EventBridge Scheduler ─▶ Lambda (container image) ─▶ CrewAI crew
                             │                          ├─ Monitoring  ─▶ Bedrock
                             │                          ├─ Diagnosis   ─▶ Bedrock (+extended thinking)
                             │                          └─ Reporting   ─▶ Bedrock
                             │
                             ├─ EC2 DescribeInstances (state)   ─▶ report (numbers)
                             ├─ CloudWatch metrics (CPU, net, disk, checks)
                             └─ SNS ─▶ e-mail
```

Interactive diagram: [`docs/architecture.drawio`](docs/architecture.drawio).

### The crew

| Agent | Role |
|---|---|
| Monitoring | Flags anomalous signals per instance (state, CPU, network, EBS I/O, status checks, empty windows) |
| Diagnosis | Classifies `HEALTHY`/`WARNING`/`CRITICAL`, names the evidence, likely cause and recommendation |
| Reporting | Writes the executive summary and priority actions |

Agent verdicts are qualitative only. In parallel, a **deterministic evaluator**
(pure Python) computes the authoritative severities from the collected data and
guards the verdicts: an agent can escalate a finding but never downgrade a
rule-based one. If Bedrock is unavailable, the run degrades gracefully to a
deterministic report instead of failing silently.

### Monitored metrics

`CPUUtilization`, `NetworkIn`/`NetworkOut`, `EBS Read/Write Op`s and
`StatusCheckFailed`, plus the instance state (`running` / `stopped` / not-found)
from `DescribeInstances`.

## Security model

- Read-only by construction: the Lambda role allows `ec2:Describe*`,
  `cloudwatch:GetMetric*`, `bedrock:InvokeModel*`, `sns:Publish` and logs for its
  own log group — nothing more.
- No `ec2:Start/Stop/Reboot/Terminate` anywhere in the stack.
- No SSH or inbound network access; operator access goes through SSM Session
  Manager only.
- IMDSv2 enforced, EBS volumes encrypted, SNS publish restricted to TLS.
- The crew has no tools and no credentials; it only reads the payload it is given.

## Prerequisites

- Python 3.12+
- Docker (the Lambda runs from a container image)
- AWS CLI with credentials to your account: `aws configure` (or `aws sso login`)
- [CDK CLI](https://docs.aws.amazon.com/cdk/v2/guide/cli.html): `cdk --version`
- Bedrock model access for `us.amazon.nova-2-lite-v1:0` in `us-east-1`
  (console → Amazon Bedrock → Model access)

## Deployment

Everything you need to build and deploy the stack:

```bash
# 1. Install the CDK app dependencies (local virtualenv; the Lambda ships its own image)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Configure the lab. Copies the template to `.env` — then EDIT it:
#    set NOTIFICATION_EMAIL to your own address and adjust any thresholds.
cp .env.example .env

# 3. Point at your AWS account (region: us-east-1) and bootstrap once
export AWS_REGION=us-east-1
cdk bootstrap

# 4. Review what will be deployed (no AWS changes)
cdk synth

# 5. Build the Docker image and deploy
cdk deploy
```

Using a named AWS profile instead of the default one?

```bash
cdk deploy --profile your-profile
```

## After deploying

1. **Confirm the SNS e-mail subscription.** AWS sends a confirmation link to the
   address in `NOTIFICATION_EMAIL`; click it, or your reports never get sent
   (console: SNS → Subscriptions → Pending).
2. **Trigger the function by hand** to verify the pipeline end-to-end before
   waiting for the schedule (the function name is the `HandlerFunctionName`
   output of `cdk deploy`, `agentic-ec2-monitoring-monitoring` by default):

   ```bash
   aws lambda invoke \
     --function-name agentic-ec2-monitoring-monitoring \
     --cli-binary-format raw-in-base64-out \
     --payload '{"source":"manual"}' \
     --region us-east-1 \
     /tmp/monitoring-result.json
   ```

3. **Check the result.** The response contains `overall_severity` and, per
   instance, `severity` / `agent_severity`. You will receive the first report by
   e-mail.

## End-to-end test with simulated CPU load

The lab is built to detect change, so it also ships a way to produce it.
`generate_cpu_load.sh` pins one of the lab instances at a target CPU load by
sending busy loops through SSM Run Command — the only channel the instances
expose (no SSH, no inbound ports). It lets you simulate a real incident and
verify the whole flow: load → metrics → crew → severity → e-mail.

Load is expressed as a percentage of the instance's vCPUs. Each busy loop pins
exactly one vCPU. A `t3.micro` has 2 vCPUs (1 core × hyper-threading), so
`--percent 100` starts two loops (~100% CPU) and `--percent 50` one loop (~50%).

```bash
./generate_cpu_load.sh                     # ~100% CPU on instance 1 (default)
./generate_cpu_load.sh --percent 50 --minutes 10
./generate_cpu_load.sh --percent 100 --instance 2
./generate_cpu_load.sh --stop              # kill any load, now
./generate_cpu_load.sh --region us-east-1  # region (default: AWS_REGION or us-east-1)
```

The load is self-terminating (`timeout` + `nohup`), so it stops on its own after
`--minutes` (default 5). Use `--stop` to end it sooner. It targets instances by
their `Name` tag (`agentic-monitoring-ec2-<n>`).

**Worked example:** start an incident, then watch it escalate.

```bash
./generate_cpu_load.sh --percent 50 --minutes 10     # instance 1
./generate_cpu_load.sh --percent 100 --minutes 10    # instance 1, maximum load
# After ~5 minutes the crew reports WARNING → CRITICAL (thresholds below).
./generate_cpu_load.sh --stop                         # resolve the incident
```

Wait one schedule period after starting the load (or invoke the function
manually), and the diagnosis for that instance escalates from `HEALTHY` to
`WARNING` → `CRITICAL`, exactly as configured by `CPU_WARNING_THRESHOLD` /
`CPU_CRITICAL_THRESHOLD`.

## Configuration

All behaviour is set in `.env` (copy `.env.example`). Precedence: real
environment variables → `.env` → `cdk.json` context → built-in defaults. The
same values are forwarded to the Lambda, so `cdk deploy` is the only step needed
to change behaviour.

| Variable | Default | Meaning |
|---|---|---|
| `BEDROCK_MODEL_ID` | `us.amazon.nova-2-lite-v1:0` | Model behind the crew |
| `BEDROCK_REASONING_ENABLED` | `true` | Extended thinking on the Diagnosis agent |
| `BEDROCK_REASONING_EFFORT` | `low` | Thinking effort (`low`/`medium`/`high`) |
| `ALWAYS_NOTIFY` | `false` | E-mail on every run vs only WARNING/CRITICAL |
| `MONITORING_WINDOW_MINUTES` | `10` | Analysis window |
| `CPU_WARNING_THRESHOLD` / `CPU_CRITICAL_THRESHOLD` | `70` / `90` | Severity thresholds (window average) |
| `SCHEDULE_EXPRESSION` | `rate(5 minutes)` | EventBridge Scheduler period |
| `NOTIFICATION_EMAIL` | `you@example.com` | SNS e-mail recipient (must confirm) |
| `EC2_INSTANCE_TYPE` / `EC2_INSTANCE_COUNT` | `t3.micro` / `2` | Lab instances |
| `LAMBDA_TIMEOUT_SECONDS` / `LAMBDA_MEMORY_MB` | `180` / `2048` | Lambda sizing |
| `LOG_RETENTION_DAYS` | `14` | Log group retention |
| `STACK_NAME` | `agentic-ec2-monitoring` | Stack name prefix |
| `AWS_REGION` | `us-east-1` | Region for stack + Lambda/Bedrock calls |

## Observability

The Lambda writes structured JSON logs (one object per line) to CloudWatch with a
correlation `invocation_id`. Useful events: `collection_started/finished`,
`crew_built`, `crew_kickoff`, `crew_finished` (agent severities + token usage),
`sns_published` and `invocation_finished`. Tail them with:

```bash
aws logs tail /aws/lambda/agentic-ec2-monitoring-monitoring --region us-east-1
```

## Project layout

```
├── app.py                     # CDK app entry point
├── cdk.json                   # stack context + configuration defaults
├── generate_cpu_load.sh       # simulate CPU incidents over SSM (e2e test)
├── infrastructure/            # CDK constructs (Python)
│   ├── monitoring_stack.py    #   stacks everything together
│   ├── lab_config.py          #   validated, layered configuration
│   └── constructs/            #   network, EC2, Lambda, SNS, scheduler
├── lambda_src/                # Lambda container image
│   ├── handler.py             #   per-run orchestration
│   ├── crew.py / agents.py / tasks.py
│   ├── collector.py / ec2_monitor.py / cloudwatch_metrics.py
│   ├── evaluator.py           #   deterministic severities + escalation guard
│   ├── report.py              #   e-mail rendering (numbers from payload only)
│   ├── bedrock_probe.py       #   pre-deploy Bedrock access check
│   └── requirements.txt / Dockerfile
├── docs/architecture.drawio   # architecture diagram
├── requirements.txt           # CDK app dependencies (local only)
└── .env.example               # configuration template
```

## Cleanup

`cdk destroy` deletes the stack, its instances, the network, the log group and
the SNS topic. The CDK bootstrap resources (toolkit stack + container-asset ECR
repository) are intentionally kept so the next `cdk deploy` does not need to
bootstrap again.

```bash
export AWS_REGION=us-east-1
cdk destroy
```

## Costs

The lab is designed to stay cheap: 2× `t3.micro` EC2, no NAT gateway, one small
Lambda and one run every 5 minutes (≈288 runs/day, each a few short Bedrock
calls). Open `.env` to tune the instance count, schedule or monitoring window.