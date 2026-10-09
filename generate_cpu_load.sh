#!/usr/bin/env bash
# Generate (or stop) CPU load on an Agentic EC2 Monitoring lab instance.
#
# It talks to the instance through SSM Run Command (AWS-RunShellScript), the
# only channel this lab exposes: no SSH, no inbound security-group port.
#
#   ./generate_cpu_load.sh                     # ~100% CPU on instance 1 (default)
#   ./generate_cpu_load.sh --percent 50 --minutes 10
#   ./generate_cpu_load.sh --percent 100 --instance 2
#   ./generate_cpu_load.sh --stop              # kill any load, now
#   ./generate_cpu_load.sh --region us-east-1
#
# The load is expressed as a percentage of the instance's vCPUs. Each busy loop
# pins exactly one vCPU, so `--percent 100` starts as many loops as the instance
# has vCPUs (a t3.micro has 2, so 100% means two loops and 50% means one).
# A t3.micro has 1 vCPU, so one busy loop pins it well above the 90% CRITICAL
# threshold. The load is self-terminating (`timeout` + nohup), so it ends by
# itself even if you forget about it or the script is interrupted.
set -euo pipefail

# --- defaults ---------------------------------------------------------------
LAB_NAME="${LAB_NAME:-agentic-ec2-monitoring}"
REGION="${AWS_REGION:-us-east-1}"
MINUTES=5
INSTANCE_NUMBER=1
PERCENT=100
STOP=0

usage() {
  sed -n '2,11p' "$0"
  exit "${1:-0}"
}

# --- prerequisites ----------------------------------------------------------
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found" >&2; exit 1; }
aws sts get-caller-identity >/dev/null 2>&1 \
  || { echo "no valid AWS credentials (aws sso login / configure a profile)" >&2; exit 1; }

# --- argument parsing -------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --minutes)  MINUTES="${2:?--minutes needs a number}"; shift 2 ;;
    --instance) INSTANCE_NUMBER="${2:?--instance needs a number}"; shift 2 ;;
    --percent)  PERCENT="${2:?--percent needs a number}"; shift 2 ;;
    --region)   REGION="${2:?--region needs a name}"; shift 2 ;;
    --stop)     STOP=1; shift ;;
    -h|--help)  usage 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; usage 1 ;;
  esac
done

# --- instance lookup (by Name tag, stable across replacements) --------------
INSTANCE_ID="$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:LabName,Values=${LAB_NAME}" \
            "Name=tag:Name,Values=agentic-monitoring-ec2-${INSTANCE_NUMBER}" \
  --query "Reservations[0].Instances[0].InstanceId" --output text 2>/dev/null)"

if [ -z "$INSTANCE_ID" ] || [ "$INSTANCE_ID" = "None" ]; then
  echo "instance agentic-monitoring-ec2-${INSTANCE_NUMBER} not found (LabName=${LAB_NAME})" >&2
  exit 1
fi

# --percent 33/50/66 on a 2-vCPU instance all round to 1 loop, so floor any
# value >= 1 to "at least one vCPU busy". 0 or negative would be meaningless.
if ! [[ "$PERCENT" =~ ^[0-9]+$ ]] || [ "$PERCENT" -lt 1 ]; then
  echo "--percent must be a positive integer (got '$PERCENT')" >&2
  exit 1
fi

STATE="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].State.Name" --output text)"
[ "$STATE" = "running" ] || { echo "instance $INSTANCE_ID is '$STATE'; SSM needs a running instance" >&2; exit 1; }

# --- build the remote command ----------------------------------------------
# The busy loop's own argv (`bash -c while true; do :; done`) is not searchable,
# so `exec -a lab-cpu-load` tags the process for --stop. The marker file records
# the exact PIDs. `timeout` + nohup guarantees the load dies by itself.
MARKER=/tmp/lab-cpu-load.pid

if [ "$STOP" = "1" ]; then
  echo "stopping CPU load on $INSTANCE_ID"
  REMOTE_COMMAND="MARKER=${MARKER}; A='lab-cpu'; B='-load'; PAT=\"\${A}\${B}\"; \
if [ -f \"\$MARKER\" ]; then kill \$(cat \"\$MARKER\") 2>/dev/null; fi; \
SKIP=\$\$; while :; do P=\$(ps -o ppid= -p \"\$SKIP\" 2>/dev/null | tr -d ' '); \
  [ -n \"\$P\" ] && [ \"\$P\" != '0' ] && SKIP=\"\$SKIP \$P\" || break; done; \
V=\$(pgrep -f \"\$PAT\" 2>/dev/null | while read -r q; do case \" \$SKIP \" in *\" \$q \"*) ;; *) echo \"\$q\" ;; esac; done); \
for p in \$V; do kill \"\$p\" 2>/dev/null; done; \
rm -f \"\$MARKER\"; sleep 1; \
SURV=''; for p in \$V; do kill -0 \"\$p\" 2>/dev/null && SURV=\"\$SURV \$p\"; done; \
if [ -n \"\$SURV\" ]; then echo \"STILL RUNNING:\$SURV\"; exit 1; else echo \"stopped (\${#V} process(es))\"; fi"
else
  # vCPUs of the instance type (t3.micro has 2). Number of busy loops is the
  # requested percentage rounded to the nearest whole vCPU, at least one.
  VCPUS="$(aws ec2 describe-instance-types --region "$REGION" \
    --instance-types "$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
      --query "Reservations[0].Instances[0].InstanceType" --output text)" \
    --query "InstanceTypes[0].VCpuInfo.DefaultVCpus" --output text)"
  LOOPS="$(python3 - "$PERCENT" "$VCPUS" <<'PY'
import math, sys
percent, vcpus = int(sys.argv[1]), int(sys.argv[2])
print(max(1, round(percent / 100 * vcpus)))
PY
)"

  echo "generating CPU load: ${PERCENT}% of ${VCPUS} vCPU(s) = ${LOOPS} busy loop(s)"
  echo "for ${MINUTES} minutes on $INSTANCE_ID"
  echo "the deterministic evaluator should report WARNING after ~70% and CRITICAL after ~90%"
  # Launch one busy loop per vCPU, recording each background PID in the marker.
  # Quoting: `$MARKER` must expand on the *remote* side, so it is escaped here.
  REMOTE_COMMAND="MARKER=${MARKER}; rm -f \$MARKER; \
for i in \$(seq 1 ${LOOPS}); do \
  nohup timeout ${MINUTES}m bash -c 'exec -a lab-cpu-load bash -c \"while true; do :; done\"' >/dev/null 2>&1 & \
  echo \$! >> \$MARKER; \
done; \
sleep 1; \
echo started loops: \$(wc -l < \$MARKER); pgrep -f lab-cpu-load | wc -l"
fi

# --- send through SSM and wait ----------------------------------------------
# AWS-RunShellScript runs each `commands` element as one shell string, so the
# whole command must be a single JSON string (no split argv).
PAYLOAD="$(python3 - "$REMOTE_COMMAND" <<'PY'
import json, sys
print(json.dumps({"commands": [sys.argv[1]]}))
PY
)"

COMMAND_ID="$(aws ssm send-command --region "$REGION" \
  --instance-ids "$INSTANCE_ID" \
  --document-name "AWS-RunShellScript" \
  --parameters "$PAYLOAD" \
  --query "Command.CommandId" --output text 2>/dev/null)"

case "$COMMAND_ID" in
  ""|None) echo "could not create the SSM command (is this instance registered with SSM?)" >&2; exit 1 ;;
esac

echo "command id: $COMMAND_ID (usually 5-20 seconds)"

# Poll instead of querying once: fresh commands are not visible to
# get-command-invocation for a moment, so a single query fails even when fine.
TIMEOUT_SECONDS="${SSM_TIMEOUT_SECONDS:-120}"
DEADLINE=$(( $(date +%s) + TIMEOUT_SECONDS ))
STATUS="Pending"

while :; do
  STATUS="$(aws ssm get-command-invocation --region "$REGION" \
    --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
    --query "Status" --output text 2>/dev/null || echo "")"

  case "$STATUS" in
    Success|Cancelled|Failed|TimedOut|AccessDenied|Undeliverable) break ;;
    *) ;;
  esac

  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "timed out after ${TIMEOUT_SECONDS}s waiting for $COMMAND_ID (last status: ${STATUS:-not found})" >&2
    exit 1
  fi
  sleep 2
done

RESULT="$(aws ssm get-command-invocation --region "$REGION" \
  --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
  --query '{status: Status, exit_code: ResponseCode, stdout: StandardOutputContent, stderr: StandardErrorContent}' \
  --output json 2>/dev/null)"

[ -n "$RESULT" ] || { echo "command $COMMAND_ID finished with status '$STATUS' but its output could not be read" >&2; exit 1; }

printf '%s\n' "$RESULT"

EXIT_CODE="$(printf '%s' "$RESULT" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("exit_code", 1))' 2>/dev/null || echo 1)"
if [ "$STATUS" != "Success" ] || [ "$EXIT_CODE" != "0" ]; then
  echo "remote command failed (status $STATUS, exit code ${EXIT_CODE:-unknown})" >&2
  exit "${EXIT_CODE:-1}"
fi