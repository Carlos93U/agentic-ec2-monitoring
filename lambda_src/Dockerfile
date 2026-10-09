# syntax=docker/dockerfile:1
#
# Lambda container image. Built by CDK (DockerImageCode.from_image_asset) on every
# deploy that changes the source, and pushed to the CDK-managed ECR repository.
#
# Why the AWS base image and not `python:3.12-slim`:
#   A custom base has no Lambda Runtime Interface Client. `aws-lambda-ric` is not
#   on PyPI, so a slim image cannot be made to boot as a Lambda at all -- and CDK
#   would deploy it happily. `public.ecr.aws/lambda/python:3.12` ships the runtime,
#   the correct ENTRYPOINT, and the local Runtime Interface Emulator, so the same
#   image runs under `docker run -p 9000:8080` and on Lambda unchanged.

FROM public.ecr.aws/lambda/python:3.12

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AWS_DEFAULT_REGION=us-east-1

# Dependencies first: this layer is cached until requirements.txt changes, so
# code-only edits rebuild in seconds.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# The source lives in a `lambda_src/` package inside the task root, because every
# module imports `from lambda_src.x import y`. COPY with context = lambda_src/ has
# to target the package directory; copying it flat would put handler.py at the
# task root with no `lambda_src` to import from, which is a ModuleNotFoundError on
# every invocation.
COPY . ${LAMBDA_TASK_ROOT}/lambda_src/

# Non-root, matching the rest of the lab's posture. The AL2023 base does not ship
# shadow-utils, so the account is written directly rather than pulling in a package
# just to create it. The numeric uid is what USER pins, so name resolution is not
# on the critical path. The runtime and the RIE are world-readable and the emulator
# port is unprivileged, so no capability is needed.
RUN printf 'appuser:x:10001:10001::/home/appuser:/sbin/nologin\n' >> /etc/passwd \
 && printf 'appuser:x:10001:\n' >> /etc/group \
 && mkdir -p /home/appuser \
 && chown -R 10001:10001 /home/appuser ${LAMBDA_TASK_ROOT}
USER 10001

# ENTRYPOINT comes from the base image (/lambda-entrypoint.sh); Lambdas pass their
# handler as the CMD argument. CDK also sets this explicitly via the construct.
CMD ["lambda_src.handler.lambda_handler"]
