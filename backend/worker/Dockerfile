FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY worker/pyproject.toml ./pyproject.toml
RUN uv sync --frozen --no-dev || uv sync --no-dev

FROM python:3.12-slim
RUN apt-get update && apt-get install -y curl ca-certificates git && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://opencode.ai/install | bash \
    && ln -s /root/.opencode/bin/opencode /usr/local/bin/opencode

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY worker/ ./worker/
COPY worker/opencode/opencode.jsonc /workspace/opencode.jsonc
RUN mkdir -p /workspace/proposals /workspace/code /workspace/experiments
ENV MAD_WORKSPACE=/workspace
ENV OPENCODE_CONFIG=/workspace/opencode.jsonc
# Default entrypoint for standalone use; Modal overrides this with its own entrypoint
CMD ["python", "-m", "worker.worker"]
