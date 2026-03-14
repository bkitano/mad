FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /api
COPY api/pyproject.toml ./pyproject.toml
RUN uv sync --frozen --no-dev || uv sync --no-dev

FROM python:3.12-slim
COPY --from=builder /api/.venv /api/.venv
ENV PATH="/api/.venv/bin:$PATH"
COPY api/ ./
EXPOSE 8000

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
