FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY api/pyproject.toml ./pyproject.toml
RUN uv sync --frozen --no-dev || uv sync --no-dev

FROM python:3.12-slim
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app
COPY api/ ./api/
COPY migrations/ ./migrations/
EXPOSE 8000
ENTRYPOINT ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
