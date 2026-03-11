.PHONY: build-api build-worker build-all run-api run-worker stop clean

# ── Build ────────────────────────────────────────────────────────────────────

build-api:
	docker build -f api/Dockerfile -t mad-api:latest .

build-worker:
	docker build -f worker/Dockerfile -t mad-worker:latest .

build-all: build-api build-worker

# ── Run ──────────────────────────────────────────────────────────────────────

run-api:
	docker run --rm -p 8000:8000 --env-file ../.env --name mad-api mad-api:latest

run-worker:
	docker run --rm --env-file ../.env --name mad-worker mad-worker:latest

run-worker-dry:
	docker run --rm --env-file ../.env --name mad-worker mad-worker:latest --dry-run --once

run-worker-idle:
	docker run --rm --env-file ../.env -p 4096:4096 --name mad-worker mad-worker:latest --idle

# ── Stop ─────────────────────────────────────────────────────────────────────

stop:
	-docker stop mad-api mad-worker 2>/dev/null

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	-docker rmi mad-api:latest mad-worker:latest 2>/dev/null

# ── Test ─────────────────────────────────────────────────────────────────────

test:
	uv run python -m pytest tests/test_opencode_types.py tests/test_opencode_service.py -v

test-all:
	uv run python -m pytest tests/ -v
