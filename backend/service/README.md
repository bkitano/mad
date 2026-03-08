# MAD Backend Service

API server + Modal workers for autonomous ML experiment execution.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Dashboard   │────▶│  API Server  │────▶│   Supabase    │
│  (React)     │◀─SSE│  (FastAPI)   │     │  (Postgres)   │
└─────────────┘     └──────┬───────┘     └───────────────┘
                           │ POST /experiments
                           ▼
                    ┌──────────────┐
                    │ Modal Worker │
                    │  (container) │
                    └──────────────┘
```

- **API server** — FastAPI app at `mad.briankitano.com`. All state lives in Supabase Postgres.
- **Modal workers** — Containers that run experiments. Spawned automatically when an experiment is created with code.
- **SSE stream** — `/events/stream` uses Supabase Realtime to push events to the dashboard.

## Environment Variables

```bash
# .env
POSTGRES_URL="postgresql://..."           # Supabase pooler connection string
SUPABASE_URL="https://xxx.supabase.co"    # For Realtime SSE streaming
SUPABASE_KEY="eyJ..."                     # Supabase anon key
MODAL_CREATE_JOB_URL="https://...modal.run"  # Modal create_job web endpoint
MAD_SERVICE_URL="http://mad.briankitano.com" # Used by workers to call back to API
```

## Setup

### 1. Install dependencies

```bash
cd backend
uv sync
```

### 2. Run database migrations

Run these against your Supabase Postgres (via SQL Editor or psql):

```bash
psql $POSTGRES_URL -f migrations/001_experiments_and_events.sql
psql $POSTGRES_URL -f migrations/002_event_parent_id.sql
```

Enable Realtime on the events table:

```sql
ALTER PUBLICATION supabase_realtime ADD TABLE events;
```

### 3. Start the API server

```bash
uv run uvicorn service.api:app --host 0.0.0.0 --port 8001
```

The server is available at `http://mad.briankitano.com`.

### 4. Deploy Modal workers

```bash
uv run python -m modal deploy service/modal_worker.py
```

This prints a web endpoint URL like `https://briankitano--mad-worker-create-job.modal.run`. Set it as `MODAL_CREATE_JOB_URL` in your `.env` and restart the API server.

Modal secrets must be configured first:

```bash
modal secret create mad-worker-secrets \
  ANTHROPIC_API_KEY=... \
  WANDB_API_KEY=... \
  MAD_SERVICE_URL=http://mad.briankitano.com \
  POSTGRES_URL=...
```

## Experiment Lifecycle

1. **Create** — `POST /experiments` with `code_files` creates the experiment and stores code
2. **Submit** — If `MODAL_CREATE_JOB_URL` is set, a Modal container is automatically spawned
3. **Run** — The Modal worker claims the proposal, runs the agent, reports results back to the API
4. **Stream** — Events are written to the `events` table and pushed via SSE at `/events/stream`
5. **Cancel** — `POST /experiments/{id}/cancel` updates the DB and terminates the Modal container

## Key API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/experiments` | List experiments (filter by `?status=`) |
| `POST` | `/experiments` | Create experiment (auto-submits to Modal if code provided) |
| `GET` | `/experiments/{id}` | Get experiment details |
| `PATCH` | `/experiments/{id}` | Update experiment fields |
| `POST` | `/experiments/{id}/cancel` | Cancel experiment and kill Modal container |
| `GET` | `/experiments/{id}/events` | Get events for an experiment |
| `GET` | `/experiments/{id}/code` | Get stored code files |
| `GET` | `/events` | Query events (filter by `experiment_id`, `event_type`, `since`) |
| `GET` | `/events/stream` | SSE stream of all new events (Supabase Realtime) |
| `POST` | `/events` | Emit a custom event |
| `GET` | `/proposals` | List proposals (filter by `?status=`) |
| `GET` | `/proposals/{id}` | Get proposal with full content |
| `POST` | `/claims` | Claim a proposal for work |
| `POST` | `/claims/release` | Release a claim |
| `GET` | `/stats` | Experiment statistics |

## Running Tests

```bash
# Unit tests (mocked, no server needed)
uv run pytest tests/test_event_parent_id.py -v

# E2E test (requires running server)
MAD_SERVICE_URL=http://mad.briankitano.com uv run pytest tests/test_e2e_experiment_lifecycle.py -v -s
```

## Running a Worker Locally

For development/debugging, you can run the worker directly instead of through Modal:

```bash
# Run one experiment cycle
uv run python -m service.worker --once

# Run a specific proposal
uv run python -m service.worker --proposal 042-monarch-gated

# Dry run (claim + read, don't execute)
uv run python -m service.worker --dry-run

# Run continuously (polls every 5 min)
uv run python -m service.worker
```

Requires `opencode serve` running locally on port 4096.
