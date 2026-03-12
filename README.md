## Running the Worker Locally
0. Make sure your `mad/.env` has everything in it.
```
POSTGRES_URL=""
SUPABASE_URL=""
SUPABASE_KEY=""
MODAL_CREATE_JOB_URL=""
WANDB_API_KEY=""
ANTHROPIC_API_KEY=""
OPENCODE_GO_API_KEY=""
OPENCODE_CONFIG="./opencode.jsonc"
```

1. Start the API.
```
cd mad/backend/
set -a; source ../.env; uv run uvicorn api.api:app --port 8001 --reload
```

2. Start up Opencode in a directory `mad/.e2e/`
```
cd mad/
set -a; source .env; opencode serve
```

3. Run the worker
```
cd mad/backend/
rm -rf ../.e2e/*; export MAD_WORKSPACE="$HOME/Desktop/projects/mad/.e2e/"; export MAD_SERVICE_URL=<local-api-host:port>; uv run python -m service.worker --proposal   999-mnist-e2e-test
```

## Deploying to Modal

