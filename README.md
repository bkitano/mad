## Running the Worker Locally
1. Make sure your `mad/.env` has everything in it.
```bash
PGUSER=""
PGPASSWORD=""
PGHOST=""
PGPORT=""
PGDATABASE=postgres
SUPABASE_URL=""
SUPABASE_KEY=""
MODAL_CREATE_JOB_URL=""
WANDB_API_KEY=""
ANTHROPIC_API_KEY=""
OPENCODE_GO_API_KEY=""
OPENCODE_CONFIG="path/to/opencode.jsonc"
HF_TOKEN=""
HF_REPO_ID=""
```

2. Start ngrok.
```bash
ngrok http 8001
```

3. Start the API.
```bash
cd mad/backend/
set -a; source ../../.env; uv run uvicorn app:app --port 8001 --reload
```

4. Run the worker
```bash
cd mad/backend/
rm -rf ../.e2e/*; MAD_SERVICE_URL=<YOUR_NGROK_URL> pytest tests/test_e2e_mnist.py -v -s
```

## Using the Dashboard

### Production
https://madder.netlify.app/

### Local
1. Start ngrok for your backend API if you haven't already.
```bash
ngrok http 8001
```
2. Start the dashboard app.
```bash
cd mad/dashboard
VITE_API_URL=<YOUR_NGROK_URL> npm run dev
```

## Deploying to Modal
