# MAD Dashboard Server

Real-time SSE server for viewing MAD Architecture Search experiment status.

## Quick Start

### 1. Install Dependencies

```bash
cd dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install Cloudflare Tunnel

```bash
# Option 1: Package manager
sudo apt install cloudflared

# Option 2: Direct download
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb
```

### 3. Configure Cloudflare Tunnel

```bash
# Login to Cloudflare
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create mad-dashboard

# Note the tunnel UUID from output, then edit cloudflared-config.yml:
# - Replace YOUR_TUNNEL_UUID_HERE with your tunnel UUID
# - Update credentials file path

# Route DNS (replace mad.yoursite.com with your domain)
cloudflared tunnel route dns mad-dashboard mad.yoursite.com
```

### 4. Start Server

```bash
# Manual start
./start-tunnel.sh

# Or run SSE server only (for testing)
source venv/bin/activate
python3 sse_server.py
```

### 5. (Optional) Auto-start with Systemd

```bash
# Edit mad-dashboard.service to match your paths
sudo cp mad-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mad-dashboard
sudo systemctl start mad-dashboard

# Check status
sudo systemctl status mad-dashboard

# View logs
sudo journalctl -u mad-dashboard -f
```

## API Endpoints

- `GET /stream` - SSE stream for real-time updates
- `GET /api/status` - Current experiment status (JSON)
- `GET /api/logs?n=50` - Recent log lines
- `GET /api/result/<id>` - Get experiment result markdown
- `GET /api/experiment-log/<id>` - Get experiment log
- `GET /health` - Health check

## Testing

```bash
# Test SSE stream
curl -N http://localhost:8001/stream

# Test status endpoint
curl http://localhost:8001/api/status | jq

# Test from browser
open http://localhost:8001
```

## Configuration

### Server Settings (sse_server.py)

- `MAX_CONNECTIONS = 100` - Max concurrent SSE connections
- `HEARTBEAT_INTERVAL = 30` - Seconds between heartbeat pings
- Port: `8001`

### Cloudflare Tunnel

Edit `cloudflared-config.yml` to configure your tunnel routing.

## Troubleshooting

### Port already in use
```bash
# Find process using port 8001
lsof -i :8001

# Kill it
kill <PID>
```

### Tunnel not connecting
```bash
# Check tunnel status
cloudflared tunnel list

# Test tunnel locally
cloudflared tunnel --config cloudflared-config.yml run
```

### No updates streaming
- Check that experiments are running and writing to `experiments/active_work.json`
- Check file permissions on experiments directory
- View server logs: `journalctl -u mad-dashboard -f`

## Security

The server:
- Limits connections to 100 concurrent clients
- Sends heartbeats every 30 seconds
- Is read-only (cannot modify experiments)
- Runs in separate process (isolated from experiments)

To add authentication:
1. Add API key to `.env` file
2. Modify `sse_server.py` to check `Authorization` header
3. Update blog to include API key in requests

## Monitoring

```bash
# Active connections
curl http://localhost:8001/health

# Resource usage
htop -p $(pgrep -f sse_server.py)

# Logs
tail -f /path/to/vault/projects/mad-architecture-search/runner.log
```
