# MAD Dashboard Setup Guide

Complete guide to set up real-time experiment monitoring on your blog.

## Architecture Overview

```
Your Machine (experiments running)
    ↓
SSE Server (localhost:8001)
    ↓
Cloudflare Tunnel (public URL)
    ↓
Your Blog (React dashboard at /mad)
```

## Step 1: Install Dependencies

### Python Dependencies

```bash
cd ~/Desktop/vault/projects/mad-architecture-search/dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Cloudflare Tunnel

```bash
# Option 1: APT (Ubuntu/Debian)
sudo apt update
sudo apt install cloudflared

# Option 2: Direct download
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Verify installation
cloudflared --version
```

## Step 2: Configure Cloudflare Tunnel

### 2.1 Login to Cloudflare

```bash
cloudflared tunnel login
```

This opens a browser window. Select your domain and authorize.

### 2.2 Create Tunnel

```bash
cloudflared tunnel create mad-dashboard
```

Output will show:
```
Tunnel credentials written to /home/bkitano/.cloudflared/[UUID].json
Created tunnel mad-dashboard with id [UUID]
```

**Save the UUID - you'll need it next!**

### 2.3 Configure the Tunnel

Edit `cloudflared-config.yml` and replace `YOUR_TUNNEL_UUID_HERE` with your actual UUID:

```yaml
tunnel: abc123-your-actual-uuid-here
credentials-file: /home/bkitano/.cloudflared/abc123-your-actual-uuid-here.json

ingress:
  - hostname: mad.yoursite.com
    service: http://localhost:8001
  - service: http_status:404
```

Replace `mad.yoursite.com` with your desired subdomain.

### 2.4 Route DNS

```bash
cloudflared tunnel route dns mad-dashboard mad.yoursite.com
```

This creates a CNAME record pointing `mad.yoursite.com` to your tunnel.

## Step 3: Test SSE Server Locally

```bash
# Activate venv if not already active
source venv/bin/activate

# Start server
python3 sse_server.py
```

In another terminal:
```bash
# Test SSE stream
curl -N http://localhost:8001/stream

# Test API
curl http://localhost:8001/api/status | jq
```

You should see experiment data streaming!

## Step 4: Test Tunnel

```bash
# Start both server and tunnel
./start-tunnel.sh
```

You should see:
```
Starting SSE server on port 8001...
Starting Cloudflare tunnel...

=================================
Dashboard is running!
=================================

Local:  http://localhost:8001
Public: https://mad.yoursite.com

Press Ctrl+C to stop
```

Test the public URL:
```bash
curl https://mad.yoursite.com/health
```

## Step 5: Configure Blog

### 5.1 Update Environment Variables

Edit `blog/.env.local`:

```bash
# For development (local testing)
VITE_MAD_SSE_URL=http://localhost:8001

# For production (after deploying)
# VITE_MAD_SSE_URL=https://mad.yoursite.com
```

### 5.2 Test in Development

```bash
cd ~/Desktop/vault/blog
npm run dev
```

Visit `http://localhost:5173/mad` - you should see the dashboard!

## Step 6: Deploy to Production

### 6.1 Update Production URL

Edit `blog/.env.local`:
```bash
VITE_MAD_SSE_URL=https://mad.yoursite.com
```

### 6.2 Build and Deploy Blog

```bash
cd ~/Desktop/vault/blog
npm run build
# Then deploy to Netlify (git push or manual upload)
```

### 6.3 Auto-start SSE Server (Optional)

To run the dashboard server automatically on boot:

```bash
# Edit service file with your paths
sudo cp mad-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mad-dashboard
sudo systemctl start mad-dashboard

# Check status
sudo systemctl status mad-dashboard

# View logs
sudo journalctl -u mad-dashboard -f
```

## Usage

### Access Dashboard

- **Local**: `http://localhost:5173/mad` (dev)
- **Production**: `https://yourblog.com/mad`

### Stop Server

```bash
# If running via start-tunnel.sh
Ctrl+C

# If running via systemd
sudo systemctl stop mad-dashboard
```

### View Logs

```bash
# SSE server logs (if using systemd)
sudo journalctl -u mad-dashboard -f

# Or if running manually, check terminal output
```

## Troubleshooting

### "Port 8001 already in use"

```bash
# Find process
lsof -i :8001

# Kill it
kill <PID>

# Or use a different port (edit sse_server.py)
```

### "Tunnel not connecting"

```bash
# Check tunnel status
cloudflared tunnel list

# Test configuration
cloudflared tunnel --config cloudflared-config.yml run

# Check DNS propagation
dig mad.yoursite.com
```

### "Dashboard shows 'Disconnected'"

1. Check SSE server is running: `curl http://localhost:8001/health`
2. Check tunnel is running: `curl https://mad.yoursite.com/health`
3. Check browser console for errors
4. Verify `.env.local` has correct URL

### "No experiments showing"

1. Check experiments are actually running
2. Verify `experiments/active_work.json` exists and has data
3. Check file permissions: `ls -la experiments/`
4. Restart SSE server

### "Connection drops frequently"

- Cloudflare Tunnel has built-in retry logic
- SSE client auto-reconnects every 5 seconds
- Check network stability
- Increase `HEARTBEAT_INTERVAL` in `sse_server.py` if needed

## Security Considerations

### Current Setup (No Auth)
- Dashboard is **publicly accessible** at `https://mad.yoursite.com`
- Read-only (cannot modify experiments)
- Shows experiment status to anyone with the URL

### Adding Authentication (Optional)

If you want to restrict access:

1. **Option 1: Cloudflare Access** (Recommended)
   - Add Cloudflare Access policy to `mad.yoursite.com`
   - Require login before accessing
   - Free for up to 50 users

2. **Option 2: API Key**
   - Add token check to `sse_server.py`
   - Store token in `blog/.env.local`
   - Include in request headers

3. **Option 3: IP Whitelist**
   - Configure Cloudflare firewall rules
   - Only allow specific IPs

## Monitoring

### Check Server Health

```bash
# HTTP endpoint
curl http://localhost:8001/health

# Check active connections
curl http://localhost:8001/api/status | jq '.active_connections'
```

### Resource Usage

```bash
# Find PID
pgrep -f sse_server.py

# Monitor with htop
htop -p $(pgrep -f sse_server.py)

# Check memory
ps aux | grep sse_server
```

### Expected Usage
- **CPU**: < 1% idle, < 10% when streaming
- **Memory**: ~30-50MB
- **Connections**: 1-10 concurrent (typical)

## Advanced Configuration

### Change Port

Edit `sse_server.py`:
```python
app.run(
    host='0.0.0.0',
    port=8002,  # Change here
    threaded=True,
    debug=False,
)
```

Then update `cloudflared-config.yml`:
```yaml
service: http://localhost:8002
```

### Adjust Connection Limits

Edit `sse_server.py`:
```python
MAX_CONNECTIONS = 200  # Increase from 100
HEARTBEAT_INTERVAL = 60  # Increase from 30 seconds
```

### Custom Log Filtering

Modify `get_recent_logs()` in `sse_server.py` to filter specific log levels:

```python
def get_recent_logs(n=50, level='INFO'):
    lines = f.readlines()
    filtered = [l for l in lines if level in l]
    return filtered[-n:]
```

## Maintenance

### Update Dependencies

```bash
cd dashboard
source venv/bin/activate
pip install --upgrade flask flask-cors watchdog
```

### Backup Configuration

```bash
# Backup tunnel config
cp cloudflared-config.yml cloudflared-config.yml.backup

# Backup tunnel credentials
cp ~/.cloudflared/*.json ~/backup/
```

### Rotate Tunnel

If you need to recreate the tunnel:

```bash
# Delete old tunnel
cloudflared tunnel delete mad-dashboard

# Create new tunnel
cloudflared tunnel create mad-dashboard-v2

# Update config with new UUID
# Update DNS routing
```

## Next Steps

- Add authentication if needed
- Monitor resource usage
- Set up alerts for agent failures
- Add more metrics to dashboard
- Integrate W&B charts directly

## Support

If you run into issues:

1. Check logs: `journalctl -u mad-dashboard -f`
2. Test each component individually (SSE server → tunnel → blog)
3. Verify firewall rules aren't blocking connections
4. Check Cloudflare dashboard for tunnel status

## Files Created

```
projects/mad-architecture-search/dashboard/
├── sse_server.py              ✅ SSE server
├── requirements.txt           ✅ Python deps
├── cloudflared-config.yml     ✅ Tunnel config
├── start-tunnel.sh            ✅ Start script
├── mad-dashboard.service      ✅ Systemd service
├── README.md                  ✅ API docs
└── SETUP_GUIDE.md            ✅ This file

blog/
├── src/
│   ├── App.tsx               ✅ Modified (added /mad route)
│   ├── pages/
│   │   └── MADDashboard.tsx  ✅ Dashboard page
│   └── components/
│       ├── Layout.tsx         ✅ Modified (added nav link)
│       └── mad/
│           ├── ExperimentCard.tsx  ✅ Experiment cards
│           ├── AgentStatus.tsx     ✅ Agent health
│           └── LogViewer.tsx       ✅ Log viewer
├── vite.config.ts            ✅ Modified (added proxy)
└── .env.local                ✅ Environment config
```

You're all set! 🚀
