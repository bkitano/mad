# MAD Dashboard - Quick Start

Get your dashboard running in 5 minutes.

## Prerequisites

- Python 3.8+
- Node.js (for blog)
- Domain with Cloudflare DNS

## Quick Setup Commands

```bash
# 1. Install Python dependencies
cd ~/Desktop/vault/projects/mad-architecture-search/dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install Cloudflare Tunnel
sudo apt install cloudflared

# 3. Configure Cloudflare Tunnel
cloudflared tunnel login
cloudflared tunnel create mad-dashboard

# 4. Edit config file (replace YOUR_TUNNEL_UUID_HERE with actual UUID)
nano cloudflared-config.yml

# 5. Route DNS (replace mad.yoursite.com)
cloudflared tunnel route dns mad-dashboard mad.yoursite.com

# 6. Start server + tunnel
./start-tunnel.sh

# 7. Test it works
curl https://mad.yoursite.com/health

# 8. Run blog in dev mode (in another terminal)
cd ~/Desktop/vault/blog
npm run dev

# 9. Visit http://localhost:5173/mad
```

## For Production

```bash
# 1. Update blog/.env.local
echo "VITE_MAD_SSE_URL=https://mad.yoursite.com" > ~/Desktop/vault/blog/.env.local

# 2. Build blog
cd ~/Desktop/vault/blog
npm run build

# 3. Deploy to Netlify (git push or manual)

# 4. (Optional) Auto-start server on boot
sudo cp ~/Desktop/vault/projects/mad-architecture-search/dashboard/mad-dashboard.service /etc/systemd/system/
sudo systemctl enable mad-dashboard
sudo systemctl start mad-dashboard
```

## Access Your Dashboard

- **Dev**: http://localhost:5173/mad
- **Prod**: https://yourblog.com/mad

## Common Issues

**"Port 8001 in use"**
```bash
lsof -i :8001 | grep LISTEN | awk '{print $2}' | xargs kill
```

**"Dashboard shows disconnected"**
```bash
# Check server is running
curl http://localhost:8001/health

# Restart server
./start-tunnel.sh
```

**"No experiments showing"**
```bash
# Check active_work.json exists
cat experiments/active_work.json
```

## Next Steps

See `SETUP_GUIDE.md` for detailed documentation.

## What You Built

✅ Real-time SSE server watching your experiments
✅ Cloudflare Tunnel exposing it publicly
✅ React dashboard integrated into your blog
✅ Sub-second latency for status updates
✅ Automatic reconnection on connection drops

Enjoy! 🎉
