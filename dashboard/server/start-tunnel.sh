#!/bin/bash
# Start MAD Dashboard Server with Cloudflare Tunnel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "==================================="
echo "MAD Dashboard Server"
echo "==================================="
echo ""

# Check if Python venv exists
if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$SCRIPT_DIR/venv"
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "ERROR: cloudflared is not installed"
    echo ""
    echo "Install with:"
    echo "  sudo apt install cloudflared"
    echo "  # or"
    echo "  wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"
    echo "  sudo dpkg -i cloudflared-linux-amd64.deb"
    echo ""
    exit 1
fi

# Check if tunnel is configured
if [ ! -f "$SCRIPT_DIR/cloudflared-config.yml" ]; then
    echo "ERROR: cloudflared-config.yml not found"
    echo "Copy cloudflared-config.yml.example and configure it"
    exit 1
fi

if grep -q "YOUR_TUNNEL_UUID_HERE" "$SCRIPT_DIR/cloudflared-config.yml"; then
    echo "ERROR: cloudflared-config.yml not configured"
    echo ""
    echo "Setup steps:"
    echo "1. cloudflared tunnel login"
    echo "2. cloudflared tunnel create mad-dashboard"
    echo "3. Update cloudflared-config.yml with your tunnel UUID"
    echo "4. cloudflared tunnel route dns mad-dashboard mad.yoursite.com"
    echo ""
    exit 1
fi

echo "Starting FastAPI server on port 8001..."
python3 -m uvicorn sse_server:app --host 0.0.0.0 --port 8001 &
SSE_PID=$!

# Wait for server to start
sleep 2

echo "Starting Cloudflare tunnel..."
cloudflared tunnel --config "$SCRIPT_DIR/cloudflared-config.yml" run &
TUNNEL_PID=$!

echo ""
echo "==================================="
echo "Dashboard is running!"
echo "==================================="
echo ""
echo "Local:  http://localhost:8001"
echo "Public: https://mad.yoursite.com"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $SSE_PID 2>/dev/null || true
    kill $TUNNEL_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

# Wait for processes
wait
