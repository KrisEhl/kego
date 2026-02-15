#!/usr/bin/env bash
# Check network connectivity between a worker and the Ray head node.
# Run this ON THE WORKER machine before joining the cluster.
#
# Usage: ./cluster/scripts/check-ray-ports.sh [head-ip] [head-ssh]
# Example: ./cluster/scripts/check-ray-ports.sh 192.168.178.32 kristian@omarchy.fritz.box

set -euo pipefail

HEAD_IP="${1:-192.168.178.32}"
HEAD_SSH="${2:-kristian@omarchyd.fritz.box}"
WORKER_IP=$(ip -4 route get 1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}' \
    || ipconfig getifaddr en0 2>/dev/null \
    || hostname -I 2>/dev/null | awk '{print $1}' \
    || echo "unknown")

echo "=== Ray Cluster Connectivity Check ==="
echo "Head node:   $HEAD_IP"
echo "Head SSH:    $HEAD_SSH"
echo "Worker IP:   $WORKER_IP"
echo ""

PASS=0
FAIL=0

check_port() {
    local host=$1 port=$2 desc=$3
    if timeout 3 bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo "  OK   $host:$port  ($desc)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL $host:$port  ($desc)"
        FAIL=$((FAIL + 1))
    fi
}

# --- Worker -> Head connectivity ---
echo "--- Worker -> Head ---"
check_port "$HEAD_IP" 6379  "GCS server (Ray head)"
check_port "$HEAD_IP" 8265  "Dashboard"
check_port "$HEAD_IP" 10001 "Ray client server"
check_port "$HEAD_IP" 5000  "MLflow server"
echo ""

# --- Ping ---
echo "--- Ping ---"
if ping -c 1 -W 2 "$HEAD_IP" >/dev/null 2>&1; then
    echo "  OK   ping $HEAD_IP"
    PASS=$((PASS + 1))
else
    echo "  FAIL ping $HEAD_IP"
    FAIL=$((FAIL + 1))
fi
echo ""

# --- Head -> Worker connectivity (reverse) ---
# Ray needs bidirectional TCP between head and workers for object transfer.
# Start a temporary listener, then ask the head to connect to it.
echo "--- Head -> Worker (reverse) ---"
TEST_PORT=19999

# Start a TCP listener on the worker
python3 -c "
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', $TEST_PORT)); s.listen(1); s.settimeout(8)
try:
    conn, _ = s.accept(); conn.close()
except socket.timeout:
    s.close(); sys.exit(1)
s.close()
" &
LISTENER_PID=$!
sleep 1

# Connect from head to worker's test port
if timeout 10 ssh -o ConnectTimeout=3 -o BatchMode=yes "$HEAD_SSH" \
    "python3 -c \"import socket; s=socket.socket(); s.settimeout(3); s.connect(('$WORKER_IP', $TEST_PORT)); s.close(); print('OK')\"" 2>/dev/null; then
    echo "  OK   $HEAD_IP -> $WORKER_IP:$TEST_PORT"
    PASS=$((PASS + 1))
else
    # SSH might not have key auth — print manual instructions
    echo "  SKIP Could not auto-test (SSH key auth to head required)"
    echo ""
    echo "  To test manually, run ON THE WORKER:"
    echo "    python3 -c \"import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); s.bind(('0.0.0.0',$TEST_PORT)); s.listen(1); s.settimeout(30); print('Listening on $TEST_PORT...'); conn,a=s.accept(); print(f'Connected from {a}'); conn.close(); s.close()\""
    echo ""
    echo "  Then ON THE HEAD NODE ($HEAD_IP):"
    echo "    python3 -c \"import socket; s=socket.socket(); s.settimeout(3); s.connect(('$WORKER_IP', $TEST_PORT)); print('OK'); s.close()\""
fi
kill $LISTENER_PID 2>/dev/null || true
wait $LISTENER_PID 2>/dev/null || true
echo ""

# --- Summary ---
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Troubleshooting:"
    echo "  Head firewall:    sudo ufw allow from 192.168.178.0/24"
    echo "  Worker firewall:  sudo ufw allow from 192.168.178.0/24"
    echo "  WSL networking:   add networkingMode=mirrored to .wslconfig"
    exit 1
fi
