#!/usr/bin/env bash
# Check network connectivity between a worker and the Ray head node.
# Run this ON THE WORKER machine before joining the cluster.
#
# Usage: ./scripts/check-ray-ports.sh [head-ip]

set -euo pipefail

HEAD_IP="${1:-192.168.178.32}"
WORKER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || ip -4 addr show | grep -oP 'inet 192\.168\.\d+\.\d+' | head -1 | grep -oP '192\.168\.\d+\.\d+' || echo "unknown")

echo "=== Ray Cluster Connectivity Check ==="
echo "Head node:   $HEAD_IP"
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
# Ray workers need to be reachable from the head for object transfer.
echo "--- Head -> Worker (reverse) ---"
echo "  Testing if head can reach this worker at $WORKER_IP..."

# Start a temporary listener, test from head via ssh, then clean up
TEST_PORT=19999
# Start listener in background
(timeout 10 bash -c "echo >/dev/tcp/0.0.0.0/$TEST_PORT || nc -l -p $TEST_PORT >/dev/null 2>&1" 2>/dev/null || python3 -c "
import socket, threading
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', $TEST_PORT)); s.listen(1); s.settimeout(10)
try: s.accept()
except: pass
s.close()
") &
LISTENER_PID=$!
sleep 1

if ssh -o ConnectTimeout=3 -o BatchMode=yes kristian@"$HEAD_IP" \
    "timeout 3 bash -c 'echo >/dev/tcp/$WORKER_IP/$TEST_PORT'" 2>/dev/null; then
    echo "  OK   $HEAD_IP -> $WORKER_IP:$TEST_PORT (reverse connectivity)"
    PASS=$((PASS + 1))
else
    echo "  FAIL $HEAD_IP -> $WORKER_IP:$TEST_PORT (reverse connectivity)"
    echo "       Head cannot reach worker. Check:"
    echo "       - Worker firewall (ufw/iptables/Windows Firewall)"
    echo "       - WSL: networkingMode=mirrored in .wslconfig"
    FAIL=$((FAIL + 1))
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
    echo "  Worker firewall:  open TCP+UDP from LAN (192.168.178.0/24)"
    echo "  WSL networking:   add networkingMode=mirrored to .wslconfig"
    exit 1
fi
