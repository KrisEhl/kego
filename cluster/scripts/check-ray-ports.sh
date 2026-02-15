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
# Ray needs bidirectional TCP between head and workers for object transfer.
echo "--- Head -> Worker (reverse) ---"
echo "  Run this ON THE HEAD NODE ($HEAD_IP) to verify:"
echo ""
echo "    timeout 3 bash -c 'echo >/dev/tcp/$WORKER_IP/22' && echo OK || echo FAIL"
echo ""
echo "  If FAIL: open worker firewall with:"
echo "    sudo ufw allow from 192.168.178.0/24"
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
