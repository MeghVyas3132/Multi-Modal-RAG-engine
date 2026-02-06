#!/usr/bin/env zsh
# ================================================================
# start.sh — Start the entire Multi-Modal RAG stack natively
#
# Usage:
#   ./start.sh          Start all services
#   ./start.sh stop     Stop all services
#   ./start.sh status   Check service status
#   ./start.sh logs     Tail FastAPI logs
# ================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$PROJECT_DIR/.venv"
QDRANT_BIN="$PROJECT_DIR/bin/qdrant"
QDRANT_PID="/tmp/qdrant.pid"
FASTAPI_PID="/tmp/fastapi.pid"
FRONTEND_PID="/tmp/frontend.pid"
FASTAPI_LOG="/tmp/fastapi.log"
QDRANT_LOG="/tmp/qdrant.log"
FRONTEND_LOG="/tmp/frontend.log"

# Homebrew on Apple Silicon
eval "$(/opt/homebrew/bin/brew shellenv zsh)" 2>/dev/null || true
export PATH="/opt/homebrew/bin:$PATH"

# ── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ok()   { echo "${GREEN}✅ $1${NC}"; }
fail() { echo "${RED}❌ $1${NC}"; }
info() { echo "${CYAN}ℹ  $1${NC}"; }
warn() { echo "${YELLOW}⚠  $1${NC}"; }

# ── Helper: wait for a port to become available ─────────────
wait_for_port() {
    local port=$1 name=$2 timeout=${3:-30}
    local elapsed=0
    while ! curl -s "http://localhost:$port" > /dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            fail "$name did not start within ${timeout}s"
            return 1
        fi
    done
    ok "$name is up on :$port  (${elapsed}s)"
}

# ── Helper: check if process is running on a port ───────────
is_port_up() {
    curl -s "http://localhost:$1" > /dev/null 2>&1
}

# ── STOP ────────────────────────────────────────────────────
do_stop() {
    echo ""
    info "Stopping all services..."
    echo ""

    # Frontend
    if [ -f "$FRONTEND_PID" ] && kill -0 "$(cat "$FRONTEND_PID")" 2>/dev/null; then
        kill "$(cat "$FRONTEND_PID")" 2>/dev/null
        rm -f "$FRONTEND_PID"
        ok "Frontend stopped"
    else
        lsof -ti:5173 | xargs kill -9 2>/dev/null && ok "Frontend stopped (port kill)" || warn "Frontend was not running"
    fi

    # FastAPI
    if [ -f "$FASTAPI_PID" ] && kill -0 "$(cat "$FASTAPI_PID")" 2>/dev/null; then
        kill "$(cat "$FASTAPI_PID")" 2>/dev/null
        rm -f "$FASTAPI_PID"
        ok "FastAPI stopped"
    else
        lsof -ti:8000 | xargs kill -9 2>/dev/null && ok "FastAPI stopped (port kill)" || warn "FastAPI was not running"
    fi

    # Qdrant
    if [ -f "$QDRANT_PID" ] && kill -0 "$(cat "$QDRANT_PID")" 2>/dev/null; then
        kill "$(cat "$QDRANT_PID")" 2>/dev/null
        rm -f "$QDRANT_PID"
        ok "Qdrant stopped"
    else
        lsof -ti:6333 | xargs kill -9 2>/dev/null && ok "Qdrant stopped (port kill)" || warn "Qdrant was not running"
    fi

    # Redis
    if redis-cli ping > /dev/null 2>&1; then
        brew services stop redis > /dev/null 2>&1
        ok "Redis stopped"
    else
        warn "Redis was not running"
    fi

    echo ""
    ok "All services stopped."
}

# ── STATUS ──────────────────────────────────────────────────
do_status() {
    echo ""
    echo "${CYAN}═══════════════════════════════════════${NC}"
    echo "${CYAN}   Multi-Modal RAG — Service Status    ${NC}"
    echo "${CYAN}═══════════════════════════════════════${NC}"
    echo ""

    # Redis
    if redis-cli ping 2>/dev/null | grep -q PONG; then
        ok "Redis         :6379"
    else
        fail "Redis         :6379"
    fi

    # Qdrant
    if is_port_up 6333; then
        local vectors=$(curl -s http://localhost:6333/collections 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    names = [c['name'] for c in d['result']['collections']]
    print(f'  ({len(names)} collections: {\", \".join(names)})')
except: print('')" 2>/dev/null)
        ok "Qdrant        :6333${vectors}"
    else
        fail "Qdrant        :6333"
    fi

    # FastAPI
    if is_port_up 8000; then
        local health=$(curl -s http://localhost:8000/health 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    parts = []
    if d.get('clip_loaded'): parts.append('CLIP')
    if d.get('text_embedder_loaded'): parts.append('MiniLM')
    print(f'  (models: {\", \".join(parts)})')
except: print('')" 2>/dev/null)
        ok "FastAPI       :8000${health}"
    else
        fail "FastAPI       :8000"
    fi

    # Frontend
    if is_port_up 5173; then
        ok "Frontend      :5173"
    else
        fail "Frontend      :5173"
    fi

    echo ""
}

# ── LOGS ────────────────────────────────────────────────────
do_logs() {
    if [ -f "$FASTAPI_LOG" ]; then
        tail -f "$FASTAPI_LOG"
    else
        fail "No FastAPI log found at $FASTAPI_LOG"
    fi
}

# ── START ───────────────────────────────────────────────────
do_start() {
    echo ""
    echo "${CYAN}═══════════════════════════════════════${NC}"
    echo "${CYAN}   Multi-Modal RAG — Starting Stack    ${NC}"
    echo "${CYAN}═══════════════════════════════════════${NC}"
    echo ""

    cd "$PROJECT_DIR"

    # ── 1. Redis ────────────────────────────────────────────
    info "Starting Redis..."
    if redis-cli ping 2>/dev/null | grep -q PONG; then
        ok "Redis already running on :6379"
    else
        brew services start redis > /dev/null 2>&1
        sleep 1
        if redis-cli ping 2>/dev/null | grep -q PONG; then
            ok "Redis started on :6379"
        else
            fail "Redis failed to start"
            exit 1
        fi
    fi

    # ── 2. Qdrant ───────────────────────────────────────────
    info "Starting Qdrant..."
    if is_port_up 6333; then
        ok "Qdrant already running on :6333"
    else
        if [ ! -f "$QDRANT_BIN" ]; then
            fail "Qdrant binary not found at $QDRANT_BIN"
            exit 1
        fi
        nohup "$QDRANT_BIN" \
            --storage-path "$PROJECT_DIR/qdrant_storage" \
            --config-path "$PROJECT_DIR/qdrant_config.yaml" \
            > "$QDRANT_LOG" 2>&1 &
        echo $! > "$QDRANT_PID"
        wait_for_port 6333 "Qdrant" 15
    fi

    # ── 3. FastAPI ──────────────────────────────────────────
    info "Starting FastAPI backend..."
    if is_port_up 8000; then
        ok "FastAPI already running on :8000"
    else
        # Kill any orphan process on the port
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true

        nohup "$VENV/bin/python" -m uvicorn \
            services.api_gateway.app:app \
            --host 0.0.0.0 \
            --port 8000 \
            --log-level info \
            > "$FASTAPI_LOG" 2>&1 &
        echo $! > "$FASTAPI_PID"
        info "Loading models (CLIP + MiniLM)... this takes ~15s"
        wait_for_port 8000 "FastAPI" 60
    fi

    # ── 4. Frontend (Vite) ──────────────────────────────────
    info "Starting frontend..."
    if is_port_up 5173; then
        ok "Frontend already running on :5173"
    else
        cd "$PROJECT_DIR/frontend"
        nohup npx vite --port 5173 > "$FRONTEND_LOG" 2>&1 &
        echo $! > "$FRONTEND_PID"
        cd "$PROJECT_DIR"
        wait_for_port 5173 "Frontend" 15
    fi

    # ── Summary ─────────────────────────────────────────────
    echo ""
    echo "${GREEN}═══════════════════════════════════════${NC}"
    echo "${GREEN}   All services running!               ${NC}"
    echo "${GREEN}═══════════════════════════════════════${NC}"
    echo ""
    echo "  ${CYAN}Frontend${NC}    http://localhost:5173"
    echo "  ${CYAN}API${NC}         http://localhost:8000"
    echo "  ${CYAN}API Docs${NC}    http://localhost:8000/docs"
    echo "  ${CYAN}Qdrant${NC}      http://localhost:6333/dashboard"
    echo ""
    echo "  ${YELLOW}Useful commands:${NC}"
    echo "    ./start.sh status   — Check service health"
    echo "    ./start.sh logs     — Tail FastAPI logs"
    echo "    ./start.sh stop     — Stop everything"
    echo ""
}

# ── Main dispatcher ─────────────────────────────────────────
case "${1:-start}" in
    start)  do_start  ;;
    stop)   do_stop   ;;
    status) do_status ;;
    logs)   do_logs   ;;
    *)
        echo "Usage: ./start.sh [start|stop|status|logs]"
        exit 1
        ;;
esac
