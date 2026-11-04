#!/usr/bin/env bash
# bench/wait-ready.sh [container]: wait until the server answers /health or the container dies; prints startup time.
NAME=${1:-tidal-moe}; PORT=${PORT:-8000}; start=$(date +%s)
while true; do
  [ "$(curl -s -o /dev/null -w "%{http_code}" "localhost:$PORT/health")" = 200 ] && { echo "$NAME ready after $(( $(date +%s)-start ))s"; exit 0; }
  [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || {
    echo "$NAME exited"; docker logs --tail 60 "$NAME" 2>&1 | grep -v "^\s*File \|^\s*\^" | cut -c1-300; exit 1; }
  sleep 15
done
