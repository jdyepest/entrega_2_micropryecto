#!/bin/sh
set -e

MODEL="${OLLAMA_MODEL:-llama3.2:1b}"

# Start Ollama server in background
ollama serve &
PID=$!

# Wait for server to be ready
for i in $(seq 1 30); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# Pre-pull model only if missing (best-effort)
if [ -n "$MODEL" ]; then
  if ! ollama list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
    ollama pull "$MODEL" || true
  fi
fi

# Keep server in foreground
wait $PID
