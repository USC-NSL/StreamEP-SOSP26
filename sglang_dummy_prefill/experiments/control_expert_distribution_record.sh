#!/bin/bash
set -euo pipefail

ACTION=$1
HOST=${2:-127.0.0.1}
PORT=${3:-30000}

case "$ACTION" in
  start|stop|dump) ;;
  *)
    echo "Usage: $0 <start|stop|dump> [host] [port]" >&2
    exit 1
    ;;
esac

curl -fsS -X POST "http://${HOST}:${PORT}/${ACTION}_expert_distribution_record"
echo
