#!/usr/bin/env bash
# force_kill_everything.sh — ONE command to nuke every experiment process on every node
# and verify the cluster is actually clean before returning.
#
#   bash experiment_utils/force_kill_everything.sh
#
# Kills, on the head and all workers:
#   - sweep orchestrators (run_head_*.sh / run_worker_*.sh) FIRST, so nothing relaunches
#     mid-cleanup (a killed server otherwise gets respawned by the surviving script);
#   - sglang: launch_server + the setproctitle-renamed "sglang::*" workers + srt +
#     bench_serving + torch inductor compile workers;
#   - StreamInfer: the frontend (benchmark/server.py — holds no GPU, matches neither
#     "disagmoe" nor "ray"), Ray (stop --force), and any process still holding GPU
#     memory (leftover Ray-actor engines survive `ray stop`);
#   - interference generator: ucx_sender / ucx_receiver;
#   - stale port holders: sglang dist/IPC ports + HTTP ports.
# Then waits until every node reports 0 matching processes, 0 MiB GPU memory, and the
# head's ports are free.  Bracket patterns ([s]glang) keep pkill from matching itself
# or its own ssh command line.
set -uo pipefail
HOSTS="${HOSTS:-sgpu6 sgpu7 sgpu8 sgpu9}"
PORTS="${PORTS:-25000 25001 25002 25003 25004 25005 30000 6699}"
LOCAL="$(hostname -s)"
log(){ echo "$(date '+%H:%M:%S') [force-kill] $*"; }
_on(){ local h="$1"; shift; if [ "$h" = "$LOCAL" ]; then bash -c "$*"; else ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "$*"; fi; }

KILL='
pkill -9 -f "[r]un_head_sglang.sh";      pkill -9 -f "[r]un_head_streaminfer.sh"
pkill -9 -f "[r]un_worker_streaminfer.sh"
pkill -9 -f "[s]glang.launch_server";    pkill -9 -f "[s]glang::"
pkill -9 -f "[s]glang.srt";              pkill -9 -f "[s]glang.bench_serving"
pkill -9 -f "[t]orch._inductor.compile_worker"
pkill -9 -f "[b]enchmark/server.py"
pkill -9 -f "[u]cx_sender";              pkill -9 -f "[u]cx_receiver"
for RAY in "$HOME"/miniconda3/envs/*/bin/ray; do "$RAY" stop --force >/dev/null 2>&1; done
pkill -9 -f "[r]aylet";                  pkill -9 -f "[r]ay::"
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
for p in '"$PORTS"'; do fuser -k "${p}"/tcp >/dev/null 2>&1; done
true'

# ── SSH reachability preflight ────────────────────────────────────────────────
# A node we can't reach can't be cleaned — report it loudly up front (and again in
# the exit status) instead of silently skipping it.
UNREACHABLE=""
for h in $HOSTS; do
  [ "$h" = "$LOCAL" ] && continue
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$h" true >/dev/null 2>&1 \
    || { UNREACHABLE="$UNREACHABLE $h"; log "ERROR: $h UNREACHABLE via ssh — cannot clean it"; }
done
REACHABLE=""
for h in $HOSTS; do case " $UNREACHABLE " in *" $h "*) :;; *) REACHABLE="$REACHABLE $h";; esac; done

log "dispatching kills to:$REACHABLE"
for h in $REACHABLE; do _on "$h" "$KILL" 2>/dev/null || true; done

log "waiting for full teardown (procs, GPU memory, ports) ..."
PAT='[s]glang|[b]enchmark/server.py|[u]cx_|[r]aylet|[r]un_head_|[r]un_worker_'
deadline=$(( SECONDS + ${TEARDOWN_TIMEOUT:-300} ))
while :; do
  busy=""
  for h in $REACHABLE; do
    read -r p m <<< "$(_on "$h" 'echo "$(pgrep -cf "'"$PAT"'" || true) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)"' 2>/dev/null)"
    p="${p:-?}"; m="${m:-?}"
    { [ "$p" != "0" ] || [ "${m:-1}" -gt 10 ] 2>/dev/null; } && busy="$busy $h(p=$p,m=${m}MiB)"
  done
  if [ -z "$busy" ] && ! ss -tln 2>/dev/null | grep -qE ":(2500[0-5]|30000|6699) "; then
    if [ -n "$UNREACHABLE" ]; then
      log "reachable nodes CLEAN, but UNREACHABLE (state unknown):$UNREACHABLE"
      exit 2
    fi
    log "ALL CLEAN: 0 procs, 0 GPU mem on every node; head ports free."
    exit 0
  fi
  [ "$SECONDS" -ge "$deadline" ] && { log "WARNING: not fully clean after ${TEARDOWN_TIMEOUT:-300}s — still busy:${busy:-' (head ports)'}. Re-dispatching kills once ..."; for h in $HOSTS; do _on "$h" "$KILL" 2>/dev/null || true; done; deadline=$(( SECONDS + 120 )); TEARDOWN_TIMEOUT=already-extended; }
  [ "${TEARDOWN_TIMEOUT:-}" = "already-extended" ] && [ "$SECONDS" -ge "$deadline" ] && { log "FATAL: still busy:$busy"; exit 1; }
  sleep 5
done
