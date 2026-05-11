#!/usr/bin/env bash
#
# rebench-full.sh — canonical 5-step rebench against the currently-running
# model. Built to eliminate the recurring mistakes from manual runs:
#
#   - Wrong cwd (`scripts/X.sh: No such file or directory`)
#   - Forgot `--save-json` on benchlocal-cli direct invocations
#   - Forgot `MODEL=` override → HTTP 404 from served-model-name mismatch
#   - Forgot `BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1` for localhost URLs
#   - Wrong port (8010 / 8011 / 8030 / 8032 — easy typo)
#   - No idempotent resume — every interrupt redoes the whole matrix
#
# Order matches docs/QUALITY_TEST.md "test pipeline":
#   1. bench.sh                — TPS narrative + code (~5 min)
#   2. verify-stress.sh        — long-context + boundary (~10-15 min)
#   3. quality-test.sh --full  — 8 packs, 150 scenarios (~45-60 min)
#   4. soak-test.sh fresh-mode — stability over 100 turns (~30-40 min)
#   5. quality-test.sh --pack aider-polyglot-30  (~20-45 min)
#
# Total per leg: ~2-2.5 hr.
#
# All artifacts land in results/rebench/<tag>/. Run twice on different models
# (e.g. one Qwen leg, one Gemma leg) to assemble a matched-config head-to-head.
#
# Usage:
#   bash scripts/rebench-full.sh                      # auto-tag from MODEL
#   bash scripts/rebench-full.sh --tag qwen-int8      # explicit tag
#   bash scripts/rebench-full.sh --skip soak,aider    # skip phases (CSV)
#   bash scripts/rebench-full.sh --resume             # skip steps that have
#                                                       artifacts already
#
# Env overrides (rarely needed — preflight auto-detects):
#   URL                 endpoint (default: auto-detect from running container)
#   MODEL               served-model-name (default: GET /v1/models)
#   TAG                 output-dir basename (default: ${MODEL}-YYYYMMDD-HHMM)
#   OUT_DIR             override the output directory
#   SOAK_SESSIONS       passed through to soak-test.sh (default: 20)
#   SOAK_TURNS          passed through to soak-test.sh (default: 5)
#

set -euo pipefail

# --- canonical cwd ----------------------------------------------------------
# This is THE fix for the recurring `scripts/X.sh: No such file or directory`
# bug — always resolve to the repo root regardless of where the user invokes
# from.
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- args -------------------------------------------------------------------
SKIP_CSV=""
RESUME=0
TAG_OVERRIDE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip)     SKIP_CSV="$2"; shift 2 ;;
    --tag)      TAG_OVERRIDE="$2"; shift 2 ;;
    --resume)   RESUME=1; shift ;;
    -h|--help)
      sed -n '2,40p' "$0"
      exit 0
      ;;
    *)
      echo "✗ unknown arg: $1 (see --help)" >&2
      exit 2
      ;;
  esac
done

skip_step() {
  IFS=',' read -ra SKIPS <<< "$SKIP_CSV"
  for s in "${SKIPS[@]}"; do [[ "$s" == "$1" ]] && return 0; done
  return 1
}

# --- endpoint + model auto-detect ------------------------------------------
# Source preflight if available; it sets URL + CONTAINER from running compose.
if [[ -f "$ROOT_DIR/scripts/preflight.sh" ]]; then
  # shellcheck source=preflight.sh
  source "$ROOT_DIR/scripts/preflight.sh"
  preflight_autodetect_endpoint
fi
URL="${URL:-http://localhost:8010}"

if ! curl -sf -m 5 "$URL/v1/models" >/dev/null 2>&1; then
  echo "✗ endpoint $URL/v1/models not responding" >&2
  echo "  start a compose first: gpu-mode <mode>" >&2
  exit 1
fi

# Resolve actual served model id — eliminates MODEL=qwen vs MODEL=gemma
# typos that produce HTTP 404 from served-model-name mismatch.
DETECTED_MODEL=$(curl -sf -m 5 "$URL/v1/models" 2>/dev/null \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null \
  || echo "")
if [[ -n "$DETECTED_MODEL" && -z "${MODEL:-}" ]]; then
  MODEL="$DETECTED_MODEL"
fi
if [[ -z "${MODEL:-}" ]]; then
  echo "✗ could not detect served model. Set MODEL=<name> explicitly." >&2
  exit 1
fi

# --- output dir -------------------------------------------------------------
TAG="${TAG_OVERRIDE:-${TAG:-${MODEL//[^a-z0-9._-]/-}-$(date +%Y%m%d-%H%M)}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/results/rebench/$TAG}"
mkdir -p "$OUT_DIR"

# --- env that benchlocal-cli + quality-test need ---------------------------
# Auto-set the localhost resolve flag so hermes sandbox can reach host vLLM.
# Idempotent — don't overwrite an explicit user value.
if [[ -z "${BENCHLOCAL_HERMES_RESOLVE_LOCALHOST:-}" ]] \
   && [[ "$URL" =~ ^https?://(localhost|127\.|\[::1\]) ]]; then
  export BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=1
fi

# --- preamble -----------------------------------------------------------------
echo "==============================================================="
echo " rebench-full.sh"
echo "==============================================================="
echo "  endpoint:    $URL"
echo "  model:       $MODEL"
echo "  out dir:     $OUT_DIR"
echo "  resume:      $RESUME"
echo "  skips:       ${SKIP_CSV:-(none)}"
echo "  hermes env:  BENCHLOCAL_HERMES_RESOLVE_LOCALHOST=${BENCHLOCAL_HERMES_RESOLVE_LOCALHOST:-0}"
echo "==============================================================="
date +"  started:     %Y-%m-%dT%H:%M:%SZ" -u
echo

# --- helpers ----------------------------------------------------------------
have_artifact() { [[ -s "$1" ]]; }

run_step() {
  local name="$1" artifact="$2"
  shift 2
  if skip_step "$name"; then
    echo "[$name] skipped (--skip)"
    return 0
  fi
  if [[ "$RESUME" == "1" ]] && have_artifact "$artifact"; then
    echo "[$name] skipped (--resume found $artifact)"
    return 0
  fi
  echo "[$name] running…"
  local t0=$(date +%s)
  if "$@" > "$OUT_DIR/$name.log" 2>&1; then
    local dt=$(( $(date +%s) - t0 ))
    echo "[$name] ✓ ${dt}s — log: $OUT_DIR/$name.log"
  else
    local rc=$? dt=$(( $(date +%s) - t0 ))
    echo "[$name] ✗ ${dt}s — failed (rc=$rc) — log: $OUT_DIR/$name.log" >&2
    return $rc
  fi
}

# copy the most recent quality.json from the shared results dir into our
# per-tag dir; that's where quality-test.sh writes its --save-json output.
snapshot_quality_json() {
  local target="$1"
  local src
  src="$(ls -t "$ROOT_DIR"/results/quality/quality-*.json 2>/dev/null | head -1)"
  if [[ -n "$src" ]]; then
    cp "$src" "$target"
  fi
}

# --- step 1: bench ----------------------------------------------------------
URL="$URL" MODEL="$MODEL" RUNS="${RUNS:-3}" WARMUPS="${WARMUPS:-1}" \
  run_step bench "$OUT_DIR/bench.log" \
    bash "$ROOT_DIR/scripts/bench.sh" || true

# --- step 2: verify-stress --------------------------------------------------
URL="$URL" MODEL="$MODEL" \
  run_step verify-stress "$OUT_DIR/verify-stress.log" \
    bash "$ROOT_DIR/scripts/verify-stress.sh" || true

# --- step 3: quality-test --full --------------------------------------------
URL="$URL" MODEL="$MODEL" \
  run_step quality-full "$OUT_DIR/quality-full.log" \
    bash "$ROOT_DIR/scripts/quality-test.sh" --full
snapshot_quality_json "$OUT_DIR/quality-full.json"

# --- step 4: soak-test ------------------------------------------------------
URL="$URL" MODEL="$MODEL" \
  SOAK_MODE="${SOAK_MODE:-fresh}" \
  SOAK_OUTPUT="$OUT_DIR/soak-artifacts" \
  SESSIONS="${SOAK_SESSIONS:-20}" \
  TURNS="${SOAK_TURNS:-5}" \
  run_step soak "$OUT_DIR/soak.log" \
    bash "$ROOT_DIR/scripts/soak-test.sh"

# --- step 5: aider-polyglot-30 ----------------------------------------------
URL="$URL" MODEL="$MODEL" \
  run_step aider-polyglot "$OUT_DIR/aider-polyglot.log" \
    bash "$ROOT_DIR/scripts/quality-test.sh" --pack aider-polyglot-30
snapshot_quality_json "$OUT_DIR/aider-polyglot.json"

# --- summary ----------------------------------------------------------------
echo
echo "==============================================================="
echo " rebench complete"
echo "==============================================================="
date +"  finished:    %Y-%m-%dT%H:%M:%SZ" -u
echo "  artifacts:   $OUT_DIR"
echo
echo "Headline pulls (grep through the logs):"
echo "  TPS:           grep -E 'mean=|decode_TPS' $OUT_DIR/bench.log"
echo "  verify-stress: tail -5 $OUT_DIR/verify-stress.log"
echo "  quality:       grep '^Quality:' $OUT_DIR/quality-full.log"
echo "  soak:          grep -E 'verdict|silent_empty|p50_decode' $OUT_DIR/soak.log"
echo "  aider:         grep 'aider-polyglot-30' $OUT_DIR/aider-polyglot.log"
