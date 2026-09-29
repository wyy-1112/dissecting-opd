#!/usr/bin/env bash
# Merge one OPD FSDP actor checkpoint into a Hugging Face directory.
set -euo pipefail
umask 077

RUN_DIR="${1:?usage: merge_checkpoint.sh <run_dir> <step> [target_dir]}"
STEP="${2:?usage: merge_checkpoint.sh <run_dir> <step> [target_dir]}"
TARGET="${3:-$RUN_DIR/merged_hf/step_${STEP}}"
ACTOR="$RUN_DIR/global_step_${STEP}/actor"
VERL_ROOT=${VERL_ROOT:?VERL_ROOT must point at the patched verl checkout}
COMPLETION_VALIDATOR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_checkpoint.py"

test -f "$VERL_ROOT/verl/__init__.py"
# Set ALLOW_INCOMPLETE_RUN=1 only to read an intermediate checkpoint of a run that
# later crashed; the validator still enforces every shard and metadata check.
# Set ALLOW_INTERMEDIATE_STEP=1 to read a mid-training checkpoint of a run that did
# finish; the run-level success markers stay enforced.
validator_flags=()
if [ "${ALLOW_INCOMPLETE_RUN:-0}" = "1" ]; then
  validator_flags+=(--allow-incomplete-run)
fi
if [ "${ALLOW_INTERMEDIATE_STEP:-0}" = "1" ]; then
  validator_flags+=(--allow-intermediate-step)
fi
python3 "$COMPLETION_VALIDATOR" "$RUN_DIR" --step "$STEP" "${validator_flags[@]}" >/dev/null

source_fingerprint="$(
  python3 - "$ACTOR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

actor = Path(sys.argv[1]).resolve()
fsdp = actor / "fsdp_config.json"
shards = sorted(actor.glob("model_world_size_*_rank_*.pt"))
payload = {
    "actor": str(actor),
    "fsdp_sha256": hashlib.sha256(fsdp.read_bytes()).hexdigest(),
    "model_shards": [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in shards
    ],
}
print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
PY
)"

mkdir -p "$(dirname "$TARGET")"
exec 9>"$TARGET.merge.lock"
flock 9

if python3 - "$TARGET" "$source_fingerprint" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
expected_source = json.loads(sys.argv[2])
marker = target / ".opd_merge_complete.json"
try:
    payload = json.loads(marker.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
if payload.get("source") != expected_source:
    raise SystemExit(1)
config = target / "config.json"
if not config.is_file() or config.stat().st_size <= 0:
    raise SystemExit(1)
index = target / "model.safetensors.index.json"
if index.is_file():
    try:
        weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    except (KeyError, TypeError, json.JSONDecodeError):
        raise SystemExit(1)
    files = [target / name for name in weights]
else:
    files = sorted(target.glob("*.safetensors"))
if not files or any(not path.is_file() or path.stat().st_size <= 0 for path in files):
    raise SystemExit(1)
PY
then
  echo "[reuse] $TARGET"
  exit 0
fi

if [ -e "$TARGET" ]; then
  mv "$TARGET" "$TARGET.incomplete.$(date -u +%Y%m%dT%H%M%SZ)"
fi
temporary="$TARGET.tmp.$$"
rm -rf "$temporary"
cleanup() {
  rm -rf "$temporary"
}
trap cleanup EXIT

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR" \
  --target_dir "$temporary"

python3 - "$temporary" "$source_fingerprint" <<'PY'
import json
import os
import sys
from pathlib import Path

target = Path(sys.argv[1])
source = json.loads(sys.argv[2])
config = target / "config.json"
if not config.is_file() or config.stat().st_size <= 0:
    raise SystemExit(f"merged config is missing: {config}")
index = target / "model.safetensors.index.json"
if index.is_file():
    weights = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    files = [target / name for name in weights]
else:
    files = sorted(target.glob("*.safetensors"))
if not files or any(not path.is_file() or path.stat().st_size <= 0 for path in files):
    raise SystemExit(f"merged weights are incomplete: {target}")
payload = {
    "schema_version": "opd_hf_merge_complete_v1",
    "source": source,
    "weights": [
        {"name": path.name, "size": path.stat().st_size}
        for path in files
    ],
}
marker = target / ".opd_merge_complete.json"
temporary_marker = marker.with_suffix(".json.tmp")
temporary_marker.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary_marker, marker)
PY

mv "$temporary" "$TARGET"
trap - EXIT
echo "[done] $TARGET"
