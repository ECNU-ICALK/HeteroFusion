#!/usr/bin/env bash

# ==============================================================================
# Paper single-source Qwen-to-LLaMA experiment runner
# - one worker per GPU
# - each worker runs its assigned configs sequentially
# - workers run in parallel
# ==============================================================================

set -euo pipefail

# -----------------------------
# 0) Base config
# -----------------------------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-infer_train}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

CONFIG_DIRS=(
  "$WORK_DIR/configs/heterofusion/paper_experiments/single_source_qwen_to_llama"
)

LOG_ROOT="$WORK_DIR/logs"
RUN_NAME="paper_single_source_qwen_to_llama_$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$RUN_NAME"

# 1=stop that GPU worker on first failure; 0=continue remaining configs on that GPU
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
# 1=skip config if final adapter already exists
SKIP_FINISHED="${SKIP_FINISHED:-1}"
# 1=plan only, do not execute training
DRY_RUN="${DRY_RUN:-0}"
# fallback GPU when no mapping found
DEFAULT_GPU="${DEFAULT_GPU:-0}"

# -----------------------------
# 1) GPU mapping priority
# -----------------------------
# Priority:
# (A) CONFIG_GPU_MAP (specific yaml)
# (B) GPU_MAP_FILE
# (C) DIR_GPU_MAP (folder default)
# (D) DEFAULT_GPU

declare -A DIR_GPU_MAP=(
  ["$WORK_DIR/configs/heterofusion/paper_experiments/single_source_qwen_to_llama"]="0"
)

declare -A CONFIG_GPU_MAP=(
  # ["llama-3.1-8b-instruct/01_baseNewYorkTimesRE.yaml"]="4"
  # ["Qwen2.5-7B-Instruct/01_baseNewYorkTimesRE.yaml"]="5"
)

GPU_MAP_FILE="${GPU_MAP_FILE:-}"
BASE_CONFIG_ROOT="$WORK_DIR/configs/heterofusion/paper_experiments/single_source_qwen_to_llama"

# -----------------------------
# 2) Helpers
# -----------------------------
print_header() {
  echo "============================================================"
  echo ">>> $1"
  echo "============================================================"
}

activate_env() {
  print_header "Initialize Environment"

  set +u
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
  set -u

  if [[ $? -ne 0 ]]; then
    echo "ERROR: failed to activate conda env: $CONDA_ENV_NAME"
    exit 1
  fi

  cd "$WORK_DIR"
  mkdir -p "$RUN_LOG_DIR"

  echo "ENV: $CONDA_ENV_NAME"
  echo "WORK_DIR: $(pwd)"
  echo "RUN_LOG_DIR: $RUN_LOG_DIR"
}

load_gpu_map_file() {
  if [[ -n "$GPU_MAP_FILE" && -f "$GPU_MAP_FILE" ]]; then
    while read -r rel gpu _; do
      [[ -z "${rel:-}" ]] && continue
      [[ "$rel" =~ ^# ]] && continue
      [[ -z "${gpu:-}" ]] && continue
      CONFIG_GPU_MAP["$rel"]="$gpu"
    done < "$GPU_MAP_FILE"
    echo "Loaded GPU map file: $GPU_MAP_FILE"
  fi
}

collect_configs() {
  local all=()
  local dir
  for dir in "${CONFIG_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
      echo "ERROR: missing config dir: $dir"
      exit 1
    fi
    mapfile -t local_cfgs < <(find "$dir" -maxdepth 1 -type f -name '*.yaml' | sort)
    all+=("${local_cfgs[@]}")
  done

  if [[ ${#all[@]} -eq 0 ]]; then
    echo "ERROR: no yaml configs found"
    exit 1
  fi
  printf '%s\n' "${all[@]}"
}

resolve_gpu_for_config() {
  local cfg="$1"
  local cfg_dir rel
  cfg_dir="$(dirname "$cfg")"
  rel="${cfg#$BASE_CONFIG_ROOT/}"

  if [[ -n "${CONFIG_GPU_MAP[$rel]+x}" ]]; then
    echo "${CONFIG_GPU_MAP[$rel]}|config"
    return
  fi
  if [[ -n "${DIR_GPU_MAP[$cfg_dir]+x}" ]]; then
    echo "${DIR_GPU_MAP[$cfg_dir]}|dir"
    return
  fi
  echo "${DEFAULT_GPU}|default"
}

get_output_dir_from_yaml() {
  local cfg="$1"
  awk '
    /^[[:space:]]*output_dir:[[:space:]]*/ {
      sub(/^[[:space:]]*output_dir:[[:space:]]*/, "", $0)
      sub(/[[:space:]]+#.*/, "", $0)
      gsub(/"/, "", $0)
      gsub(/\047/, "", $0)
      print $0
      exit
    }
  ' "$cfg"
}

abs_path_from_config_output_dir() {
  local out_dir="$1"
  if [[ "$out_dir" = /* ]]; then
    echo "$out_dir"
  else
    echo "$WORK_DIR/$out_dir"
  fi
}

run_one_config_on_gpu() {
  local cfg="$1"
  local idx="$2"
  local gpu="$3"
  local queue_id="$4"

  local rel cfg_name cfg_dir output_dir abs_output done_marker
  cfg_dir="$(dirname "$cfg")"
  cfg_name="$(basename "$cfg" .yaml)"
  rel="${cfg#$BASE_CONFIG_ROOT/}"

  output_dir="$(get_output_dir_from_yaml "$cfg")"
  if [[ -z "$output_dir" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] ERROR: cannot parse output_dir in $cfg"
    return 1
  fi
  abs_output="$(abs_path_from_config_output_dir "$output_dir")"
  done_marker="$(find "$abs_output" -path '*/merged_lora/adapter_model.safetensors' -print -quit 2>/dev/null || true)"

  local parent log_file
  parent="$(basename "$cfg_dir")"
  log_file="$RUN_LOG_DIR/gpu${gpu}_$(printf '%02d' "$idx")_${parent}_${cfg_name}.log"

  echo "[$queue_id:$idx][GPU $gpu] config=$rel"
  echo "[$queue_id:$idx][GPU $gpu] output_dir=$output_dir"
  echo "[$queue_id:$idx][GPU $gpu] log=$log_file"

  if [[ "$SKIP_FINISHED" == "1" && -f "$done_marker" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] SKIP: already finished"
    return 2
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] DRY_RUN=1"
    return 0
  fi

  local start_ts end_ts dur
  start_ts="$(date +%s)"
  if CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 python main.py --config_path "$cfg" > "$log_file" 2>&1; then
    end_ts="$(date +%s)"
    dur="$((end_ts - start_ts))"
    echo "[$queue_id:$idx][GPU $gpu] OK (${dur}s)"
    return 0
  else
    end_ts="$(date +%s)"
    dur="$((end_ts - start_ts))"
    echo "[$queue_id:$idx][GPU $gpu] FAIL (${dur}s)"
    return 1
  fi
}

worker() {
  local gpu="$1"
  local list_file="$2"

  local queue_id="W${gpu}"
  local summary_file="$RUN_LOG_DIR/summary_gpu${gpu}.txt"
  local worker_log="$RUN_LOG_DIR/worker_gpu${gpu}.log"

  local ok=0 skip=0 fail=0 idx=0

  {
    echo "[$queue_id] start worker on GPU $gpu"
    while IFS= read -r cfg; do
      [[ -z "$cfg" ]] && continue
      idx=$((idx + 1))
      if run_one_config_on_gpu "$cfg" "$idx" "$gpu" "$queue_id"; then
        ok=$((ok + 1))
      else
        rc=$?
        if [[ "$rc" -eq 2 ]]; then
          skip=$((skip + 1))
        else
          fail=$((fail + 1))
          if [[ "$STOP_ON_ERROR" == "1" ]]; then
            echo "[$queue_id] STOP_ON_ERROR=1, stop this worker"
            break
          fi
        fi
      fi
    done < "$list_file"

    echo "[$queue_id] done: ok=$ok skip=$skip fail=$fail"
  } > >(tee "$worker_log") 2>&1

  {
    echo "gpu=$gpu"
    echo "ok=$ok"
    echo "skip=$skip"
    echo "fail=$fail"
  } > "$summary_file"

  if [[ "$fail" -gt 0 ]]; then
    return 1
  fi
  return 0
}

aggregate_and_print_summary() {
  local total_ok=0 total_skip=0 total_fail=0
  local f
  for f in "$RUN_LOG_DIR"/summary_gpu*.txt; do
    [[ -f "$f" ]] || continue
    local ok skip fail
    ok="$(awk -F= '/^ok=/{print $2}' "$f")"
    skip="$(awk -F= '/^skip=/{print $2}' "$f")"
    fail="$(awk -F= '/^fail=/{print $2}' "$f")"
    total_ok=$((total_ok + ok))
    total_skip=$((total_skip + skip))
    total_fail=$((total_fail + fail))
  done

  print_header "Parallel Run Finished"
  echo "success: $total_ok"
  echo "skipped: $total_skip"
  echo "failed:  $total_fail"
  echo "run logs: $RUN_LOG_DIR"

  if [[ "$total_fail" -gt 0 ]]; then
    return 1
  fi
  return 0
}

main() {
  activate_env
  load_gpu_map_file

  mapfile -t CONFIG_LIST < <(collect_configs)
  print_header "Build GPU Queues"
  echo "config_count=${#CONFIG_LIST[@]}"
  echo "STOP_ON_ERROR=$STOP_ON_ERROR SKIP_FINISHED=$SKIP_FINISHED DRY_RUN=$DRY_RUN"

  local queue_dir="$RUN_LOG_DIR/queues"
  mkdir -p "$queue_dir"

  # build queue files by resolved gpu
  local cfg info gpu gpu_from rel qf
  for cfg in "${CONFIG_LIST[@]}"; do
    info="$(resolve_gpu_for_config "$cfg")"
    gpu="${info%%|*}"
    gpu_from="${info##*|}"
    rel="${cfg#$BASE_CONFIG_ROOT/}"
    qf="$queue_dir/gpu${gpu}.list"
    echo "$cfg" >> "$qf"
    echo "assign: $rel -> GPU $gpu (from $gpu_from)"
  done

  print_header "Start Workers"
  local qfile gpu pid
  local -a pids=()
  for qfile in "$queue_dir"/gpu*.list; do
    [[ -f "$qfile" ]] || continue
    gpu="$(basename "$qfile" .list)"
    gpu="${gpu#gpu}"
    worker "$gpu" "$qfile" &
    pid=$!
    pids+=("$pid")
    echo "worker started: GPU $gpu pid=$pid"
  done

  local any_fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      any_fail=1
    fi
  done

  if ! aggregate_and_print_summary; then
    exit 1
  fi

  if [[ "$any_fail" -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
