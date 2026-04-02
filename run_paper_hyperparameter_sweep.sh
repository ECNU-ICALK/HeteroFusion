#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-infer_train}"
WORK_DIR="${WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

CONFIG_DIR="${CONFIG_DIR:-$WORK_DIR/configs/heterofusion/paper_experiments/sensitivity_alpha}"
CONFIG_PATTERN="${CONFIG_PATTERN:-mit_movie_alpha_*.yaml}"

LOG_ROOT="$WORK_DIR/logs"
RUN_NAME="paper_hyperparameter_sweep_$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="$LOG_ROOT/$RUN_NAME"

STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
SKIP_FINISHED="${SKIP_FINISHED:-1}"
DRY_RUN="${DRY_RUN:-0}"

# GPU_POOL and CUDA_VISIBLE_DEVICES are both treated as physical GPU ids.
# Example: CUDA_VISIBLE_DEVICES=0,3 => schedule on physical 0 and 3.
GPU_POOL="${GPU_POOL:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

declare -a AUTO_GPUS=()
declare -A GPU_LOAD=()

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

  cd "$WORK_DIR"
  mkdir -p "$RUN_LOG_DIR"

  echo "ENV: $CONDA_ENV_NAME"
  echo "WORK_DIR: $(pwd)"
  echo "RUN_LOG_DIR: $RUN_LOG_DIR"
}

init_gpu_pool() {
  local g

  if [[ -n "$GPU_POOL" ]]; then
    IFS=',' read -r -a AUTO_GPUS <<< "$GPU_POOL"
  elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _VISIBLE <<< "$CUDA_VISIBLE_DEVICES"
    for g in "${_VISIBLE[@]}"; do
      g="${g//[[:space:]]/}"
      [[ -z "$g" ]] && continue
      AUTO_GPUS+=("$g")
    done
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "Using physical GPUs: ${AUTO_GPUS[*]}"
  else
    AUTO_GPUS=("0")
  fi

  local -a cleaned=()
  for g in "${AUTO_GPUS[@]}"; do
    g="${g//[[:space:]]/}"
    [[ -z "$g" ]] && continue
    cleaned+=("$g")
  done
  AUTO_GPUS=("${cleaned[@]}")

  if [[ ${#AUTO_GPUS[@]} -eq 0 ]]; then
    AUTO_GPUS=("0")
  fi

  for g in "${AUTO_GPUS[@]}"; do
    GPU_LOAD["$g"]=0
  done

  echo "GPU_POOL_RESOLVED=${AUTO_GPUS[*]}"
  echo "WORKERS_PER_GPU=$WORKERS_PER_GPU"
}

collect_configs() {
  if [[ ! -d "$CONFIG_DIR" ]]; then
    echo "ERROR: missing config dir: $CONFIG_DIR"
    exit 1
  fi

  mapfile -t CONFIG_LIST < <(find "$CONFIG_DIR" -maxdepth 1 -type f -name "$CONFIG_PATTERN" | sort)
  if [[ ${#CONFIG_LIST[@]} -eq 0 ]]; then
    echo "ERROR: no yaml configs found for pattern: $CONFIG_PATTERN"
    exit 1
  fi
}

pick_least_loaded_gpu() {
  local chosen="" chosen_load=999999
  local g load
  for g in "${AUTO_GPUS[@]}"; do
    load="${GPU_LOAD[$g]:-0}"
    if [[ -z "$chosen" || "$load" -lt "$chosen_load" ]]; then
      chosen="$g"
      chosen_load="$load"
    fi
  done
  echo "$chosen"
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

  local cfg_name output_dir abs_output done_marker log_file
  cfg_name="$(basename "$cfg" .yaml)"

  output_dir="$(get_output_dir_from_yaml "$cfg")"
  if [[ -z "$output_dir" ]]; then
    echo "[$queue_id:$idx][GPU $gpu] ERROR: cannot parse output_dir in $cfg"
    return 1
  fi
  abs_output="$(abs_path_from_config_output_dir "$output_dir")"
  done_marker="$(find "$abs_output" -path '*/merged_lora/adapter_model.safetensors' -print -quit 2>/dev/null || true)"
  log_file="$RUN_LOG_DIR/gpu${gpu}_$(printf '%02d' "$idx")_${cfg_name}.log"

  echo "[$queue_id:$idx][GPU $gpu] config=$(basename "$cfg")"
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

split_queue_file() {
  local src_list="$1"
  local workers="$2"

  if (( workers <= 1 )); then
    echo "$src_list"
    return
  fi

  local base out i cfg n idx
  base="${src_list%.list}"
  local -a outs=()

  for ((i = 1; i <= workers; i++)); do
    out="${base}_w${i}.list"
    : > "$out"
    outs+=("$out")
  done

  n=0
  while IFS= read -r cfg; do
    [[ -z "$cfg" ]] && continue
    idx=$((n % workers))
    echo "$cfg" >> "${outs[$idx]}"
    n=$((n + 1))
  done < "$src_list"

  printf '%s\n' "${outs[@]}"
}

worker() {
  local gpu="$1"
  local list_file="$2"

  local queue_tag queue_id summary_file worker_log
  queue_tag="$(basename "$list_file" .list)"
  queue_id="$queue_tag"
  summary_file="$RUN_LOG_DIR/summary_${queue_tag}.txt"
  worker_log="$RUN_LOG_DIR/worker_${queue_tag}.log"

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
  for f in "$RUN_LOG_DIR"/summary_*.txt; do
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
  init_gpu_pool
  collect_configs

  print_header "Build GPU Queues"
  echo "config_count=${#CONFIG_LIST[@]}"
  echo "STOP_ON_ERROR=$STOP_ON_ERROR SKIP_FINISHED=$SKIP_FINISHED DRY_RUN=$DRY_RUN"

  local queue_dir="$RUN_LOG_DIR/queues"
  mkdir -p "$queue_dir"

  local cfg gpu qf
  for cfg in "${CONFIG_LIST[@]}"; do
    gpu="$(pick_least_loaded_gpu)"
    GPU_LOAD["$gpu"]=$((GPU_LOAD[$gpu] + 1))
    qf="$queue_dir/gpu${gpu}.list"
    echo "$cfg" >> "$qf"
    echo "assign: $(basename "$cfg") -> GPU $gpu (load=${GPU_LOAD[$gpu]})"
  done

  print_header "Start Workers"
  local qfile shard gpu pid
  local -a pids=()

  for qfile in "$queue_dir"/gpu*.list; do
    [[ -f "$qfile" ]] || continue
    gpu="$(basename "$qfile" .list)"
    gpu="${gpu#gpu}"

    mapfile -t SHARDS < <(split_queue_file "$qfile" "$WORKERS_PER_GPU")
    for shard in "${SHARDS[@]}"; do
      [[ -s "$shard" ]] || continue
      worker "$gpu" "$shard" &
      pid=$!
      pids+=("$pid")
      echo "worker started: GPU $gpu queue=$(basename "$shard") pid=$pid"
    done
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
