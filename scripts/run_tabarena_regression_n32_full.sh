#!/usr/bin/env bash
#
# Reproducible 32-estimator TabArena regression comparison.
#
# Usage:
#
#   ./scripts/run_tabarena_regression_n32_full.sh DATASET [RESULTS_DIR]
#
# DATASET may be a TabArena regression dataset name or an OpenML dataset ID.
# For example:
#
#   ./scripts/run_tabarena_regression_n32_full.sh airfoil_self_noise
#   ./scripts/run_tabarena_regression_n32_full.sh 46904
#   ./scripts/run_tabarena_regression_n32_full.sh \
#     concrete_compressive_strength \
#     ttt_results/concrete_n32_full
#
# This experiment runs the four configurations needed to separate the effects
# of test-time training (TTT) and engineered-feature ensembling:
#
#   default:
#     Raw features, no TTT.
#   default_ttt:
#     Raw-feature TTT, followed by raw-feature inference.
#   ensemble:
#     No TTT; raw, feature-cross, and SVD ensemble inference with NNLS weights.
#   ensemble_ttt:
#     Each of 32 members independently performs TTT using raw features only.
#     NNLS weights stay the plain ensemble's, fitted from its own
#     leakage-free out-of-fold predictions.
#
# The command intentionally spells out the experiment settings rather than
# relying on the Python runner's defaults. Negative CLI switches are omitted
# deliberately so that AMP, bfloat16 base-model casting, NNLS, and prediction
# saving remain enabled.

set -Eeuo pipefail

readonly SCRIPT_DIR="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
  pwd
)"
readonly REPO_ROOT="$(
  cd -- "${SCRIPT_DIR}/.."
  pwd
)"
readonly PYTHON="${REPO_ROOT}/.venv/bin/python"
readonly RUNNER="${REPO_ROOT}/scripts/run_tabarena_regression_pytorch_ensemble_ttt.py"

readonly OPENML_SUITE="tabarena-v0.1"
readonly OPENML_CACHE_DIR="${OPENML_CACHE_DIR:-${HOME}/.cache/openml}"

usage() {
  echo "Usage: $0 DATASET [RESULTS_DIR]" >&2
  echo >&2
  echo "DATASET may be a TabArena regression name or OpenML dataset ID." >&2
}

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  usage
  exit 2
fi

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

readonly DATASET="$1"
readonly DATASET_SLUG="${DATASET//[^[:alnum:]]/_}"
readonly RESULTS_DIR="${2:-${REPO_ROOT}/ttt_results/${DATASET_SLUG}_n32_full}"

if [[ -z "${DATASET_SLUG}" ]]; then
  echo "DATASET must contain at least one letter or digit." >&2
  exit 2
fi

readonly DEVICE="cuda"
readonly N_ESTIMATORS="32"
readonly BATCH_SIZE="1"
readonly SEED="42"
readonly REPEAT="0"
readonly FOLD="0"

readonly TTT_LORA_RANK="8"
readonly TTT_TARGET_LAYERS="cell_embedder.in_linear"
readonly TTT_STEPS="8"
# Independent context/query draws are forwarded together as one batch, which
# both cuts gradient noise and keeps the GPU busy. Activation memory scales
# with TTT_BATCH_SIZE; on a 40GB card 4 is the practical ceiling for ~1000
# training rows. Override both together for memory-constrained datasets:
# TTT_BATCH_SIZE=1 TTT_GRAD_ACCUM=4 ./scripts/run_tabarena_regression_n32_full.sh ...
readonly TTT_BATCH_SIZE="${TTT_BATCH_SIZE:-4}"
readonly TTT_GRAD_ACCUM="${TTT_GRAD_ACCUM:-1}"
readonly TTT_LEARNING_RATE="1e-3"
readonly TTT_TEST_FRACTION="0.2"

readonly -a METHODS=(
  "default"
  "default_ttt"
  "ensemble"
  "ensemble_ttt"
)

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing executable virtual-environment Python: ${PYTHON}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "Missing regression experiment runner: ${RUNNER}" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; this experiment requires a CUDA GPU." >&2
  exit 1
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | grep -q .; then
  echo "No CUDA GPU was detected by nvidia-smi." >&2
  exit 1
fi

mkdir -p -- "${RESULTS_DIR}" "${OPENML_CACHE_DIR}"
cd -- "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="1"

command_args=(
  "${PYTHON}"
  "${RUNNER}"
  "--dataset" "${DATASET}"
  "--results-dir" "${RESULTS_DIR}"
  "--openml-cache-dir" "${OPENML_CACHE_DIR}"
  "--openml-suite" "${OPENML_SUITE}"
  "--repeat" "${REPEAT}"
  "--fold" "${FOLD}"
  "--device" "${DEVICE}"
  "--n-estimators" "${N_ESTIMATORS}"
  "--batch-size" "${BATCH_SIZE}"
  "--seed" "${SEED}"
  "--ttt-lora-rank" "${TTT_LORA_RANK}"
  "--ttt-target-layers" "${TTT_TARGET_LAYERS}"
  "--ttt-steps" "${TTT_STEPS}"
  "--ttt-batch-size" "${TTT_BATCH_SIZE}"
  "--ttt-gradient-accumulation-steps" "${TTT_GRAD_ACCUM}"
  "--ttt-learning-rate" "${TTT_LEARNING_RATE}"
  "--ttt-test-fraction" "${TTT_TEST_FRACTION}"
  "--ttt-use-all-rows"
  "--log-file"
  "--fail-fast"
)

for method in "${METHODS[@]}"; do
  command_args+=("--method" "${method}")
done

echo "Starting the pinned TabFM regression experiment:"
echo "  dataset:                  ${DATASET}"
echo "  methods:                  ${METHODS[*]}"
echo "  ensemble members:         ${N_ESTIMATORS}"
echo "  TTT target layers:        ${TTT_TARGET_LAYERS}"
echo "  TTT LoRA rank:            ${TTT_LORA_RANK}"
echo "  TTT steps:                ${TTT_STEPS}"
echo "  TTT batch size:           ${TTT_BATCH_SIZE}"
echo "  TTT grad accumulation:    ${TTT_GRAD_ACCUM}"
echo "  TTT row policy:           all rows"
echo "  TTT checkpointing:        off (chunking left at model defaults)"
echo "  AMP:                      enabled"
echo "  bfloat16 frozen model:    enabled"
echo "  float32 LoRA adapters:    enabled"
echo "  ensemble_ttt NNLS:        baseline weights (not refit)"
echo "  prediction saving:        enabled"
echo "  repeat/fold/seed:         ${REPEAT}/${FOLD}/${SEED}"
echo "  results directory:        ${RESULTS_DIR}"
echo

printf 'Command:'
printf ' %q' "${command_args[@]}"
printf '\n\n'

"${command_args[@]}"

echo
echo "Experiment finished successfully."
echo "Results are under: ${RESULTS_DIR}"
