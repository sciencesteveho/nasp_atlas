#!/bin/bash
#PBS -N nasp_tissue
#PBS -l select=1:ncpus=16:mem=128gb
#PBS -l walltime=12:00:00
#PBS -m a
#PBS -j oe
#PBS -o /rds/general/user/sho3/projects/lms-scott-raw/live/steve/tabula_sapiens/scripts/job_out/

# Run NASP scoring and donor-aware analysis for one tissue h5ad.
#
# Submit an already split tissue h5ad using its existing embedding:
#   qsub -v H5AD_NAME=liver_tabula_sapiens.h5ad,RUN_NAME=liver score_modules.sh
#
# Subset a larger h5ad and recompute the tissue UMAP using X_scvi:
#   qsub -v H5AD_NAME=all.h5ad,TISSUE_LABEL=Liver,RUN_NAME=liver score_modules.sh
#
# Resume after scoring, or run only one scorer:
#   qsub -v H5AD_NAME=liver.h5ad,RUN_NAME=liver,RESUME=1 score_modules.sh
#   qsub -v H5AD_NAME=liver.h5ad,RUN_NAME=liver,SCORERS=scanpy score_modules.sh

set -o errexit
set -o nounset
set -o pipefail

# Conda environment
CONDA_ROOT="/rds/general/user/sho3/projects/lms-scott-raw/live/steve/envs/base"
CONDA_ENV="${CONDA_ROOT}/envs/nasp_atlas"

# Project layout
PROJECT_DIR="/rds/general/user/sho3/projects/lms-scott-raw/live/steve/tabula_sapiens"
REPO_DIR="/rds/general/user/sho3/projects/lms-scott-raw/live/steve/nasp_atlas"
DATA_DIR="${PROJECT_DIR}/data"
OUT_DIR="${PROJECT_DIR}/results/nasp_tissue_analysis"

# qsub -v parameters
H5AD_NAME="${H5AD_NAME:-}"
TISSUE_LABEL="${TISSUE_LABEL:-${SINGLE_TISSUE:-}}"
RUN_NAME="${RUN_NAME:-}"
SCORERS="${SCORERS:-scanpy:aucell}"
RESUME="${RESUME:-0}"
PLOT_MODULES="${PLOT_MODULES:-1}"
PLOT_NASP_VISUALIZATIONS="${PLOT_NASP_VISUALIZATIONS:-1}"
EXPRESSION_LAYER="${EXPRESSION_LAYER:-}"
SINGLE_TISSUE_USE_REP="${SINGLE_TISSUE_USE_REP:-X_scvi}"
STATISTICAL_UNIT="${STATISTICAL_UNIT:-donor}"
AGGREGATION="${AGGREGATION:-mean}"
MAX_PLOTS="${MAX_PLOTS:-200}"
RANDOM_STATE="${RANDOM_STATE:-42}"
AUCELL_CHUNK_SIZE="${AUCELL_CHUNK_SIZE:-1000}"
AUCELL_NUM_WORKERS="${AUCELL_NUM_WORKERS:-8}"

# PBS logging
PBS_COMMON="${HOME}/pbs_common.sh"
LOG_DIR="${PROJECT_DIR}/scripts/job_out"
JOB_NAME="${PBS_JOBNAME:-nasp_tissue}"
JOB_ID="${PBS_JOBID:-local}"
LIVE_LOG="${LOG_DIR}/${JOB_NAME}.${JOB_ID}.live.log"


# Resolve an h5ad filename relative to DATA_DIR, or retain an absolute path.
resolve_h5ad_path() {
  local h5ad_name="$1"
  local data_dir="$2"

  if [[ "${h5ad_name}" = /* ]]; then
    printf '%s\n' "${h5ad_name}"
    return
  fi
  printf '%s/%s\n' "${data_dir}" "${h5ad_name}"
}


# Run the Python tissue orchestrator with PBS-configured options.
run_analysis() {
  local h5ad_path="$1"
  local output_dir="$2"
  local tissue_label="$3"
  local run_name="$4"
  local scorer_text="$5"
  local python_args=()
  local scorer_values=()

  python_args=(
    --h5ad-path "${h5ad_path}"
    --output-path "${output_dir}"
    --random-state "${RANDOM_STATE}"
    --aucell-chunk-size "${AUCELL_CHUNK_SIZE}"
    --aucell-num-workers "${AUCELL_NUM_WORKERS}"
    --statistical-unit "${STATISTICAL_UNIT}"
    --aggregation "${AGGREGATION}"
    --max-plots "${MAX_PLOTS}"
  )

  IFS=':' read -r -a scorer_values <<< "${scorer_text}"
  python_args+=(--scorers "${scorer_values[@]}")
  if [[ -n "${tissue_label}" ]]; then
    python_args+=(--tissue-label "${tissue_label}")
  fi
  if [[ -n "${run_name}" ]]; then
    python_args+=(--run-name "${run_name}")
  fi
  if [[ -n "${EXPRESSION_LAYER}" ]]; then
    python_args+=(--expression-layer "${EXPRESSION_LAYER}")
  fi
  case "${SINGLE_TISSUE_USE_REP}" in
    none|None|NONE) python_args+=(--single-tissue-use-x) ;;
    *) python_args+=(--single-tissue-use-rep "${SINGLE_TISSUE_USE_REP}") ;;
  esac

  case "${RESUME}" in
    1|true|TRUE|yes|YES) python_args+=(--resume) ;;
    0|false|FALSE|no|NO) ;;
    *)
      printf 'ERROR: RESUME must be 0/1 or false/true, got %s\n' "${RESUME}" >&2
      return 2
      ;;
  esac
  case "${PLOT_MODULES}" in
    1|true|TRUE|yes|YES) python_args+=(--plot-modules) ;;
    0|false|FALSE|no|NO) python_args+=(--no-plot-modules) ;;
    *)
      printf 'ERROR: PLOT_MODULES must be 0/1 or false/true, got %s\n' \
        "${PLOT_MODULES}" >&2
      return 2
      ;;
  esac
  case "${PLOT_NASP_VISUALIZATIONS}" in
    1|true|TRUE|yes|YES) python_args+=(--nasp-visualizations) ;;
    0|false|FALSE|no|NO) python_args+=(--no-nasp-visualizations) ;;
    *)
      printf 'ERROR: PLOT_NASP_VISUALIZATIONS must be boolean, got %s\n' \
        "${PLOT_NASP_VISUALIZATIONS}" >&2
      return 2
      ;;
  esac

  cd "${REPO_DIR}"
  export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
  python -u "${REPO_DIR}/run_scripts/score_modules.py" "${python_args[@]}"
}


main() {
  local h5ad_path=""
  local job_tmp=""

  if [[ -z "${H5AD_NAME}" ]]; then
    printf 'ERROR: submit with H5AD_NAME=<tissue.h5ad>\n' >&2
    exit 2
  fi
  if [[ ! -f "${PBS_COMMON}" ]]; then
    printf 'ERROR: PBS helper does not exist: %s\n' "${PBS_COMMON}" >&2
    exit 2
  fi
  h5ad_path="$(resolve_h5ad_path "${H5AD_NAME}" "${DATA_DIR}")"
  if [[ ! -f "${h5ad_path}" ]]; then
    printf 'ERROR: h5ad file does not exist: %s\n' "${h5ad_path}" >&2
    exit 2
  fi

  mkdir -p "${LOG_DIR}" "${OUT_DIR}"
  source "${PBS_COMMON}"
  pbs::setup_logging "${LIVE_LOG}"
  pbs::activate_conda_environment "${CONDA_ROOT}" "${CONDA_ENV}"
  pbs::log_job_context "${JOB_ID}" "${PROJECT_DIR}" "${LIVE_LOG}"

  job_tmp="${TMPDIR:-/tmp}/${USER}/nasp_${JOB_ID}"
  mkdir -p "${job_tmp}/matplotlib" "${job_tmp}/numba"
  export MPLCONFIGDIR="${job_tmp}/matplotlib"
  export NUMBA_CACHE_DIR="${job_tmp}/numba"
  export PYTHONUNBUFFERED=1

  printf 'Input h5ad: %s\n' "${h5ad_path}"
  printf 'Output root: %s\n' "${OUT_DIR}"
  printf 'Run name: %s\n' "${RUN_NAME:-auto}"
  printf 'Tissue label: %s\n' "${TISSUE_LABEL:-pre-split input}"
  printf 'Scorers: %s\n' "${SCORERS}"
  printf 'Resume: %s\n' "${RESUME}"

  run_analysis \
    "${h5ad_path}" \
    "${OUT_DIR}" \
    "${TISSUE_LABEL}" \
    "${RUN_NAME}" \
    "${SCORERS}"

  printf 'Job finished: %s\n' "$(date -Is)"
}


main "$@"
