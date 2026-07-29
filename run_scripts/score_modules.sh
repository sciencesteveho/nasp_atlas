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

# Paths
ROOT="/rds/general/user/sho3/projects/lms-scott-raw/live/steve"

# Conda
CONDA_BASE="${ROOT}/software/conda/base"
CONDA_ENV="${ROOT}/software/envs/nasp_atlas"

# Project layout
PROJECT_DIR="${ROOT}/tabula_sapiens"
REPO_DIR="${ROOT}/nasp_atlas"
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

#######################################
# Resolve an h5ad filename against its data directory.
# Arguments:
#   $1: h5ad filename or absolute path.
#   $2: Data directory for relative filenames.
# Outputs:
#   Writes the resolved path to stdout.
# Returns:
#   0 on success.
#######################################
resolve_h5ad_path() {
  local h5ad_name="$1"
  local data_dir="$2"

  if [[ "${h5ad_name}" = /* ]]; then
    printf '%s\n' "${h5ad_name}"
    return
  fi
  printf '%s/%s\n' "${data_dir}" "${h5ad_name}"
}

#######################################
# Run the Python tissue orchestrator from the repository environment.
# Arguments:
#   $1: Repository directory.
#   $2: Python module search path.
#   Remaining arguments: Options passed to score_modules.py.
# Outputs:
#   Writes analysis output to stdout and stderr.
# Returns:
#   The exit status from score_modules.py.
#######################################
run_analysis() {
  local repo_dir="$1"
  local python_path="$2"
  shift 2

  (
    cd "${repo_dir}"
    PYTHONPATH="${python_path}" \
      python -u "${repo_dir}/run_scripts/score_modules.py" "$@"
  )
}

#######################################
# Validate job settings and orchestrate the tissue analysis.
# Outputs:
#   Writes resolved job settings and analysis output to stdout and stderr.
# Returns:
#   0 on success; 2 for invalid job settings; otherwise the failing command's
#   exit status.
#######################################
main() {
  local h5ad_path=""
  local job_tmp=""
  local python_args=()
  local python_path="${REPO_DIR}:${PYTHONPATH:-}"
  local scorer_values=()

  if [[ -z "${H5AD_NAME}" ]]; then
    printf 'ERROR: submit with H5AD_NAME=<tissue.h5ad>\n' >&2
    return 2
  fi
  if [[ ! -f "${PBS_COMMON}" ]]; then
    printf 'ERROR: PBS helper does not exist: %s\n' "${PBS_COMMON}" >&2
    return 2
  fi
  h5ad_path="$(resolve_h5ad_path "${H5AD_NAME}" "${DATA_DIR}")"
  if [[ ! -f "${h5ad_path}" ]]; then
    printf 'ERROR: h5ad file does not exist: %s\n' "${h5ad_path}" >&2
    return 2
  fi

  python_args=(
    --h5ad-path "${h5ad_path}"
    --output-path "${OUT_DIR}"
    --random-state "${RANDOM_STATE}"
    --aucell-chunk-size "${AUCELL_CHUNK_SIZE}"
    --aucell-num-workers "${AUCELL_NUM_WORKERS}"
    --statistical-unit "${STATISTICAL_UNIT}"
    --aggregation "${AGGREGATION}"
    --max-plots "${MAX_PLOTS}"
  )

  IFS=':' read -r -a scorer_values <<< "${SCORERS}"
  python_args+=(--scorers "${scorer_values[@]}")
  if [[ -n "${TISSUE_LABEL}" ]]; then
    python_args+=(--tissue-label "${TISSUE_LABEL}")
  fi
  if [[ -n "${RUN_NAME}" ]]; then
    python_args+=(--run-name "${RUN_NAME}")
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

  mkdir -p "${LOG_DIR}" "${OUT_DIR}"
  source "${PBS_COMMON}"
  pbs::setup_logging "${LIVE_LOG}"
  pbs::activate_conda_environment "${CONDA_BASE}" "${CONDA_ENV}"
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

  run_analysis "${REPO_DIR}" "${python_path}" "${python_args[@]}"

  printf 'Job finished: %s\n' "$(date -Is)"
}


main "$@"
