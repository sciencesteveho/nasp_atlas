#!/bin/bash
#PBS -N nasp_target_evidence
#PBS -l select=1:ncpus=2:mem=32gb
#PBS -l walltime=02:00:00
#PBS -m a
#PBS -j oe
#PBS -o /rds/general/user/sho3/projects/lms-scott-raw/live/steve/tabula_sapiens/scripts/job_out/

# Combine completed condition-cohort runs into sensor target evidence, tiers,
# the evidence-matrix figure and report.md. Nothing is refitted.
# Both current cohorts (run paths are relative to the cohort output root):
#   qsub -v RUNS=kpmp_sn_v2/living_biopsy:sle_perez/female target_evidence.sh
#
# Preview the worker command without submitting work:
#   DRY_RUN=1 RUNS=kpmp_sn_v2/living_biopsy:sle_perez/female bash target_evidence.sh

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

# qsub -v parameters
RUNS="${RUNS:-}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/results/condition_cohorts}"
OUTPUT_NAME="${OUTPUT_NAME:-target_evidence}"
ANNOTATION_PANEL="${ANNOTATION_PANEL:-}"
ALPHA="${ALPHA:-0.05}"
SIRNA_DETECTION="${SIRNA_DETECTION:-0.10}"
CRISPRI_DETECTION="${CRISPRI_DETECTION:-0.25}"
DONOR_FRACTION="${DONOR_FRACTION:-0.75}"
DRY_RUN="${DRY_RUN:-0}"

# PBS logging
PBS_COMMON="${HOME}/pbs_common.sh"
LOG_DIR="${PROJECT_DIR}/scripts/job_out"
JOB_NAME="${PBS_JOBNAME:-nasp_target_evidence}"
JOB_ID="${PBS_JOBID:-local}"
LIVE_LOG="${LOG_DIR}/${JOB_NAME}.${JOB_ID}.live.log"

#######################################
# Validate job settings and summarize target evidence across runs.
# Outputs:
#   Writes resolved job settings and analysis output to stdout and stderr.
# Returns:
#   0 on success; 2 for invalid job settings; otherwise the failing command's
#   exit status.
#######################################
main() {
  local preview=0
  local run_names=()
  local run_name=""
  local python_args=()

  if [[ -z "${RUNS}" ]]; then
    printf 'ERROR: submit with RUNS=<cohort>/<population>[:...]\n' >&2
    return 2
  fi
  case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) preview=1 ;;
    0|false|FALSE|no|NO) ;;
    *) printf 'ERROR: DRY_RUN must be boolean\n' >&2; return 2 ;;
  esac
  if (( ! preview )) && [[ ! -f "${PBS_COMMON}" ]]; then
    printf 'ERROR: PBS helper does not exist: %s\n' "${PBS_COMMON}" >&2
    return 2
  fi

  python_args=(--runs)
  IFS=':' read -r -a run_names <<< "${RUNS}"
  for run_name in "${run_names[@]}"; do
    if (( ! preview )) && [[ ! -d "${RUN_ROOT}/${run_name}" ]]; then
      printf 'ERROR: run does not exist: %s\n' "${RUN_ROOT}/${run_name}" >&2
      return 2
    fi
    python_args+=("${RUN_ROOT}/${run_name}")
  done
  python_args+=(
    --output-path "${RUN_ROOT}/${OUTPUT_NAME}"
    --alpha "${ALPHA}"
    --sirna-detection "${SIRNA_DETECTION}"
    --crispri-detection "${CRISPRI_DETECTION}"
    --donor-fraction "${DONOR_FRACTION}"
  )
  if [[ -n "${ANNOTATION_PANEL}" ]]; then
    python_args+=(--annotation-panel "${ANNOTATION_PANEL}")
  fi

  printf 'Worker command: '
  printf '%q ' python -u "${REPO_DIR}/run_scripts/target_evidence.py" "${python_args[@]}"
  printf '\n'
  if (( preview )); then
    return 0
  fi

  mkdir -p "${LOG_DIR}"
  source "${PBS_COMMON}"
  pbs::setup_logging "${LIVE_LOG}"
  pbs::activate_conda_environment "${CONDA_BASE}" "${CONDA_ENV}"
  pbs::log_job_context "${JOB_ID}" "${PROJECT_DIR}" "${LIVE_LOG}"
  export MPLCONFIGDIR="${TMPDIR:-/tmp}/${USER}/nasp_${JOB_ID}/matplotlib"
  mkdir -p "${MPLCONFIGDIR}"

  (
    cd "${REPO_DIR}"
    PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
      python -u "${REPO_DIR}/run_scripts/target_evidence.py" "${python_args[@]}"
  )

  printf 'Job finished: %s\n' "$(date -Is)"
}


main "$@"
