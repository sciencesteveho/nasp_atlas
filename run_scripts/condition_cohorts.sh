#!/bin/bash
#PBS -N nasp_condition_cohort
#PBS -l select=1:ncpus=8:mem=200gb
#PBS -l walltime=23:59:59
#PBS -m a
#PBS -j oe
#PBS -o /rds/general/user/sho3/projects/lms-scott-raw/live/steve/tabula_sapiens/scripts/job_out/

# Prepare and analyze one condition-cohort (A/B) population from its spec.
# Full lupus run:
#   qsub -v SPEC=sle_perez_female.yaml condition_cohorts.sh
#
# KPMP living-donor biopsies, Scanpy only:
#   qsub -v SPEC=kpmp_sn_v2_living_biopsy.yaml,SCORERS=scanpy condition_cohorts.sh
#
# Download and verify the input into ${PROJECT_DIR}/data first (the job
# needs outbound HTTPS; otherwise run fetch_cohort_input.py on a login node):
#   qsub -v SPEC=sle_perez_female.yaml,FETCH_INPUT=1 condition_cohorts.sh
#
# Lupus with the Tabula Sapiens blood calibration (AB8) after the run:
#   qsub -v SPEC=sle_perez_female.yaml,CALIBRATION=sle_perez_female_ts_blood_calibration.yaml condition_cohorts.sh
#
# Fast development subset (ten whole donors per arm):
#   qsub -v SPEC=sle_perez_female.yaml,DEVELOPMENT_DONORS_PER_ARM=10 condition_cohorts.sh
#
# Internal T2D_REM adipose object, read in place (audit first, then run):
#   T2D_DATA=/rds/general/user/sho3/projects/lms-scott-raw/live/T2D_REM/Phase_2/objects
#   qsub -v SPEC=t2d_rem_adipose_baseline.yaml,DATA_DIR="${T2D_DATA}",AUDIT_ONLY=1 condition_cohorts.sh
#   qsub -v SPEC=t2d_rem_adipose_baseline.yaml,DATA_DIR="${T2D_DATA}" condition_cohorts.sh
#   qsub -v SPEC=t2d_rem_adipose_weight_loss.yaml,DATA_DIR="${T2D_DATA}" condition_cohorts.sh
#   qsub -v SPEC=t2d_rem_adipose_remission.yaml,DATA_DIR="${T2D_DATA}" condition_cohorts.sh
#
# Rerun into a separate results root, e.g. after a compendium update:
#   qsub -v SPEC=sle_perez_female.yaml,OUT_DIR=/path/to/results/condition_cohorts_<commit> condition_cohorts.sh
#
# Preview the worker commands without submitting work:
#   DRY_RUN=1 SPEC=sle_perez_female.yaml bash condition_cohorts.sh

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
SPEC_DIR="${REPO_DIR}/configs/condition_cohorts"

# qsub -v parameters
SPEC="${SPEC:-}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
# A new root per compendium release keeps earlier runs intact.
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/results/condition_cohorts}"
SCORERS="${SCORERS:-scanpy:aucell}"
MINIMUM_CELLS="${MINIMUM_CELLS:-10}"
MINIMUM_DONORS="${MINIMUM_DONORS:-3}"
MINIMUM_STATE_PAIRS="${MINIMUM_STATE_PAIRS:-6}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
RANDOM_STATE="${RANDOM_STATE:-42}"
AUCELL_CHUNK_SIZE="${AUCELL_CHUNK_SIZE:-5000}"
AUCELL_NUM_WORKERS="${AUCELL_NUM_WORKERS:-1}"
DEVELOPMENT_DONORS_PER_ARM="${DEVELOPMENT_DONORS_PER_ARM:-}"
VISUALIZATIONS="${VISUALIZATIONS:-1}"
FETCH_INPUT="${FETCH_INPUT:-0}"
CALIBRATION="${CALIBRATION:-}"
AUDIT_ONLY="${AUDIT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

# PBS logging
PBS_COMMON="${HOME}/pbs_common.sh"
LOG_DIR="${PROJECT_DIR}/scripts/job_out"
JOB_NAME="${PBS_JOBNAME:-nasp_condition_cohort}"
JOB_ID="${PBS_JOBID:-local}"
LIVE_LOG="${LOG_DIR}/${JOB_NAME}.${JOB_ID}.live.log"

#######################################
# Resolve a spec filename against the condition-cohort config directory.
# Arguments:
#   $1: Spec filename or absolute path.
#   $2: Config directory for relative filenames.
# Outputs:
#   Writes the resolved path to stdout.
#######################################
resolve_spec_path() {
  local spec_name="$1"
  local spec_dir="$2"

  if [[ "${spec_name}" = /* ]]; then
    printf '%s\n' "${spec_name}"
    return
  fi
  printf '%s/%s\n' "${spec_dir}" "${spec_name}"
}

#######################################
# Run one repository Python script from the repository directory.
# Arguments:
#   $1: Script name under run_scripts/.
#   Remaining arguments: Options passed to the script.
# Returns:
#   The script's exit status.
#######################################
run_script() {
  local script="$1"
  shift

  (
    cd "${REPO_DIR}"
    PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
      python -u "${REPO_DIR}/run_scripts/${script}" "$@"
  )
}

#######################################
# Convert a boolean job setting to a Python flag.
# Arguments:
#   $1: Variable name, used in error messages.
#   $2: Variable value.
#   $3: Flag stem, such as "visualizations".
# Outputs:
#   Writes --<stem> or --no-<stem> to stdout.
# Returns:
#   0 on success; 2 when the value is not boolean.
#######################################
boolean_flag() {
  local name="$1"
  local value="$2"
  local stem="$3"

  case "${value}" in
    1|true|TRUE|yes|YES) printf -- '--%s\n' "${stem}" ;;
    0|false|FALSE|no|NO) printf -- '--no-%s\n' "${stem}" ;;
    *) printf 'ERROR: %s must be boolean, got %s\n' "${name}" "${value}" >&2; return 2 ;;
  esac
}

#######################################
# Validate job settings and run one condition-cohort population.
# Outputs:
#   Writes resolved job settings and analysis output to stdout and stderr.
# Returns:
#   0 on success; 2 for invalid job settings; otherwise the failing command's
#   exit status.
#######################################
main() {
  local spec_path=""
  local calibration_path=""
  local job_tmp=""
  local preview=0
  local fetch=0
  local audit_only=0
  local scorer_values=()
  local python_args=()
  local fetch_args=()
  local calibration_args=()

  if [[ -z "${SPEC}" ]]; then
    printf 'ERROR: submit with SPEC=<cohort>.yaml\n' >&2
    return 2
  fi
  case "${DRY_RUN}" in
    1|true|TRUE|yes|YES) preview=1 ;;
    0|false|FALSE|no|NO) ;;
    *) printf 'ERROR: DRY_RUN must be boolean\n' >&2; return 2 ;;
  esac
  case "${FETCH_INPUT}" in
    1|true|TRUE|yes|YES) fetch=1 ;;
    0|false|FALSE|no|NO) ;;
    *) printf 'ERROR: FETCH_INPUT must be boolean\n' >&2; return 2 ;;
  esac
  case "${AUDIT_ONLY}" in
    1|true|TRUE|yes|YES) audit_only=1 ;;
    0|false|FALSE|no|NO) ;;
    *) printf 'ERROR: AUDIT_ONLY must be boolean\n' >&2; return 2 ;;
  esac
  spec_path="$(resolve_spec_path "${SPEC}" "${SPEC_DIR}")"
  if (( ! preview )) && [[ ! -f "${spec_path}" ]]; then
    printf 'ERROR: spec does not exist: %s\n' "${spec_path}" >&2
    return 2
  fi
  if (( ! preview && ! fetch )) && [[ ! -d "${DATA_DIR}" ]]; then
    printf 'ERROR: data directory does not exist: %s\n' "${DATA_DIR}" >&2
    return 2
  fi
  if (( ! preview )) && [[ ! -f "${PBS_COMMON}" ]]; then
    printf 'ERROR: PBS helper does not exist: %s\n' "${PBS_COMMON}" >&2
    return 2
  fi

  IFS=':' read -r -a scorer_values <<< "${SCORERS}"
  python_args=(
    --spec "${spec_path}"
    --output-path "${OUT_DIR}"
    --data-dir "${DATA_DIR}"
    --minimum-cells "${MINIMUM_CELLS}"
    --minimum-donors "${MINIMUM_DONORS}"
    --minimum-state-pairs "${MINIMUM_STATE_PAIRS}"
    --n-bootstrap "${N_BOOTSTRAP}"
    --random-state "${RANDOM_STATE}"
    --aucell-chunk-size "${AUCELL_CHUNK_SIZE}"
    --aucell-num-workers "${AUCELL_NUM_WORKERS}"
    --scorers "${scorer_values[@]}"
    "$(boolean_flag VISUALIZATIONS "${VISUALIZATIONS}" visualizations)"
  )
  if [[ -n "${DEVELOPMENT_DONORS_PER_ARM}" ]]; then
    python_args+=(--development-donors-per-arm "${DEVELOPMENT_DONORS_PER_ARM}")
  fi
  if (( audit_only )); then
    python_args+=(--dry-run)
  fi

  fetch_args=(--spec "${spec_path}" --data-dir "${DATA_DIR}")
  if [[ -n "${CALIBRATION}" ]] && (( ! audit_only )); then
    calibration_path="$(resolve_spec_path "${CALIBRATION}" "${SPEC_DIR}")"
    calibration_args=(
      --spec "${spec_path}"
      --calibration "${calibration_path}"
      --data-dir "${DATA_DIR}"
      --output-path "${OUT_DIR}"
      --minimum-cells "${MINIMUM_CELLS}"
      --minimum-donors "${MINIMUM_DONORS}"
      "$(boolean_flag VISUALIZATIONS "${VISUALIZATIONS}" visualizations)"
    )
    if [[ -n "${DEVELOPMENT_DONORS_PER_ARM}" ]]; then
      calibration_args+=(
        --development-donors-per-arm "${DEVELOPMENT_DONORS_PER_ARM}"
      )
    fi
  fi

  if (( fetch )); then
    printf 'Fetch command: '
    printf '%q ' python -u "${REPO_DIR}/run_scripts/fetch_cohort_input.py" "${fetch_args[@]}"
    printf '\n'
  fi
  printf 'Worker command: '
  printf '%q ' python -u "${REPO_DIR}/run_scripts/condition_cohorts.py" "${python_args[@]}"
  printf '\n'
  if [[ -n "${calibration_path}" ]]; then
    printf 'Calibration command: '
    printf '%q ' python -u "${REPO_DIR}/run_scripts/atlas_calibration.py" "${calibration_args[@]}"
    printf '\n'
  fi
  if (( preview )); then
    return 0
  fi
  if [[ -n "${calibration_path}" && ! -f "${calibration_path}" ]]; then
    printf 'ERROR: calibration spec does not exist: %s\n' "${calibration_path}" >&2
    return 2
  fi

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

  printf 'Spec: %s\n' "${spec_path}"
  printf 'Data directory: %s\n' "${DATA_DIR}"
  printf 'Output root: %s\n' "${OUT_DIR}"
  printf 'Scorers: %s\n' "${SCORERS}"
  printf 'Development donors per arm: %s\n' "${DEVELOPMENT_DONORS_PER_ARM:-all}"
  printf 'Calibration: %s\n' "${CALIBRATION:-none}"

  if (( fetch )); then
    run_script fetch_cohort_input.py "${fetch_args[@]}"
  fi
  run_script condition_cohorts.py "${python_args[@]}"
  if [[ -n "${calibration_path}" ]]; then
    run_script atlas_calibration.py "${calibration_args[@]}"
  fi

  printf 'Job finished: %s\n' "$(date -Is)"
}


main "$@"
