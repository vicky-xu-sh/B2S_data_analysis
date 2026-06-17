#!/bin/bash
# Submit the full overt → covert pipeline.
#
# Usage:
#   bash submit_pipeline.sh                    # overt + overt plots + all covert conditions
#   bash submit_pipeline.sh --overt-only       # overt + overt plots only
#   bash submit_pipeline.sh --covert-only      # covert only (overt already done)
#   bash submit_pipeline.sh --array 0-1,3,5-7,9   # custom subject subset
#
# The overt plot job is gated on the overt array completing successfully.
# The covert jobs are gated on the overt array completing successfully
# (--dependency=afterok) unless --covert-only is used.
#
# Conditions submitted:
#   corr0.8_rank1_brain  corr0.8_allranks_brain

set -euo pipefail

CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
ARRAY_SPEC="0-1,3,5-7,9"   # subjects with data (skip 2=subj-03, 4=subj-05, 8=subj-11)
RUN_OVERT=true
RUN_COVERT=true

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --overt-only)  RUN_COVERT=false; shift ;;
        --covert-only) RUN_OVERT=false;  shift ;;
        --array)       ARRAY_SPEC="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

CONDITIONS=(
    corr0.8_rank1_brain     # other conditions (e.g. corr0.7_rank1_brain) can be added later if needed
    corr0.8_allranks_brain
)

# ---------------------------------------------------------------------------
# Stage 1: Overt
# ---------------------------------------------------------------------------
OVERT_JOB=""

if $RUN_OVERT; then
    OVERT_JOB=$(sbatch --parsable \
                       --array="${ARRAY_SPEC}" \
                       "${CLUSTER_DIR}/run_overt_array.sh")
    echo "Submitted overt array job: ${OVERT_JOB}  (array: ${ARRAY_SPEC})"
fi

# ---------------------------------------------------------------------------
# Stage 2: Plot overt summary figures (gated on overt completing)
# ---------------------------------------------------------------------------
if $RUN_OVERT; then
    PLOT_DEP=""
    [ -n "${OVERT_JOB}" ] && PLOT_DEP="--dependency=afterok:${OVERT_JOB}"
    PLOT_JOB=$(sbatch --parsable \
                      ${PLOT_DEP} \
                      --array="${ARRAY_SPEC}" \
                      "${CLUSTER_DIR}/run_plot_overt_array.sh")
    echo "Submitted overt plot job:  ${PLOT_JOB}  (array: ${ARRAY_SPEC})"
fi

# ---------------------------------------------------------------------------
# Stage 3: Covert — one array job per condition
# ---------------------------------------------------------------------------
if $RUN_COVERT; then
    DEPENDENCY=""
    if [ -n "${OVERT_JOB}" ]; then
        DEPENDENCY="--dependency=afterok:${OVERT_JOB}"
    fi

    for COND in "${CONDITIONS[@]}"; do
        JOB=$(sbatch --parsable \
                     ${DEPENDENCY} \
                     --array="${ARRAY_SPEC}" \
                     --job-name="covert_${COND}" \
                     --export=ALL,MATCH_COND="${COND}" \
                     "${CLUSTER_DIR}/run_covert_array.sh")
        echo "Submitted covert ${COND}: ${JOB}  (array: ${ARRAY_SPEC})"
    done
fi

echo ""
echo "Done. Monitor with: squeue -u \$USER"
echo "Logs: /scratch/st-ssfels-1/vickywx/logs_classification/"
