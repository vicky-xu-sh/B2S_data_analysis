#!/bin/bash
#SBATCH --job-name=dipfit
#SBATCH --account=st-ssfels-1
#SBATCH --time=06:00:00                
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G                       
#SBATCH --output=/scratch/st-ssfels-1/vickywx/logs_dipfit/%x_%j.out
#SBATCH --error=/scratch/st-ssfels-1/vickywx/logs_dipfit/%x_%j.err
#SBATCH --mail-user=vickywx@student.ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL

# ---------------------------------------------------------------------------
# Subject and condition — edit here to run a specific subject
# ---------------------------------------------------------------------------
export SUBJ="subj-02"
export SPEECH_TYPE="sp"          # "sp" = spoken/overt | "im" = imagined/covert
export HEADMODEL_TYPE="openmeeg"    # "bemcp" or "openmeeg"
export RV_THRES="0.15"

# ---------------------------------------------------------------------------
# MATLAB environment setup
# ---------------------------------------------------------------------------
module load gcc/9.4.0
module load matlab/R2024b

# Redirect MATLAB prefs to scratch (home is read-only on compute nodes)
export MATLAB_PREFDIR=/scratch/st-ssfels-1/vickywx/matlab_prefs
mkdir -p $MATLAB_PREFDIR

export XDG_CACHE_HOME=/scratch/st-ssfels-1/vickywx/cache
mkdir -p $XDG_CACHE_HOME

# ---------------------------------------------------------------------------
# Create log directory
# ---------------------------------------------------------------------------
mkdir -p /scratch/st-ssfels-1/vickywx/logs_dipfit

# ---------------------------------------------------------------------------
# Run source analysis
# ---------------------------------------------------------------------------
echo "===== Job started: $(date) ====="
echo "Node: $(hostname)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"
echo "Subject: ${SUBJ} | Speech type: ${SPEECH_TYPE} | Headmodel: ${HEADMODEL_TYPE}"

matlab -nodisplay -nosplash -nodesktop -r \
    "try; run('/scratch/st-ssfels-1/vickywx/B2S_data_analysis/src/matlab/cluster/source_analysis_dipfit_hpc.m'); catch e; fprintf('ERROR: %s\n', e.message); exit(1); end; exit(0);"

echo "===== Job finished: $(date) ====="
