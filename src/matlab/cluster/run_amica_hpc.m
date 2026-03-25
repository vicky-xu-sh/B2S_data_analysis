% run_amica_hpc.m
% Runs AMICA on a pre-processed EEG .set file and saves the result.
% All paths and parameters are read from environment variables set in the sbatch script.

%% =========================================================================
%  CONFIGURATION
% ==========================================================================

EEGLAB_PATH = '/arc/project/st-ssfels-1/tools/eeglab2025.0.0';

%% =========================================================================
%  READ PATHS AND PARAMETERS FROM ENVIRONMENT
% ==========================================================================

INPUT_DIR        = getenv('INPUT_DIR');        % directory containing input .set file
INPUT_SET_NAME   = getenv('INPUT_SET_NAME');   % input filename without .set extension
OUTPUT_DIR       = getenv('OUTPUT_DIR');       % directory to save output .set file
OUTPUT_SET_NAME  = getenv('OUTPUT_SET_NAME');  % output filename without .set extension
AMICA_DIR        = getenv('AMICA_DIR');        % directory for AMICA intermediate output

% Numeric parameters
MAX_ITER = str2double(getenv('MAX_ITER'));
N_PCA    = str2double(getenv('N_PCA'));

% Fallback defaults (useful for interactive testing)
% if isempty(INPUT_DIR),       INPUT_DIR       = '/scratch/st-ssfels-1/vickywx/EEG_output/subj-02/sp/datasets'; end
% if isempty(INPUT_SET_NAME),  INPUT_SET_NAME  = 'subj-02_pilot_sp_raw_2hz_hp_badchan_removed_reref_resampled'; end
% if isempty(OUTPUT_DIR),      OUTPUT_DIR      = '/scratch/st-ssfels-1/vickywx/EEG_output/subj-02/sp/datasets'; end
% if isempty(OUTPUT_SET_NAME), OUTPUT_SET_NAME = 'subj-02_pilot_sp_AMICA'; end
% if isempty(AMICA_DIR),       AMICA_DIR       = '/scratch/st-ssfels-1/vickywx/EEG_output/subj-02/sp/amica'; end
% if isnan(MAX_ITER),          MAX_ITER        = 3000; end
% if isnan(N_PCA),             N_PCA           = 80;   end

% Auto-read SLURM CPU allocation for AMICA threading
n_threads_str = getenv('SLURM_CPUS_PER_TASK');
if isempty(n_threads_str)
    max_threads = 4;
else
    max_threads = str2double(n_threads_str);
end

fprintf('=== AMICA job ===\n');
fprintf('Input:      %s/%s.set\n', INPUT_DIR, INPUT_SET_NAME);
fprintf('Output:     %s/%s.set\n', OUTPUT_DIR, OUTPUT_SET_NAME);
fprintf('AMICA dir:  %s\n', AMICA_DIR);
fprintf('max_iter:   %d\n', MAX_ITER);
fprintf('n_pca:      %d\n', N_PCA);
fprintf('threads:    %d\n', max_threads);

%% =========================================================================
%  SETUP
% ==========================================================================

% Suppress all figure windows (no display on compute nodes)
set(0, 'DefaultFigureVisible', 'off');

% Add EEGLAB and start without GUI
addpath(EEGLAB_PATH);
eeglab nogui;

% Initialize EEGLAB globals
global ALLEEG EEG CURRENTSET;
ALLEEG     = [];
EEG        = [];
CURRENTSET = 0;

%% =========================================================================
%  LOAD INPUT DATASET
% ==========================================================================

fprintf('\n--- Loading input dataset ---\n');
input_file = fullfile(INPUT_DIR, [INPUT_SET_NAME, '.set']);

if ~exist(input_file, 'file')
    error('Input file not found: %s', input_file);
end

EEG = pop_loadset('filename', [INPUT_SET_NAME, '.set'], 'filepath', INPUT_DIR);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0);
fprintf('Loaded: %s (%d channels, %d timepoints)\n', input_file, EEG.nbchan, EEG.pnts);

%% =========================================================================
%  RUN AMICA
% ==========================================================================

% Set unique working directory so temp files don't collide between parallel jobs
slurm_tmp_dir = getenv('TMPDIR');
tmp_dir = fullfile(slurm_tmp_dir, ['tmp_amica_' OUTPUT_SET_NAME]);
mkdir(tmp_dir);
cd(tmp_dir);
fprintf('Using temp directory for AMICA temporary eeg fdt files: %s\n', tmp_dir);

fprintf('\n--- Running AMICA (%d threads, %d PCs, %d max iterations) ---\n', max_threads, N_PCA, MAX_ITER);

[weights, sphere, ~] = runamica15(EEG.data, ...
    'num_models',  1, ...
    'outdir',      AMICA_DIR, ...
    'numprocs',    1, ...
    'max_threads', max_threads, ...
    'max_iter',    MAX_ITER, ...
    'pcakeep',     N_PCA);

% Clean up temp directory
cd(getenv('SLURM_SUBMIT_DIR'));
rmdir(tmp_dir, 's');   % 's' flag removes recursively including all contents
fprintf('Cleaned up temp directory: %s\n', tmp_dir);

% Assign ICA weights to EEG
EEG.icaweights  = weights;
EEG.icasphere   = sphere;
EEG.icawinv     = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;

%% =========================================================================
%  SAVE OUTPUT DATASET
% ==========================================================================

fprintf('\n--- Saving output dataset ---\n');

filename = [OUTPUT_SET_NAME, '.set'];
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', OUTPUT_SET_NAME, ...
    'savenew', fullfile(OUTPUT_DIR, filename), ...
    'gui', 'off');

fprintf('Saved: %s\n', fullfile(OUTPUT_DIR, filename));
fprintf('\n=== AMICA complete ===\n');