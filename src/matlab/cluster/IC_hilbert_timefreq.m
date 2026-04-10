% IC_hilbert_timefreq.m
 
%% =========================================================================
%  CONFIGURATION — update paths for your allocation
% ==========================================================================
 
EEGLAB_PATH     = '/arc/project/st-ssfels-1/tools/eeglab2025.0.0';
INPUT_BASE_PATH  = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/03_interim_cluster';
OUTPUT_MAT_PATH = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/04_processed';
 
 
%% =========================================================================
%  READ SUBJECT / CONDITION FROM ENVIRONMENT (set by sbatch script)
% ==========================================================================
 
SUBJ        = getenv('SUBJ');
SPEECH_TYPE = getenv('SPEECH_TYPE');
 
if isempty(SUBJ)
    error('Error: SUBJ environment variable not set.\n');
end
if isempty(SPEECH_TYPE)
    error('Error: SPEECH_TYPE environment variable not set.\n');
end
 
fprintf('=== Processing: %s | speech_type=%s ===\n', SUBJ, SPEECH_TYPE);
 
 
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

if strcmp(SPEECH_TYPE, 'sp')
    INPUT_DIR   = fullfile(INPUT_BASE_PATH, SUBJ, 'spoken', 'datasets');
elseif strcmp(SPEECH_TYPE, 'im')
    INPUT_DIR   = fullfile(INPUT_BASE_PATH, SUBJ, 'imagined', 'datasets');
end

fprintf('Input directory: %s\n', INPUT_DIR);
fprintf('Output directory: %s\n', OUTPUT_MAT_PATH);

 
%% =========================================================================
%  Load preprocessed and epoched dataset
% ==========================================================================
 
setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname, '.set'];
 
fprintf('[STEP 1] Loading dataset: %s\n', filename);
 
if ~exist(fullfile(INPUT_DIR, filename), 'file')
    error('[ERROR] Dataset not found:\n  %s', fullfile(INPUT_DIR, filename));
end
 
EEG = pop_loadset('filename', filename, 'filepath', INPUT_DIR);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
 
fprintf('  -> Loaded: %s\n',         EEG.setname);
fprintf('  -> Channels:       %d\n', EEG.nbchan);
fprintf('  -> ICA components: %d\n', size(EEG.icaweights, 1));
fprintf('  -> Trials:         %d\n', EEG.trials);
fprintf('  -> Epoch length:   %.0f samples  (%.3f s)\n', EEG.pnts, EEG.pnts / EEG.srate);
fprintf('  -> Sampling rate:  %.0f Hz\n\n', EEG.srate);
 
 
%% =========================================================================
%  Get trial class labels
% ==========================================================================
 
nEpochs = EEG.trials;
classes = {'gi', 'gu', 'mi', 'mu', 'si', 'su'};
labels  = nan(1, nEpochs);
 
for i = 1:nEpochs
    etype = EEG.epoch(i).eventtype;
 
    if iscell(etype)
        cond_idx       = find(contains(etype, 'EVNT_STIM_'));
        cond_str       = etype{cond_idx};
        class_match_idx = find(cellfun(@(c) contains(cond_str, c), classes));
        labels(i)      = class_match_idx;
    end
end
 
fprintf('  -> Labels assigned: %d / %d trials\n\n', sum(~isnan(labels)), nEpochs);
 
 
%% =========================================================================
%  Compute broadband IC activations once
% ========================================================================== 
fprintf('[STEP 2] Computing broadband IC activations...\n');
 
icaact = (EEG.icaweights * EEG.icasphere) * EEG.data(EEG.icachansind, :);
% Reshape to [ICs x time x trials]
icaact = reshape(icaact, size(EEG.icaweights, 1), EEG.pnts, EEG.trials);
 
fs    = EEG.srate;
times = EEG.times;
 
nComps  = size(icaact, 1);
nTimes  = size(icaact, 2);
nTrials = size(icaact, 3);
 
fprintf('  -> IC activations shape: [%d ICs x %d timepoints x %d trials]\n\n', ...
    nComps, nTimes, nTrials);
 
 
%% =========================================================================
%  Define frequency bands
% ==========================================================================
 
theta_band      = [4   7  ];
alpha_band      = [8   12 ];
beta_band       = [13  30 ];
gamma_band      = [30  75 ];
high_gamma_band = [75  150];
 
bands      = {theta_band, alpha_band, beta_band, gamma_band, high_gamma_band};
band_names = {'Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma'};
nBands     = length(bands);

 
%% =========================================================================
%  Filter IC activations per band and apply Hilbert transform
% ==========================================================================
 
% Preallocate: [ICs x bands x time x trials] — complex analytic signal
analytic_signal = complex(nan(nComps, nBands, nTimes, nTrials));
 
% Reshape ICA activations to 2D [ICs x (time * trials)] for filtering.
% eegfilt/firfilt treat rows as independent channels — exactly what we want.
% This avoids filtering all EEG channels 5 times and re-projecting each time.
icaact_2d = reshape(icaact, nComps, nTimes * nTrials);
 
for b = 1:nBands
    band_range = bands{b};
    fprintf('[STEP 3] Band %d/%d: %s  [%d-%d Hz]...\n', ...
        b, nBands, band_names{b}, band_range(1), band_range(2));
 
    % Filter IC time-series in 2D, then reshape back to 3D
    icaact_filt_2d = eegfilt(icaact_2d, fs, band_range(1), band_range(2));
    icaact_filt    = reshape(icaact_filt_2d, nComps, nTimes, nTrials);
 
    % Apply Hilbert transform trial-by-trial to avoid edge artifacts
    % at trial boundaries
    for c = 1:nComps
        for tr = 1:nTrials
            signal = squeeze(icaact_filt(c, :, tr));
            analytic_signal(c, b, :, tr) = hilbert(signal);
        end
    end
end
 
fprintf('\n');
 
 
%% =========================================================================
%  Save analytic signal and metadata
% ==========================================================================
 
out_filename = fullfile(OUTPUT_MAT_PATH, [SUBJ, '_', SPEECH_TYPE, '_eeg_analytic.mat']);
 
fprintf('[STEP 4] Saving output to:\n  %s\n', out_filename);
 
save(out_filename, 'analytic_signal', 'labels', 'times', 'band_names', 'bands', '-v7.3');
 
fprintf('  -> Saved variables: analytic_signal [%d x %d x %d x %d], labels, times, band_names, bands\n', ...
    nComps, nBands, nTimes, nTrials);
 
fprintf('\n=== Hilbert transform time-frequency complete for %s | %s ===\n', SUBJ, SPEECH_TYPE);