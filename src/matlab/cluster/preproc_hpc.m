% preproc_hpc.m
% HPC-adapted EEG preprocessing pipeline for Sockeye cluster.
%
% Run via sbatch (see preproc_hpc.sh), or directly:
%   matlab -nodisplay -nosplash -r "run('preproc_hpc.m'); exit;"
%
% SUBJ and SPEECH_TYPE are read from environment variables set in the sbatch script.


%% =========================================================================
%  CONFIGURATION — update paths for your allocation
% ==========================================================================

EEGLAB_PATH = '/arc/project/st-ssfels-1/tools/eeglab2025.0.0';
INPUT_SET_PATH = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/02_interim_local'; 
OUTPUT_SET_PATH = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data/03_interim_cluster'; 
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

% Auto-read SLURM CPU allocation for AMICA threading
n_threads_str = getenv('SLURM_CPUS_PER_TASK');
if isempty(n_threads_str)
    MAX_THREADS = 4;
else
    MAX_THREADS = str2double(n_threads_str);
end

fprintf('=== Processing: %s | speech_type=%s | threads=%d ===\n', SUBJ, SPEECH_TYPE, MAX_THREADS);


%% =========================================================================
%  SETUP
% ==========================================================================

% Suppress all figure windows (no display on compute nodes)
set(0, 'DefaultFigureVisible', 'off');
set(0, 'DefaultFigurePosition', [0 0 1000 700]);

% Add EEGLAB and start without GUI
addpath(EEGLAB_PATH);
eeglab nogui;

% Initialize EEGLAB globals
global ALLEEG EEG CURRENTSET;
ALLEEG     = [];
EEG        = [];
CURRENTSET = 0;

% Build output directories
if strcmp(SPEECH_TYPE, 'sp')
    BASE_OUTPUT_DIR    = fullfile(OUTPUT_SET_PATH, SUBJ, 'spoken');
    INPUT_DIR   = fullfile(INPUT_SET_PATH, SUBJ, 'spoken');
elseif strcmp(SPEECH_TYPE, 'im')
    BASE_OUTPUT_DIR    = fullfile(OUTPUT_SET_PATH, SUBJ, 'imagined');
    INPUT_DIR   = fullfile(INPUT_SET_PATH, SUBJ, 'imagined');
end

% Output save path — must be on /scratch (read-write on compute nodes).
% After job completes, should copy final outputs to /arc/project for long-term storage.
FIG_DIR     = fullfile(BASE_OUTPUT_DIR, 'figures');
AMICA_DIR1  = fullfile(BASE_OUTPUT_DIR, 'amica_run1');
AMICA_DIR2  = fullfile(BASE_OUTPUT_DIR, 'amica_run2');
OUTPUT_DATASET_DIR = fullfile(BASE_OUTPUT_DIR, 'datasets');

for d = {BASE_OUTPUT_DIR, FIG_DIR, OUTPUT_DATASET_DIR}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

fprintf('Output directory: %s\n', BASE_OUTPUT_DIR);
fprintf('Output figures directory: %s\n', FIG_DIR);
fprintf('Output EEGLAB datasets directory: %s\n', OUTPUT_DATASET_DIR);
fprintf('Directories for AMICA: %s, %s. They will be created later when amica is run.\n', AMICA_DIR1, AMICA_DIR2);

% Filtering params
if strcmp(SPEECH_TYPE, 'sp')
    HP_CUTOFF = 2;
else
    HP_CUTOFF = 1;
end

% AMICA params
NUMPROC  = 1;
NUM_MODELS = 1;
MAX_ITER  = 3000;

% Epoch param
MAX_EPOCH_LENGTH = 2.0;

%% =========================================================================
%  CHECKPOINT FILENAMES
% ==========================================================================
ckpt_badchan        = fullfile(OUTPUT_DATASET_DIR, [SUBJ, '_pilot_', SPEECH_TYPE, '_', num2str(HP_CUTOFF), 'hz_hp_badchan_removed.set']);
ckpt_amica1         = fullfile(OUTPUT_DATASET_DIR, [SUBJ, '_pilot_', SPEECH_TYPE, '_', num2str(HP_CUTOFF), 'hz_hp_badchan_removed_reref_resampled_seg_removed_PCA*_AMICA.set']);
ckpt_ics_marked     = fullfile(OUTPUT_DATASET_DIR, [SUBJ, '_pilot_', SPEECH_TYPE, '_bp_1_150hz_bad_data_removed_full_transferred_ICs_marked_non_brain.set']);
ckpt_amica2         = fullfile(OUTPUT_DATASET_DIR, [SUBJ, '_pilot_', SPEECH_TYPE, '_bp_1_150hz_bad_data_removed_cleaned_2ndICA_*comps.set']);

%% =========================================================================
%  LOAD INPUT RAW AND EDITED SET 
% ==========================================================================

setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_raw_edited'];
filename = [setname, '.set'];
if ~exist(fullfile(INPUT_DIR, filename), 'file')
    error('Input file not found: %s', filename);
end

EEG = pop_loadset('filename', filename, 'filepath', INPUT_DIR);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
fprintf('Loaded: %s (%d channels, %d timepoints)\n', setname, EEG.nbchan, EEG.pnts);

fig = figure('Visible', 'off');
pop_spectopo(EEG, 1, [0 EEG.pnts], 'EEG', 'freq', [10 20 80], 'freqrange', [0.1 200], 'electrodes', 'off');
sgtitle('Raw data power spectra');
save_fig(fig, FIG_DIR, 'raw_data_power_spectra');

EEG_orig = EEG;   % keep a copy of the original before filtering

%% =========================================================================
%  HIGH-PASS FILTER + LINE NOISE REMOVAL
% ==========================================================================
if exist(ckpt_badchan, 'file')
    fprintf('\n--- CHECKPOINT: Loading filtered + bad-channel-removed set ---\n');
    [folder, setname, ext] = fileparts(ckpt_badchan);
    EEG = pop_loadset('filename', [setname, '.set'], 'filepath', OUTPUT_DATASET_DIR);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
else
    fprintf('\n--- Filtering ---\n');

    EEG = pop_eegfiltnew(EEG, 'locutoff', HP_CUTOFF);
    EEG = pop_eegfiltnew(EEG, 'locutoff',  58, 'hicutoff',  62, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 118, 'hicutoff', 122, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 178, 'hicutoff', 182, 'revfilt', 1);

    fig = figure('Visible', 'off');
    pop_spectopo(EEG, 1, [0 EEG.pnts], 'EEG', 'freq', [10 20 80], 'freqrange', [0.1 200], 'electrodes', 'off');
    sgtitle(sprintf('Power spectra after %d Hz HP and line noise removed', HP_CUTOFF));
    save_fig(fig, FIG_DIR, sprintf('power_spectra_after_%dhz_hp_linenoise_removed', HP_CUTOFF));


    %% =========================================================================
    %  REMOVE BAD CHANNELS
    % ==========================================================================

    fprintf('\n--- Removing bad channels ---\n');

    EEG = pop_clean_rawdata(EEG, ...
        'FlatlineCriterion',   5, ...
        'ChannelCriterion',    0.8, ...
        'LineNoiseCriterion',  4, ...
        'Highpass',            'off', ...
        'BurstCriterion',      'off', ...
        'WindowCriterion',     'off', ...
        'BurstRejection',      'off', ...
        'Distance',            'Euclidian');

    removed_chans  = EEG_orig.chanlocs(~EEG.etc.clean_channel_mask);
    removed_labels = {removed_chans.labels};
    fprintf('Removed channels (%d): %s\n', numel(removed_labels), strjoin(removed_labels, ', '));

    % Topoplot of removed channels
    if ~isempty(removed_labels)
        EEG_removed_only = pop_select(EEG_orig, 'channel', removed_labels);
        fig = figure('Visible', 'off');
        topoplot([], EEG_removed_only.chanlocs, 'style', 'blank', 'electrodes', 'labelpoint', 'chaninfo', EEG_removed_only.chaninfo);
        title(sprintf('Removed channels (%d)', numel(removed_labels)));
        save_fig(fig, FIG_DIR, 'removed_channels_topoplot');
        % NOTE: pop_eegplot of removed channels removed (interactive).
    end

    fig = figure('Visible', 'off');
    pop_spectopo(EEG, 1, [0 EEG.pnts], 'EEG', 'freq', [10 20 80], 'freqrange', [0.1 200], 'electrodes', 'off');
    sgtitle('Power spectra after filtered and bad channels removed');
    save_fig(fig, FIG_DIR, 'power_spectra_after_bad_chan_removed');


    %% =========================================================================
    %  SAVE FILTERED + BAD-CHANNEL-REMOVED SET
    % ==========================================================================

    setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_', num2str(HP_CUTOFF), 'hz_hp_badchan_removed'];
    filename = [setname, '.set'];
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', setname, 'savenew', fullfile(OUTPUT_DATASET_DIR, filename), 'gui', 'off');
end

ckpt_amica1_files = dir(ckpt_amica1);
if ~isempty(ckpt_amica1_files)
    ckpt_amica1_filename = ckpt_amica1_files(1).name;   % take first match
    fprintf('\n--- CHECKPOINT: Loading re-reference + resampled + segment-removed + 1st ICA set ---\n');
    [folder, setname, ext] = fileparts(ckpt_amica1_filename);
    EEG = pop_loadset('filename', [setname, '.set'], 'filepath', OUTPUT_DATASET_DIR);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

else

    %% =========================================================================
    %  RE-REFERENCE + RESAMPLE
    % ==========================================================================

    fprintf('\n--- Re-referencing to average and resampling to 500 Hz ---\n');
    EEG = pop_reref(EEG, []);
    EEG = pop_resample(EEG, 500);


    %% =========================================================================
    %  AUTOMATED BAD SEGMENT REJECTION (trim 1s from start and end)
    % ==========================================================================

    fprintf('\n--- Trimming 1s from start and end ---\n');

    trim_samples = floor(1 * EEG.srate);   % 1 second in samples
    start_trim   = 1;
    end_trim     = EEG.pnts - trim_samples;

    EEG = eeg_eegrej(EEG, [start_trim, trim_samples; end_trim, EEG.pnts]);

    fprintf('Trimmed samples 1-%d (start) and %d-%d (end)\n', trim_samples, end_trim, EEG.pnts);

    %% =========================================================================
    %  PCA + AMICA (Run 1)
    % ==========================================================================

    fprintf('\n--- PCA for AMICA run 1 ---\n');
    [n_pca_99, n_pca_999] = run_PCA(EEG, FIG_DIR, [SUBJ, '_', SPEECH_TYPE, '_pca_run1']);
    n_pca_run_1 = n_pca_99;
    if n_pca_run_1 < 60, n_pca_run_1 = n_pca_999; end
    if n_pca_run_1 > 90, n_pca_run_1 = 90; end   % cap at 90 to make sure k = 30 (k = num_samples/n_pca^2)

    fprintf('\nn_pca for ICA run 1: %d\n', n_pca_run_1);

    fprintf('\n--- Running AMICA (Run 1, %d threads) ---\n', MAX_THREADS);

    % Use SLURM's local temp directory — faster, unique per job, auto-cleaned
    tmp_dir = getenv('TMPDIR');
    fprintf('Using temp directory for AMICA temporary eeg fdt files: %s\n', tmp_dir);
    cd(tmp_dir);

    pcakeep   = n_pca_run_1;

    [weights, sphere, ~] = runamica15(EEG.data, ...
        'num_models',  NUM_MODELS, ...
        'outdir',      AMICA_DIR1, ...
        'numprocs',    NUMPROC, ...
        'max_threads', MAX_THREADS, ...
        'max_iter',    MAX_ITER, ...
        'pcakeep',     pcakeep);

    EEG.icaweights  = weights;
    EEG.icasphere   = sphere;
    EEG.icawinv     = pinv(EEG.icaweights * EEG.icasphere);
    EEG.icachansind = 1:EEG.nbchan;

    % Return to original directory 
    cd(getenv('SLURM_SUBMIT_DIR'));

    setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_', num2str(HP_CUTOFF), ...
                'hz_hp_badchan_removed_reref_resampled_seg_removed_PCA', num2str(pcakeep), '_AMICA'];
    filename = [setname, '.set'];
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', setname, 'savenew', fullfile(OUTPUT_DATASET_DIR, filename), 'gui', 'off');

end

EEG_ICA = EEG;   % keep copy with ICA weights

%% =========================================================================
%  PREPARE 1-150 Hz ANALYSIS DATASET
% ==========================================================================
if exist(ckpt_ics_marked, 'file')
    fprintf('\n--- CHECKPOINT: Loading bandpassed and non-brains ICs marked (from 1st ICA) set  ---\n');
    [folder, setname, ext] = fileparts(ckpt_ics_marked);
    EEG = pop_loadset('filename', [setname, '.set'], 'filepath', OUTPUT_DATASET_DIR);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

else

    fprintf('\n--- Preparing 1-150 Hz analysis dataset ---\n');

    EEG = pop_eegfiltnew(EEG_orig, 1, 150);
    EEG = pop_eegfiltnew(EEG, 'locutoff',  58, 'hicutoff',  62, 'revfilt', 1);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 118, 'hicutoff', 122, 'revfilt', 1);

    % Remove the same bad channels as in the HP-filtered dataset
    removed_chans2 = EEG.chanlocs(~EEG_ICA.etc.clean_channel_mask);
    removed_labels2 = {removed_chans2.labels};
    EEG = pop_select(EEG, 'nochannel', removed_labels2);

    % Sanity check
    if EEG_ICA.nbchan ~= EEG.nbchan
        error('Channel count mismatch between ICA dataset (%d) and analysis dataset (%d).', EEG_ICA.nbchan, EEG.nbchan);
    elseif ~isequal({EEG_ICA.chanlocs.labels}, {EEG.chanlocs.labels})
        error('Channel labels or order do not match — cannot safely transfer ICA.');
    else
        disp('Channel count and order are consistent.');
    end

    EEG = pop_reref(EEG, []);
    EEG = pop_resample(EEG, 500);

    % Apply same bad segment rejection
    fprintf('\n--- Trimming 1s from start and end ---\n');

    trim_samples = floor(1 * EEG.srate);   % 1 second in samples
    start_trim   = 1;
    end_trim     = EEG.pnts - trim_samples;

    EEG = eeg_eegrej(EEG, [start_trim, trim_samples; end_trim, EEG.pnts]);

    fprintf('Trimmed samples 1-%d (start) and %d-%d (end)\n', trim_samples, end_trim, EEG.pnts);


    %% =========================================================================
    %  TRANSFER ICA WEIGHTS + ICLABEL
    % ==========================================================================

    fprintf('\n--- Transferring ICA weights to 1-150 Hz dataset ---\n');

    W = EEG_ICA.icaweights * EEG_ICA.icasphere;
    EEG.icaweights  = W;
    EEG.icawinv     = pinv(W);
    EEG.icasphere   = eye(size(W, 2));
    EEG.icachansind = 1:size(W, 2);
    EEG.icaact      = [];
    EEG = eeg_checkset(EEG);

    fprintf('\n--- Running ICLabel (lite) ---\n');
    EEG = pop_iclabel(EEG, 'lite');
    % NOTE: pop_viewprops removed (interactive). IC figures saved below instead.


    %% =========================================================================
    %  MARK NON-BRAIN ICs FOR REJECTION
    % ==========================================================================

    ic_probs = EEG.etc.ic_classification.ICLabel.classifications;

    % Columns: 1=Brain 2=Muscle 3=Eye 4=Heart 5=LineNoise 6=ChanNoise 7=Other
    all_muscle_ICs   = find(ic_probs(:, 2) > 0.5);
    [~, max_classes] = max(ic_probs, [], 2);
    other_ICs        = find(max_classes == 7);
    thresh           = 0.05;
    non_brain_ICs    = find(ic_probs(:, 1) < thresh);
    ICs_to_reject    = setdiff(non_brain_ICs, other_ICs);

    fprintf('\nNon-brain ICs (brain prob < %.2f): %s\n', thresh, num2str(non_brain_ICs'));
    fprintf('ICs marked for rejection (excl. "other"): %s\n', num2str(ICs_to_reject'));
    fprintf('Rejected muscle ICs: %s\n', num2str(intersect(all_muscle_ICs, ICs_to_reject)'));

    EEG.reject.gcompreject = zeros(1, size(EEG.icaweights, 1));
    EEG.reject.gcompreject(ICs_to_reject) = 1;
    % NOTE: pop_selectcomps removed (interactive). Use the saved IC summary above.

    setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_bp_1_150hz_bad_data_removed_full_transferred_ICs_marked_non_brain'];
    filename = [setname, '.set'];
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', setname, 'savenew', fullfile(OUTPUT_DATASET_DIR, filename), 'gui', 'off');
end

%% =========================================================================
%  SUBTRACT REJECTED ICs
% ==========================================================================

fprintf('\n--- Subtracting rejected ICs ---\n');
num_rejected = sum(EEG.reject.gcompreject);
EEG = pop_subcomp(EEG, [], 0); % 0 means do not ask for confirmation
fprintf('\n Number of components subtracted: %d\n', num_rejected);

ckpt_amica2_files = dir(ckpt_amica2);
if ~isempty(ckpt_amica2_files)
    ckpt_amica2_filename = ckpt_amica2_files(1).name;   % take first match
    fprintf('\n--- CHECKPOINT: Loading 2nd ICA done set ---\n');
    [folder, setname, ext] = fileparts(ckpt_amica2_filename);
    EEG = pop_loadset('filename', [setname, '.set'], 'filepath', OUTPUT_DATASET_DIR);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
else

    %% =========================================================================
    %  PCA + AMICA (Run 2)
    % ==========================================================================

    fprintf('\n--- PCA for AMICA run 2 ---\n');

    run_PCA(EEG, FIG_DIR, [SUBJ, '_', SPEECH_TYPE, '_pca_run2']);  % called for figure only, return value not used

    n_pca_run_2 = 90 - num_rejected; % adjust PCA components to keep after IC rejection to maintain at least k = 30 (k = num_samples/n_pca^2)
    fprintf('\nn_pca for ICA run 2: %d\n', n_pca_run_2);

    fprintf('\n--- Running AMICA (Run 2, %d threads) ---\n', MAX_THREADS);
    pcakeep = n_pca_run_2;

    % Use SLURM's local temp directory — faster, unique per job, auto-cleaned
    tmp_dir = getenv('TMPDIR');
    fprintf('Using temp directory for AMICA temporary eeg fdt files: %s\n', tmp_dir);
    cd(tmp_dir);

    [weights, sphere, ~] = runamica15(EEG.data, ...
        'num_models',  NUM_MODELS, ...
        'outdir',      AMICA_DIR2, ...
        'numprocs',    NUMPROC, ...
        'max_threads', MAX_THREADS, ...
        'max_iter',    MAX_ITER, ...
        'pcakeep',     pcakeep);

    % Return to original directory 
    cd(getenv('SLURM_SUBMIT_DIR'));

    EEG.icaweights  = weights;
    EEG.icasphere   = sphere;
    EEG.icawinv     = pinv(EEG.icaweights * EEG.icasphere);
    EEG.icachansind = 1:EEG.nbchan;

    fprintf('\n--- Running ICLabel (default) ---\n');
    EEG = pop_iclabel(EEG, 'default');
    % NOTE: pop_viewprops removed (interactive).

    setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_bp_1_150hz_bad_data_removed_cleaned_2ndICA_', num2str(pcakeep), 'comps'];
    filename = [setname, '.set'];
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', setname, 'savenew', fullfile(OUTPUT_DATASET_DIR, filename), 'gui', 'off');
end

%% =========================================================================
%  SPEECH STATISTICS (continuous EEG — overt speech only)
% ==========================================================================

if strcmp(SPEECH_TYPE, 'sp')
    fprintf('\n--- Computing speech onset/offset statistics (continuous EEG) ---\n');

    cond_labels     = {'giSP', 'guSP', 'miSP', 'muSP', 'siSP', 'suSP'};
    cond_onsets     = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));
    cond_offsets    = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));
    overall_onsets  = [];
    overall_offsets = [];

    nEvents   = length(EEG.event);
    all_types = {EEG.event.type};
    all_lats  = [EEG.event.latency];

    stim_indices = [];
    stim_conds   = {};
    for i = 1:nEvents
        lbl = extract_cond_label(all_types{i}, cond_labels);
        if ~isempty(lbl)
            stim_indices(end+1) = i;
            stim_conds{end+1}   = lbl;
        end
    end
    fprintf('Found %d stimulus events.\n', length(stim_indices));

    for s = 1:length(stim_indices)
        stim_idx     = stim_indices(s);
        stim_lat_smp = all_lats(stim_idx);
        cond         = stim_conds{s};
        if s < length(stim_indices)
            search_end = stim_indices(s+1) - 1;
        else
            search_end = nEvents;
        end
        for j = (stim_idx + 1) : search_end
            etype  = all_types{j};
            rel_ms = (all_lats(j) - stim_lat_smp) / EEG.srate * 1000;
            if strcmp(etype, 'onset')
                overall_onsets(end+1)  = rel_ms;
                cond_onsets(cond)      = [cond_onsets(cond), rel_ms];
            elseif strcmp(etype, 'offset')
                overall_offsets(end+1) = rel_ms;
                cond_offsets(cond)     = [cond_offsets(cond), rel_ms];
            end
        end
    end

    fprintf('\n--- Overall Latency Statistics (ms relative to stimulus onset) ---\n');
    if ~isempty(overall_onsets)
        fprintf('Onset:  Mean=%.2f  Min=%.2f  Max=%.2f ms\n', mean(overall_onsets), min(overall_onsets), max(overall_onsets));
    else
        fprintf('Onset:  No data\n');
    end
    if ~isempty(overall_offsets)
        fprintf('Offset: Mean=%.2f  Min=%.2f  Max=%.2f ms\n', mean(overall_offsets), min(overall_offsets), max(overall_offsets));
    else
        fprintf('Offset: No data\n');
    end

    fprintf('\n--- Per-Condition Latency Statistics ---\n');
    for k = 1:length(cond_labels)
        label   = cond_labels{k};
        onsets  = cond_onsets(label);
        offsets = cond_offsets(label);
        fprintf('%s:\n', label);
        if ~isempty(onsets)
            fprintf('  Onset:  Mean=%.2f  Min=%.2f  Max=%.2f  [N=%d]\n', mean(onsets), min(onsets), max(onsets), length(onsets));
        else
            fprintf('  Onset:  No data\n');
        end
        if ~isempty(offsets)
            fprintf('  Offset: Mean=%.2f  Min=%.2f  Max=%.2f  [N=%d]\n', mean(offsets), min(offsets), max(offsets), length(offsets));
        else
            fprintf('  Offset: No data\n');
        end
    end
end


%% =========================================================================
%  EPOCH
% ==========================================================================

fprintf('\n--- Epoching ---\n');
setname = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];

if strcmp(SPEECH_TYPE, 'sp')
    epoch_events = { ...
        'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _guSP_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _miSP_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _muSP_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _siSP_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _suSP_[]_ECI TCP-IP 55513' };
else
    epoch_events = { ...
        'EVNT_STIM_    _giIM_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _guIM_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _miIM_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _muIM_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _siIM_[]_ECI TCP-IP 55513' ...
        'EVNT_STIM_    _suIM_[]_ECI TCP-IP 55513' };
end

EEG = pop_epoch(EEG, epoch_events, [-0.5 MAX_EPOCH_LENGTH], 'newname', setname, 'epochinfo', 'yes');

filename = [setname, '.set'];
EEG = pop_saveset(EEG, 'filename', filename, 'filepath', OUTPUT_DATASET_DIR);
fprintf('Epoched dataset saved: %s\n', filename);


%% =========================================================================
%  COMPUTE VOICE ONSET AND OFFSET RELATIVE TO STIMULUS (epoched data, overt speech only)
% ==========================================================================
if strcmp(SPEECH_TYPE, 'sp')
    fprintf('\n--- Computing epoch-level onset/offset latencies ---\n');

    nEpochs          = length(EEG.epoch);
    onset_latencies  = nan(1, nEpochs);
    offset_latencies = nan(1, nEpochs);
    cond_labels      = {'giSP', 'guSP', 'miSP', 'muSP', 'siSP', 'suSP'};
    cond_onsets      = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));
    cond_offsets     = containers.Map(cond_labels, repmat({[]}, size(cond_labels)));

    for i = 1:nEpochs
        etype  = EEG.epoch(i).eventtype;
        elat   = EEG.epoch(i).eventlatency;
        elabel = EEG.epoch(i).eventlabel;

        if ~iscell(etype),  etype  = {etype};  end
        if ~iscell(elat),   elat   = {elat};   end
        if ~iscell(elabel), elabel = {elabel}; end

        onset_idx  = strcmp(etype, 'onset');
        offset_idx = strcmp(etype, 'offset');

        if any(onset_idx),  onset_latencies(i)  = elat{onset_idx};  end
        if any(offset_idx), offset_latencies(i) = elat{offset_idx}; end

        cond_match = intersect(elabel, cond_labels);
        if ~isempty(cond_match)
            label = cond_match{1};
            if any(onset_idx),  cond_onsets(label)  = [cond_onsets(label),  elat{onset_idx}];  end
            if any(offset_idx), cond_offsets(label) = [cond_offsets(label), elat{offset_idx}]; end
        end
    end

    fprintf('--- Overall Latency Statistics ---\n');
    fprintf('Onset:  Mean=%.2f  Min=%.2f  Max=%.2f ms\n', ...
        mean(onset_latencies,'omitnan'), min(onset_latencies,[],'omitnan'), max(onset_latencies,[],'omitnan'));
    fprintf('Offset: Mean=%.2f  Min=%.2f  Max=%.2f ms\n', ...
        mean(offset_latencies,'omitnan'), min(offset_latencies,[],'omitnan'), max(offset_latencies,[],'omitnan'));

    fprintf('\n--- Per-Condition Latency Statistics ---\n');
    for k = 1:length(cond_labels)
        label   = cond_labels{k};
        onsets  = cond_onsets(label);
        offsets = cond_offsets(label);
        fprintf('%s:\n', label);
        if ~isempty(onsets)
            fprintf('  Onset:  Mean=%.2f  Min=%.2f  Max=%.2f\n', mean(onsets), min(onsets), max(onsets));
        else
            fprintf('  Onset:  No data\n');
        end
        if ~isempty(offsets)
            fprintf('  Offset: Mean=%.2f  Min=%.2f  Max=%.2f\n', mean(offsets), min(offsets), max(offsets));
        else
            fprintf('  Offset: No data\n');
        end
    end


    %% =========================================================================
    %  SAVE SPEECH ONSET/OFFSET
    % ==========================================================================

    speech_timing_filename = fullfile(OUTPUT_MAT_PATH, [SUBJ, '_speech_onset_offset.mat']);
    save(speech_timing_filename, 'onset_latencies', 'offset_latencies');
    fprintf('Saved speech timing: %s\n', speech_timing_filename);
end


fprintf('\n=== Preprocessing complete for %s | %s ===\n', SUBJ, SPEECH_TYPE);


%% =========================================================================
%  LOCAL FUNCTIONS
% ==========================================================================

function save_fig(fig, FIG_DIR, name)
% Save figure as PNG and close it.
    fname = fullfile(FIG_DIR, [name, '.png']);
    saveas(fig, fname);
    close(fig);
    fprintf('Saved figure: %s\n', fname);
end


function [n_pca_99, n_pca_999] = run_PCA(EEG, FIG_DIR, fig_prefix)
% Run PCA on EEG data and return number of components explaining 99% and 99.9% variance.
% Uses covariance matrix eigendecomposition (faster than SVD for large data).
% Saves cumulative explained variance figure to FIG_DIR.

    data    = double(EEG.data);
    data_2d = reshape(data, EEG.nbchan, []);
    data_2d = data_2d - mean(data_2d, 2);   % centre

    fprintf('PCA input: %d channels x %d timepoints\n', size(data_2d, 1), size(data_2d, 2));

    % Covariance matrix eigendecomposition — much faster than full SVD
    cov_matrix   = (data_2d * data_2d') / (size(data_2d, 2) - 1);
    eigenvalues  = eig(cov_matrix);
    eigenvalues  = sort(eigenvalues, 'descend');
    eigenvalues  = max(eigenvalues, 0);   % clamp negatives to zero
    explained    = eigenvalues / sum(eigenvalues);
    cumulative_explained = cumsum(explained);

    n_pca_99  = find(cumulative_explained >= 0.99,  1);
    n_pca_999 = find(cumulative_explained >= 0.999, 1);

    fprintf('PCs explaining 99%%:   %d\n', n_pca_99);
    fprintf('PCs explaining 99.9%%: %d\n', n_pca_999);

    % Plot
    fig = figure('Visible', 'off');
    plot(cumulative_explained * 100, 'LineWidth', 2);
    xlabel('Number of Principal Components');
    ylabel('Cumulative Explained Variance (%)');
    title(sprintf('PCA Cumulative Explained Variance (%s)', strrep(fig_prefix, '_', ' ')));
    grid on;
    hold on;
    yline(99,   '--r', '99% threshold');
    yline(99.9, '--g', '99.9% threshold');
    xline(n_pca_99,  '-.r', sprintf('99%% = %d',  n_pca_99));
    xline(n_pca_999, '-.g', sprintf('99.9%% = %d', n_pca_999));
    xlim([1 EEG.nbchan]);
    save_fig(fig, FIG_DIR, [fig_prefix, '_cumvar']);
end


function label = extract_cond_label(etype, cond_labels)
% Extract condition label from a stimulus event string.
% e.g. 'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513' --> 'giSP'
    label = '';
    for k = 1:length(cond_labels)
        if contains(etype, cond_labels{k})
            label = cond_labels{k};
            return;
        end
    end
end
