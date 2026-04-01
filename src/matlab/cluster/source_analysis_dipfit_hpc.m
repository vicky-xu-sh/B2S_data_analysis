% source_analysis_dipfit_hpc.m
% HPC-adapted source analysis pipeline for Sockeye cluster.
%
% Run via sbatch (see run_dipfit_hpc.sh), or directly:
%   matlab -nodisplay -nosplash -r "run('source_analysis_dipfit_hpc.m'); exit;"
%
% SUBJ, SPEECH_TYPE, and HEADMODEL_TYPE are read from environment variables set in the sbatch script.

%% =========================================================================
%  CONFIGURATION — update paths for your allocation
% ==========================================================================

EEGLAB_PATH    = '/arc/project/st-ssfels-1/tools/eeglab2025.0.0';
% FIELDTRIP_PATH = '/arc/project/st-ssfels-1/tools/fieldtrip';

BASE_PATH      = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/data';
HEADMODEL_DIR_ROOT = fullfile(BASE_PATH, '02_interim_local');
INPUT_DIR_ROOT     = fullfile(BASE_PATH, '03_interim_cluster');
OUTPUT_DIR_ROOT    = fullfile(BASE_PATH, '05_source_analysis');

%% =========================================================================
%  READ SUBJECT / CONDITION FROM ENVIRONMENT (set by sbatch script)
% ==========================================================================

SUBJ           = getenv('SUBJ');
SPEECH_TYPE    = getenv('SPEECH_TYPE');
HEADMODEL_TYPE = getenv('HEADMODEL_TYPE');
RV_THRES_STR   = getenv('RV_THRES');

if isempty(SUBJ)
    error('Error: SUBJ environment variable not set.');
end
if isempty(SPEECH_TYPE)
    error('Error: SPEECH_TYPE environment variable not set.');
end
if isempty(HEADMODEL_TYPE)
    HEADMODEL_TYPE = 'bemcp';   % default
    fprintf('[INFO] HEADMODEL_TYPE not set — defaulting to: %s\n', HEADMODEL_TYPE);
end
if isempty(RV_THRES_STR)
    RV_THRES = 0.15;            % default
    fprintf('[INFO] RV_THRES not set — defaulting to: %.2f\n', RV_THRES);
else
    RV_THRES = str2double(RV_THRES_STR);
end

if ~ismember(SPEECH_TYPE, {'sp', 'im'})
    error('SPEECH_TYPE must be ''sp'' or ''im''. Got: %s', SPEECH_TYPE);
end
if ~ismember(HEADMODEL_TYPE, {'bemcp', 'openmeeg'})
    error('HEADMODEL_TYPE must be ''bemcp'' or ''openmeeg''. Got: %s', HEADMODEL_TYPE);
end

fprintf('=== Processing: %s | speech_type=%s | headmodel=%s | rv_thres=%.2f ===\n', ...
    SUBJ, SPEECH_TYPE, HEADMODEL_TYPE, RV_THRES);

%% =========================================================================
%  SUBJECT-SPECIFIC ALIGNMENT PARAMETERS
%  Add a new case block here when processing a new subject
% ==========================================================================

switch SUBJ
    case 'subj-02'
        coord_transform = [-2 0 23 -0.142526 -0.00426732 0.117257 10.3 10.3 10.3];
        if strcmp(SPEECH_TYPE, 'sp')
            chanomit_idx = [210 211 217:245];
        else
            chanomit_idx = [60 66 204 205 211:240];
        end
    otherwise
        error('[ERROR] No alignment parameters defined for subject: %s\n  Add a case block in the CONFIGURATION section.', SUBJ);
end

%% =========================================================================
%  SETUP
% ==========================================================================

% Suppress all figure windows (no display on compute nodes)
set(0, 'DefaultFigureVisible', 'off');

% Add toolboxes and start EEGLAB without GUI
addpath(EEGLAB_PATH);
% addpath(FIELDTRIP_PATH);
% ft_defaults;
eeglab nogui;

global ALLEEG EEG CURRENTSET;
ALLEEG     = [];
EEG        = [];
CURRENTSET = 0;

% Build subject-specific paths
HEADMODEL_DIR = fullfile(HEADMODEL_DIR_ROOT, SUBJ, 'custom_headmodel');

if strcmp(SPEECH_TYPE, 'sp')
    INPUT_DIR  = fullfile(INPUT_DIR_ROOT,  SUBJ, 'spoken',   'datasets');
    OUTPUT_DIR = fullfile(OUTPUT_DIR_ROOT, SUBJ, 'spoken');
else
    INPUT_DIR  = fullfile(INPUT_DIR_ROOT,  SUBJ, 'imagined', 'datasets');
    OUTPUT_DIR = fullfile(OUTPUT_DIR_ROOT, SUBJ, 'imagined');
end

if ~exist(OUTPUT_DIR, 'dir'), mkdir(OUTPUT_DIR); end

fprintf('\n========================================================\n');
fprintf('  Subject:        %s\n', SUBJ);
fprintf('  Speech type:    %s\n', SPEECH_TYPE);
fprintf('  Headmodel type: %s\n', HEADMODEL_TYPE);
fprintf('  RV threshold:   %.2f\n', RV_THRES);
fprintf('  Input dir:      %s\n', INPUT_DIR);
fprintf('  Output dir:     %s\n', OUTPUT_DIR);
fprintf('========================================================\n\n');

%% =========================================================================
%  STEP 1 — Load preprocessed and epoched dataset
% ==========================================================================

setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname, '.set'];

fprintf('[STEP 1] Loading dataset: %s\n', filename);

if ~exist(fullfile(INPUT_DIR, filename), 'file')
    error('[ERROR] Dataset not found:\n  %s', fullfile(INPUT_DIR, filename));
end

EEG = pop_loadset('filename', filename, 'filepath', INPUT_DIR);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

fprintf('  → Loaded: %s\n',         EEG.setname);
fprintf('  → Channels:       %d\n', EEG.nbchan);
fprintf('  → ICA components: %d\n', size(EEG.icaweights, 1));
fprintf('  → Trials:         %d\n', EEG.trials);
fprintf('  → Epoch length:   %.0f samples  (%.3f s)\n', EEG.pnts, EEG.pnts / EEG.srate);
fprintf('  → Sampling rate:  %.0f Hz\n\n', EEG.srate);

%% =========================================================================
%  STEP 2 — Load headmodel files
% ==========================================================================

headmodel_path = fullfile(HEADMODEL_DIR, [SUBJ, '_headmodel_', HEADMODEL_TYPE, '.mat']);
mri_path       = fullfile(HEADMODEL_DIR, [SUBJ, '_mri_unbiased.mat']);
chanlocs_path  = fullfile(HEADMODEL_DIR, [SUBJ, '_fid_chanlocs.mat']);

fprintf('[STEP 2] Loading headmodel files...\n');

if ~exist(headmodel_path, 'file')
    error('[ERROR] Headmodel file not found:\n  %s', headmodel_path);
end
if strcmp(HEADMODEL_TYPE, 'openmeeg')
    load(headmodel_path, 'headmodel_openmeeg');
    headmodel = headmodel_openmeeg;
else
    load(headmodel_path, 'headmodel_bemcp');
    headmodel = headmodel_bemcp;
end
fprintf('  → Headmodel: %s\n', headmodel_path);

if ~exist(mri_path, 'file')
    error('[ERROR] MRI file not found:\n  %s', mri_path);
end
load(mri_path, 'mri_unbiased');
fprintf('  → MRI:       %s\n', mri_path);

if ~exist(chanlocs_path, 'file')
    error('[ERROR] Channel locations file not found:\n  %s', chanlocs_path);
end
load(chanlocs_path, 'chanlocs');
fprintf('  → Chanlocs:  %s\n\n', chanlocs_path);

% Attach headmodel info to EEG struct
EEG.dipfit.hdmfile  = headmodel;
EEG.dipfit.mrifile  = mri_unbiased;
EEG.dipfit.chanfile = chanlocs;

fprintf('[STEP 2b] Applying electrode alignment...\n');
EEG = pop_dipfit_settings(EEG, ...
    'coordformat',     'SCS', ...
    'coord_transform',  coord_transform, ...
    'chanomit',         chanomit_idx);

fprintf('  → coord_transform: [%s]\n', num2str(coord_transform));
fprintf('  → Omitting %d channels: [%s]\n\n', length(chanomit_idx), num2str(chanomit_idx));

%% =========================================================================
%  STEP 3 — Run DIPFIT autofit
% ==========================================================================

num_comps = size(EEG.icaweights, 1);
fprintf('[STEP 3] Running dipole fitting on %d ICA components...\n', num_comps);

EEG = pop_multifit(EEG, 1:num_comps, 'threshold', 100);

rv_values = [EEG.dipfit.model.rv];
good_ics  = find(rv_values <= RV_THRES);
poor_ics  = find(rv_values >  RV_THRES);

fprintf('  → Fitting complete.\n');
fprintf('  → RV ≤ %.2f (good fit): %d / %d — ICs: [%s]\n', RV_THRES, length(good_ics), num_comps, num2str(good_ics));
fprintf('  → RV >  %.2f (poor fit): %d / %d — ICs: [%s]\n\n', RV_THRES, length(poor_ics), num_comps, num2str(poor_ics));

% Save CTF-space result
setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_ctf_', HEADMODEL_TYPE, '_headmodel'];
filename = [setname, '.set'];
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname', setname, 'savenew', fullfile(OUTPUT_DIR, filename), 'gui', 'off');
fprintf('[SAVE] CTF-space result saved to:\n  %s\n\n', fullfile(OUTPUT_DIR, filename));

%% =========================================================================
%  STEP 4 — Warp dipoles to MNI space
% ==========================================================================

norm_mri_path = fullfile(HEADMODEL_DIR, [SUBJ, '_mri_normalised.mat']);
if ~exist(norm_mri_path, 'file')
    error('[ERROR] Normalised MRI file not found:\n  %s', norm_mri_path);
end
load(norm_mri_path, 'mri_normalised');

fprintf('[STEP 4] Warping dipoles to MNI space...\n');
fprintf('  → Normalised MRI: %s\n', norm_mri_path);

EEG2      = EEG;
model_mni = EEG.dipfit.model;

for i = 1:length(model_mni)
    if ~isempty(model_mni(i).posxyz)
        pos_ctf = model_mni(i).posxyz;

        if any(isnan(pos_ctf)) || all(pos_ctf == 0)
            fprintf('  [WARNING] IC %02d — suspicious CTF posxyz before warp: [%.2f %.2f %.2f]  rv=%.3f\n', ...
                i, pos_ctf(1), pos_ctf(2), pos_ctf(3), model_mni(i).rv);
        end

        pos_prealigned = ft_warp_apply(mri_normalised.initial, pos_ctf, 'homogeneous');
        if any(isnan(pos_prealigned))
            fprintf('  [WARNING] IC %02d — NaN after Step A (rigid-body).  CTF in: [%.2f %.2f %.2f]\n', ...
                i, pos_ctf(1), pos_ctf(2), pos_ctf(3));
        end

        pos_mni = ft_warp_apply(mri_normalised.params, pos_prealigned, 'individual2sn');
        if any(isnan(pos_mni))
            fprintf('  [WARNING] IC %02d — NaN after Step B (non-linear warp).  Pre-aligned: [%.2f %.2f %.2f]\n', ...
                i, pos_prealigned(1), pos_prealigned(2), pos_prealigned(3));
        end

        model_mni(i).posxyz = pos_mni;
    else
        fprintf('  [WARNING] IC %02d — posxyz is empty, skipping warp.\n', i);
    end
end

fprintf('  → MNI warp complete for %d components.\n\n', length(model_mni));

EEG2.dipfit.model           = model_mni;
EEG2.dipfit.coordformat     = 'MNI';
EEG2.dipfit.coord_transform = [];
EEG2.dipfit.mrifile         = mri_normalised;
EEG2.dipfit.hdmfile         = [];

fprintf('[STEP 4b] Running eeg_compatlas (DK atlas lookup)...\n');
EEG2 = eeg_compatlas(EEG2);
fprintf('  → eeg_compatlas complete.\n\n');

% Save MNI-space result
setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_ctf_', HEADMODEL_TYPE, '_headmodel_warptoMNI'];
filename = [setname, '.set'];
[ALLEEG, EEG2, CURRENTSET] = pop_newset(ALLEEG, EEG2, CURRENTSET, ...
    'setname', setname, 'savenew', fullfile(OUTPUT_DIR, filename), 'gui', 'off');
fprintf('[SAVE] MNI-space result saved to:\n  %s\n\n', fullfile(OUTPUT_DIR, filename));

%% =========================================================================
%  STEP 5 — AAL atlas lookup + dipole summary table
% ==========================================================================

fprintf('[STEP 5] Loading AAL atlas...\n');
[ftver, ftpath] = ft_version;
aal_path = fullfile(ftpath, 'template', 'atlas', 'aal', 'ROI_MNI_V4.nii');
if ~exist(aal_path, 'file')
    error('[ERROR] AAL atlas not found:\n  %s', aal_path);
end
atlas = ft_read_atlas(aal_path);
fprintf('  → AAL atlas loaded (FieldTrip v%s)\n\n', ftver);

fprintf('\n========================================================\n');
fprintf('  DIPOLE SUMMARY — %s  |  %s\n', SUBJ, SPEECH_TYPE);
fprintf('  RV threshold: %.2f\n', RV_THRES);
fprintf('========================================================\n');
fprintf('%-6s  %-6s  %7s %7s %7s   %s\n', 'IC', 'RV', 'X(MNI)', 'Y(MNI)', 'Z(MNI)', 'AAL Region');
fprintf('%s\n', repmat('-', 1, 65));

for i = 1:length(EEG2.dipfit.model)
    rv        = EEG2.dipfit.model(i).rv;
    mni_coord = EEG2.dipfit.model(i).posxyz;

    if rv > RV_THRES,         continue; end
    if any(isnan(mni_coord)), continue; end
    if all(mni_coord == 0),   continue; end

    evalc('region = lookup_aal_region(mni_coord, atlas);');
    fprintf('%-6d  %.3f  %7.1f %7.1f %7.1f   %s\n', ...
        i, rv, mni_coord(1), mni_coord(2), mni_coord(3), region);
end

fprintf('%s\n', repmat('-', 1, 65));
fprintf('========================================================\n\n');

fprintf('\n=== Source analysis complete for %s | %s ===\n', SUBJ, SPEECH_TYPE);

%% =========================================================================
%  LOCAL FUNCTIONS
% ==========================================================================

function region = lookup_aal_region(mni_coord, atlas)
    cfg        = [];
    cfg.roi    = mni_coord;
    cfg.atlas  = atlas;
    cfg.output = 'single';
    cfg.radius = 5;
    labels = ft_volumelookup(cfg, atlas);
    [maxcount, idx] = max(labels.count);
    if isempty(idx) || maxcount == 0
        region = 'Outside atlas';
    else
        region = labels.name{idx};
    end
end
