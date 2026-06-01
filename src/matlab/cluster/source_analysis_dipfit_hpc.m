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
    HEADMODEL_TYPE = 'template';   % default
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
if ~ismember(HEADMODEL_TYPE, {'bemcp', 'openmeeg', 'template'})
    error('HEADMODEL_TYPE must be ''bemcp'', ''openmeeg'', or ''template''. Got: %s', HEADMODEL_TYPE);
end

fprintf('=== Processing: %s | speech_type=%s | headmodel=%s | rv_thres=%.2f ===\n', ...
    SUBJ, SPEECH_TYPE, HEADMODEL_TYPE, RV_THRES);

%% =========================================================================
%  SUBJECT-SPECIFIC ALIGNMENT PARAMETERS
%  Add a new case block here when processing a new subject
%  (Only used for custom headmodels; skipped for template)
% ==========================================================================

if ~strcmp(HEADMODEL_TYPE, 'template')
    switch SUBJ
        case 'subj-02'
            coord_transform = [-2 0 23 -0.142526 -0.00426732 0.117257 10.3 10.3 10.3];
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [210 211 217:245];
            else
                chanomit_idx = [60 66 204 205 211:240];
            end
        case 'subj-03'
            coord_transform = [10.5 -1.2 17 0.086 -0.03 -0.06 10.01 10.01 10.01]; 
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [98 106 116 162:184];
            else
                chanomit_idx = [67 98 105 114 125 135 144 152 171 182 183 187:216];
            end
        case 'subj-04'
            coord_transform = [3.8 -3 18.7 -0.12 -0.09 -0.03 10.1 10.1 10.1]; 
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [181 187:210];
            else
                chanomit_idx = [72 80 183 184 190:217];
            end
        
        case 'subj-05'
            coord_transform = [10 3 33 0.09 -0.1 0 9.8 9.8 9.8];  
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [64 70 163 169 170 171 176:204];
            else
                chanomit_idx = [64 70 78 117 143 163 169 170 171 172 177:207];
            end
        case 'subj-06'
            coord_transform = [4 0.5 38 0.02 0.18 0.04 10.05 10.05 10.05];  
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [60 64 160:174];
            else
                chanomit_idx = [67 72 185 191:216];
            end
        case 'subj-07'
            % coord_transform = [6 -1.2 33 0.06 0.14 0 10.6 10.6 10.6]; % bad
            coord_transform = [8 -1.2 35 0.06 0.06 0 10.4 10.4 10.4];
            if strcmp(SPEECH_TYPE, 'sp')
                chanomit_idx = [66 72 207 208 214:235];
            else
                chanomit_idx = [66 72 81 89 212 213 214 221:249];
            end
        otherwise
            error('[ERROR] No alignment parameters defined for subject: %s\n  Add a case block in the CONFIGURATION section.', SUBJ);
    end
end % if ~strcmp(HEADMODEL_TYPE, 'template')

%% =========================================================================
%  SETUP
% ==========================================================================

% Suppress all figure windows (no display on compute nodes)
set(0, 'DefaultFigureVisible', 'off');

% Add toolboxes and start EEGLAB without GUI
addpath(EEGLAB_PATH);

if strcmp(HEADMODEL_TYPE, 'openmeeg')
    setenv('PATH', [getenv('PATH') ':/arc/project/st-ssfels-1/tools/OpenMEEG-2.4.1-Linux/bin/']);
end

eeglab nogui;

global ALLEEG EEG CURRENTSET;
ALLEEG     = [];
EEG        = [];
CURRENTSET = 0;

% Build subject-specific paths
HEADMODEL_DIR = fullfile(HEADMODEL_DIR_ROOT, SUBJ, 'custom_headmodel');

% Default EGI channel location file (for template headmodel config)
TEMPLATE_ELEC_CHANLOC_FILE = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/default_template_chanlocs.ced';

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

if strcmp(HEADMODEL_TYPE, 'template')
    fprintf('Lookup standard EGI ELECTRODE locations...\n');
    EEG = pop_chanedit(EEG, 'lookup', TEMPLATE_ELEC_CHANLOC_FILE);
end 

%% =========================================================================
%  STEP 2 — DIPFIT setup
% ==========================================================================

if strcmp(HEADMODEL_TYPE, 'template')

    fprintf('[STEP 2] Loading standard headmodel files and settings...\n');

    target_labels = { ...
        'E67','E73','E82','E91','E92','E102','E103','E111','E120', ...
        'E133','E145','E165','E174','E187','E199','E208','E209','E216','E217', ...
        'E218','E219','E225','E226','E227','E228','E229','E230', ...
        'E231','E232','E233','E234','E235','E236','E237','E238', ...
        'E239','E240','E241','E242','E243','E244','E245','E246', ...
        'E247','E248','E249','E250','E251','E252','E253','E254', ...
        'E255','E256'};

    all_labels = {EEG.chanlocs.labels};
    [found, chan_idx] = ismember(target_labels, all_labels);
    chan_idx = chan_idx(found);
    missing_labels = target_labels(~found);

    fprintf('Found %d channels in the target chanomit set.\n', numel(chan_idx));
    if ~isempty(missing_labels)
        fprintf(['Channels not found (should already be removed in previous ' ...
                 'preprocessing steps): %s\n'], strjoin(missing_labels, ', '));
    end

    headmodel_path = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/dipfit_standard_templates/standard_vol.mat';
    mri_path       = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/dipfit_standard_templates/standard_mri.mat';
    chanlocs_path  = '/scratch/st-ssfels-1/vickywx/B2S_data_analysis/dipfit_standard_templates/standard_1005.elc';

    if ~exist(headmodel_path, 'file')
        error('[ERROR] Headmodel file not found:\n  %s', headmodel_path);
    end
    load(headmodel_path);
    fprintf('  → Headmodel: %s\n', headmodel_path);

    if ~exist(mri_path, 'file')
        error('[ERROR] MRI file not found:\n  %s', mri_path);
    end
    load(mri_path);
    fprintf('  → MRI:       %s\n', mri_path);

    if ~exist(chanlocs_path, 'file')
        error('[ERROR] Channel locations file not found:\n  %s', chanlocs_path);
    end
    fprintf('  → Chanlocs:  %s\n\n', chanlocs_path);

    % Attach headmodel info to EEG struct
    EEG.dipfit.hdmfile  = vol;
    EEG.dipfit.mrifile  = mri;
    EEG.dipfit.chanfile = chanlocs_path;

    fprintf('[STEP 2b] Applying electrode alignment...\n');
    EEG = pop_dipfit_settings(EEG, ... 
        'coordformat','MNI', ...
        'coord_transform',[-0.13015 -20.1331 0 0.12132 0.00027375 -1.5707 10.2 10.8 10.8], ...
        'chanomit', chan_idx);
    fprintf('  → coord_transform: [%s]\n', num2str([-0.13015 -20.1331 0 0.12132 0.00027375 -1.5707 10.2 10.8 10.8]));
    fprintf('  → Omitting %d channels: [%s]\n\n', length(chan_idx), num2str(chan_idx));

else
    % Custom headmodel (bemcp or openmeeg)
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
end

%% =========================================================================
%  STEP 3 — Run DIPFIT autofit
% ==========================================================================

num_comps = size(EEG.icaweights, 1);
fprintf('[STEP 3] Running dipole fitting on %d ICA components...\n', num_comps);

EEG = pop_multifit(EEG, 1:num_comps);

rv_values = [EEG.dipfit.model.rv];
good_ics  = find(rv_values <= RV_THRES);
poor_ics  = find(rv_values >  RV_THRES);

fprintf('  → Fitting complete.\n');
fprintf('  → RV ≤ %.2f (good fit): %d / %d — ICs: [%s]\n', RV_THRES, length(good_ics), num_comps, num2str(good_ics));
fprintf('  → RV >  %.2f (poor fit): %d / %d — ICs: [%s]\n\n', RV_THRES, length(poor_ics), num_comps, num2str(poor_ics));

if strcmp(HEADMODEL_TYPE, 'template')

    fprintf('[STEP 4] Running eeg_compatlas (DK atlas lookup)...\n');
    EEG.dipfit.coord_transform = [];
    EEG = eeg_compatlas(EEG);
    fprintf('  → eeg_compatlas complete.\n\n');

    setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_mni_', HEADMODEL_TYPE, '_headmodel'];
    filename = [setname, '.set'];
    [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', setname, 'savenew', fullfile(OUTPUT_DIR, filename), 'gui', 'off');
    fprintf('[SAVE] MNI-space standard template result saved to:\n  %s\n\n', fullfile(OUTPUT_DIR, filename));

else
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

end

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

if strcmp(HEADMODEL_TYPE, 'template')
    EEG2 = EEG;
end

fprintf('\n========================================================\n');
fprintf('  DIPOLE SUMMARY — %s  |  %s\n', SUBJ, SPEECH_TYPE);
fprintf('  RV threshold: %.2f\n', RV_THRES);
fprintf('========================================================\n');
fprintf('%-6s  %-6s  %7s %7s %7s   %-25s   %s\n', ...
    'IC', 'RV', 'X(MNI)', 'Y(MNI)', 'Z(MNI)', 'DK Region', 'AAL Region');
fprintf('%s\n', repmat('-', 1, 100));

for i = 1:length(EEG2.dipfit.model)
    rv        = EEG2.dipfit.model(i).rv;
    mni_coord = EEG2.dipfit.model(i).posxyz;

    if rv > RV_THRES,         continue; end
    if any(isnan(mni_coord)), continue; end
    if all(mni_coord == 0),   continue; end

    % DK label
    dk_region = EEG2.dipfit.model(i).areadk;
    if isempty(dk_region)
        dk_region = 'Unknown';
    end

    evalc('region = lookup_aal_region(mni_coord, atlas);');
    fprintf('%-6d  %.3f  %7.1f %7.1f %7.1f   %-25s   %s\n', ...
        i, rv, mni_coord(1), mni_coord(2), mni_coord(3), dk_region, region);
end

fprintf('%s\n', repmat('-', 1, 100));
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
