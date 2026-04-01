% source_analysis_dipfit.m
% Source localization pipeline using DIPFIT with a custom BEM headmodel.

%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================

% CHANGE THESE
SUBJ        = 'subj-02';
SPEECH_TYPE = 'im';   % 'sp' = spoken/overt | 'im' = imagined/covert
HEADMODEL_TYPE = 'openmeeg';  %'bemcp' or 'openmeeg'
RV_THRES = 0.15;

coord_transform = [-2 0 23 -0.142526 -0.00426732 0.117257 10.3 10.3 10.3];
% chanomit_idx    = [210 211 217:245]; % subj-02, spoken
chanomit_idx    = [60 66 204 205 211:240]; % subj-02, imagined

BASE_PATH     = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/data';
HEADMODEL_DIR = fullfile(BASE_PATH, '02_interim_local', SUBJ, 'custom_headmodel');
% set env for openmeeg
setenv('PATH', [getenv('PATH') ':/Users/vickyxu/Documents/MATLAB/Toolboxes/OpenMEEG-2.4.1-MacOSX/bin/']);


% Build I/O paths (mirrors cluster directory structure)
if strcmp(SPEECH_TYPE, 'sp')
    INPUT_DIR  = fullfile(BASE_PATH, '03_interim_cluster', SUBJ, 'spoken',   'datasets');
    OUTPUT_DIR = fullfile(BASE_PATH, '05_source_analysis', SUBJ, 'spoken');
else
    INPUT_DIR  = fullfile(BASE_PATH, '03_interim_cluster', SUBJ, 'imagined', 'datasets');
    OUTPUT_DIR = fullfile(BASE_PATH, '05_source_analysis', SUBJ, 'imagined');
end

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
    fprintf('[INFO] Created output directory: %s\n', OUTPUT_DIR);
else
    fprintf('[INFO] Output directory already exists: %s\n', OUTPUT_DIR);
end

% Launch EEGLAB 
eeglab;
global ALLEEG EEG CURRENTSET;

fprintf('\n========================================================\n');
fprintf('  Subject:     %s\n', SUBJ);
fprintf('  Speech type: %s\n', SPEECH_TYPE);
fprintf('  Input dir:   %s\n', INPUT_DIR);
fprintf('  Output dir:  %s\n', OUTPUT_DIR);
fprintf('========================================================\n\n');

%% =========================================================================
%  Load preprocessed and epoched dataset
% ==========================================================================

setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname, '.set'];

fprintf('[STEP 1] Loading dataset: %s\n', filename);
EEG = pop_loadset('filename', filename, 'filepath', INPUT_DIR);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw;

% Report key dataset properties after loading
fprintf('  → Loaded: %s\n',      EEG.setname);
fprintf('  → Channels:      %d\n',  EEG.nbchan);
fprintf('  → ICA components: %d\n', size(EEG.icaweights, 1));
fprintf('  → Trials:        %d\n',  EEG.trials);
fprintf('  → Epoch length:  %.0f samples  (%.3f s)\n', EEG.pnts, EEG.pnts / EEG.srate);
fprintf('  → Sampling rate: %.0f Hz\n\n', EEG.srate);

%% =========================================================================
%  DIPFIT setup
%  Use headmodel built from FieldTrip (coordinate system: CTF / SCS).
%  Interactively align EEG electrodes with headmodel on first run, then
%  hard-code the resulting coord_transform and chanomit below.
% ==========================================================================

headmodel_path = fullfile(HEADMODEL_DIR, [SUBJ, '_headmodel_',HEADMODEL_TYPE,'.mat']);
mri_path       = fullfile(HEADMODEL_DIR, [SUBJ, '_mri_unbiased.mat']);
chanlocs_path  = fullfile(HEADMODEL_DIR, [SUBJ, '_fid_chanlocs.mat']);

fprintf('[STEP 2] Loading headmodel files...\n');

% --- Load and verify headmodel ---
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
fprintf('  → Headmodel:  %s\n', headmodel_path);

% --- Load and verify MRI ---
if ~exist(mri_path, 'file')
    error('[ERROR] MRI file not found:\n  %s', mri_path);
end
load(mri_path, 'mri_unbiased');
fprintf('  → MRI:        %s\n', mri_path);

% --- Load and verify channel locations ---
if ~exist(chanlocs_path, 'file')
    error('[ERROR] Channel locations file not found:\n  %s', chanlocs_path);
end
load(chanlocs_path, 'chanlocs');
fprintf('  → Chanlocs:   %s\n\n', chanlocs_path);

% Attach headmodel info to EEG struct (keeps everything in subject-specific space)
EEG.dipfit.hdmfile    = headmodel;
EEG.dipfit.mrifile    = mri_unbiased;
EEG.dipfit.chanfile   = chanlocs;

% Apply subject-specific electrode-to-headmodel alignment.
% To re-derive this interactively, comment out pop_dipfit_settings below
% and run:  EEG = pop_dipfit_settings(EEG);
% Then copy the coord_transform and chanomit values printed to the command window.

fprintf('[STEP 2b] Applying electrode alignment...\n');

EEG = pop_dipfit_settings( EEG, ...
    'coordformat',     'SCS', ...
    'coord_transform',  coord_transform, ...
    'chanomit',         chanomit_idx );

fprintf('  → coord_transform: [%s]\n',   num2str(coord_transform));
fprintf('  → Omitting %d channels (non-scalp / bad): [%s]\n\n', ...
    length(chanomit_idx), num2str(chanomit_idx));

%% =========================================================================
%  Run DIPFIT autofit
%  pop_multifit fits a single dipole per IC in one pass (no coarse/fine
%  separation needed when using a realistic BEM headmodel).
% ==========================================================================

% Use size(...,1) to be explicit: we want the number of rows = components
num_comps = size(EEG.icaweights, 1);

fprintf('[STEP 3] Running dipole fitting on %d ICA components...\n', num_comps);

EEG = pop_multifit(EEG, 1:num_comps, 'threshold', 100);

% Quick post-fit summary: which components fit well?
rv_values    = [EEG.dipfit.model.rv];
good_ics     = find(rv_values <= RV_THRES);
poor_ics     = find(rv_values >  RV_THRES);

fprintf('  → Fitting complete.\n');
fprintf('  → RV ≤ %.2f (good fit): %d / %d components — ICs: [%s]\n', ...
    RV_THRES, length(good_ics), num_comps, num2str(good_ics));
fprintf('  → RV > %.2f (poor fit): %d / %d components — ICs: [%s]\n\n', ...
    RV_THRES, length(poor_ics), num_comps, num2str(poor_ics));

% Save CTF (subject-specific-space) dipfit result
setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_ctf_',HEADMODEL_TYPE,'_headmodel'];
filename = [setname, '.set'];
[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
    'setname',  setname, ...
    'savenew',  fullfile(OUTPUT_DIR, filename), ...
    'gui',      'off');
eeglab('redraw');
fprintf('[SAVE] CTF-space result saved to:\n  %s\n\n', fullfile(OUTPUT_DIR, filename));

%% =========================================================================
%  Warp dipole positions to MNI space
%  Two-step warp using the normalisation parameters stored in mri_normalised:
%    Step A — rigid-body (initial) transform  →  pre-aligned space
%    Step B — non-linear (SPM-style sn) warp  →  MNI space
% ==========================================================================

norm_mri_path = fullfile(HEADMODEL_DIR, [SUBJ, '_mri_normalised.mat']);

if ~exist(norm_mri_path, 'file')
    error('[ERROR] Normalised MRI file not found:\n  %s', norm_mri_path);
end
load(norm_mri_path, 'mri_normalised');
fprintf('[STEP 4] Warping dipoles to MNI space...\n');
fprintf('  → Normalised MRI: %s\n', norm_mri_path);

% Copy EEG struct so we keep the original CTF-space dipfit intact
EEG2       = EEG;
model_mni  = EEG.dipfit.model;

for i = 1:length(model_mni)
    if ~isempty(model_mni(i).posxyz)

        pos_ctf = model_mni(i).posxyz;

        % Flag suspicious input before doing anything
        if any(isnan(pos_ctf)) || all(pos_ctf == 0)
            fprintf('  [WARNING] IC %02d — suspicious CTF posxyz before warp: [%.2f %.2f %.2f]  rv=%.3f\n', ...
                i, pos_ctf(1), pos_ctf(2), pos_ctf(3), model_mni(i).rv);
        end

        % Step A: rigid-body alignment (subject → pre-aligned space)
        pos_prealigned = ft_warp_apply(mri_normalised.initial, pos_ctf, 'homogeneous');

        if any(isnan(pos_prealigned))
            fprintf('  [WARNING] IC %02d — NaN after Step A (rigid-body).  CTF in: [%.2f %.2f %.2f]\n', ...
                i, pos_ctf(1), pos_ctf(2), pos_ctf(3));
        end

        % Step B: non-linear SPM warp (pre-aligned → MNI)
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

% Update EEG2 to reflect MNI space — clear CTF-specific fields to avoid confusion
EEG2.dipfit.model         = model_mni;
EEG2.dipfit.coordformat   = 'MNI';
EEG2.dipfit.coord_transform = [];   % must be empty so eeg_compatlas uses raw MNI coords
EEG2.dipfit.mrifile       = mri_normalised;
EEG2.dipfit.hdmfile       = [];     % headmodel is in CTF space — not valid for MNI plotting

fprintf('[STEP 4b] Running eeg_compatlas (DK atlas lookup)...\n');
EEG2 = eeg_compatlas(EEG2);
fprintf('  → eeg_compatlas complete.\n\n');

%% =========================================================================
%  Save MNI-space result
% ==========================================================================

setname  = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_ctf_', HEADMODEL_TYPE,'_headmodel_warptoMNI'];
filename = [setname, '.set'];

[ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG2, CURRENTSET, ...
    'setname',  setname, ...
    'savenew',  fullfile(OUTPUT_DIR, filename), ...
    'gui',      'off');
eeglab('redraw');
fprintf('[SAVE] MNI-space result saved to:\n  %s\n\n', fullfile(OUTPUT_DIR, filename));

%% =========================================================================
%  Load AAL atlas 
% ==========================================================================
fprintf('[STEP 5] Loading AAL atlas...\n');
% Load FieldTrip AAL atlas for ft_volumelookup
[ftver, ftpath] = ft_version;
aal_path = fullfile(ftpath, 'template', 'atlas', 'aal', 'ROI_MNI_V4.nii');
if ~exist(aal_path, 'file')
    error('[ERROR] AAL atlas not found at expected FieldTrip path:\n  %s', aal_path);
end
atlas = ft_read_atlas(aal_path);
fprintf('  → AAL atlas loaded (FieldTrip v%s)\n\n', ftver);

%% =========================================================================
%  Summary table: well-fitted dipoles with MNI coords and AAL labels
% ==========================================================================

fprintf('\n========================================================\n');
fprintf('  DIPOLE SUMMARY — %s  |  %s\n', SUBJ, SPEECH_TYPE);
fprintf('  RV threshold: %.2f\n', RV_THRES);
fprintf('========================================================\n');
fprintf('%-6s  %-6s  %7s %7s %7s   %s\n', 'IC', 'RV', 'X(MNI)', 'Y(MNI)', 'Z(MNI)', 'AAL Region');
fprintf('%s\n', repmat('-', 1, 65));

for i = 1:length(EEG2.dipfit.model)
    rv = EEG2.dipfit.model(i).rv;

    mni_coord = EEG2.dipfit.model(i).posxyz;
    if rv > RV_THRES,                  continue; end   % poor fit
    if any(isnan(mni_coord)),          continue; end   % NaN from warp
    if all(mni_coord == 0),            continue; end   % failed fit

    % Suppress ft_volumelookup console chatter
    evalc('region = lookup_aal_region(mni_coord, atlas);');

    fprintf('%-6d  %.3f  %7.1f %7.1f %7.1f   %s\n', ...
        i, rv, mni_coord(1), mni_coord(2), mni_coord(3), region);
end

fprintf('%s\n', repmat('-', 1, 65));
fprintf('========================================================\n\n');

fprintf('[DONE] source_analysis_dipfit.m completed successfully.\n');

%% =========================================================================
%  Helper: AAL region lookup via FieldTrip
% ==========================================================================

function region = lookup_aal_region(mni_coord, atlas)
    cfg        = [];
    cfg.roi    = mni_coord;
    cfg.atlas  = atlas;
    cfg.output = 'single';
    cfg.radius = 5;   % search within 5 mm — catches dipoles landing in white matter gaps
    labels = ft_volumelookup(cfg, atlas);
    [maxcount, idx] = max(labels.count);
    if isempty(idx) || maxcount == 0
        region = 'Outside atlas';
    else
        region = labels.name{idx};
    end
end





%% Run the following interactively

%% =========================================================================
%  Plot dipoles in MNI space
% ==========================================================================

fprintf('[STEP 6] Plotting dipoles in MNI space...\n');
dipplot(EEG2.dipfit.model(1:length(EEG2.dipfit.model)), ...
    'coordformat', 'MNI', ...
    'transform',   [], ...
    'mri',         mri_normalised, ...
    'num',         'on', ...
    'axistight',   'on', ...
    'pointout',    'on', ...
    'normlen',     'on');

%% =========================================================================
%  Plot a single dipole in CTF space (sanity check vs. MNI plot)
% ==========================================================================

% EDIT: change i to the IC you want to inspect (and make sure EEG is the
% correct dataset)
i = 4;

fprintf('[STEP 6b] Plotting IC %d in CTF space (sanity check)...\n', i);
figure; hold on;
ft_plot_dipole(EEG.dipfit.model(i).posxyz, EEG.dipfit.model(i).momxyz, 'color', 'g', 'unit', 'mm');

pos = EEG.dipfit.model(i).posxyz + 2;  % small offset so slices sit just behind dipole
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, ...
    'location', pos, 'orientation', [1 0 0], 'resolution', 0.1);
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, ...
    'location', pos, 'orientation', [0 1 0], 'resolution', 0.1);
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, ...
    'location', pos, 'orientation', [0 0 1], 'resolution', 0.1);
ft_plot_crosshair(pos, 'color', [1 1 1]/2);

axis tight; axis off;
view(12, -10);
title(sprintf('IC %d — CTF space (rv = %.3f)', i, EEG.dipfit.model(i).rv));

%% Plot atlas as slices

% First interpolate the atlas onto the normalised MRI grid
cfg            = [];
cfg.parameter  = 'tissue';
atlas_interp   = ft_sourceinterpolate(cfg, atlas, mri_normalised);

% Plot
cfg              = [];
cfg.method       = 'slice';
cfg.funparameter = 'tissue';      % atlas regions as coloured overlay
cfg.anaparameter = 'anatomy';     % normalised MRI as grey background
cfg.nslices      = 20;            % number of slices to show
cfg.slicerange   = 'auto';
ft_sourceplot(cfg, atlas_interp);


%% =========================================================================
%  3D rotatable atlas overlay on normalised MRI (for sanity check)
% ==========================================================================
atlas_aal = atlas;
% Interpolate atlas onto the normalised MRI grid (must share the same space)
cfg           = [];
cfg.parameter = 'tissue';
atlas_interp  = ft_sourceinterpolate(cfg, atlas_aal, mri_normalised);

% Pick regions to show — plot all, or restrict to a subset
% e.g. speech-motor regions only:
%   regions_to_plot = {'Precentral_L','Precentral_R','Supp_Motor_Area_L',...
%                      'Supp_Motor_Area_R','Rolandic_Oper_L','Rolandic_Oper_R',...
%                      'Insula_L','Insula_R','Temporal_Sup_L','Temporal_Sup_R'};
% For all regions:
regions_to_plot = atlas_aal.tissuelabel;

n_regions = length(regions_to_plot);
cmap      = lines(n_regions);   % distinct colour per region

figure('Color', 'k', 'Name', 'AAL Atlas — 3D');
hold on;

for r = 1:n_regions

    % Find index of this label in the atlas
    label_idx = find(strcmp(atlas_aal.tissuelabel, regions_to_plot{r}));
    if isempty(label_idx), continue; end

    % Binary mask for this region in interpolated volume
    mask = (atlas_interp.tissue == label_idx);
    if sum(mask(:)) < 10, continue; end   % skip tiny/empty regions

    % Smooth the mask slightly so isosurface is cleaner
    mask_smooth = smooth3(double(mask), 'gaussian', 3);

    % Extract surface
    [faces, verts] = isosurface(mask_smooth, 0.3);
    if isempty(faces), continue; end

    % Convert voxel indices → MRI world coordinates (mm)
    % isosurface returns verts in [col row slice] order — permute to [x y z]
    verts_xyz  = [verts(:,2), verts(:,1), verts(:,3)];   % swap col/row
    verts_hom  = [verts_xyz, ones(size(verts_xyz,1), 1)]';
    verts_mni  = (mri_normalised.transform * verts_hom)';
    verts_mni  = verts_mni(:, 1:3);

    patch('Faces',     faces, ...
          'Vertices',  verts_mni, ...
          'FaceColor', cmap(r,:), ...
          'EdgeColor', 'none', ...
          'FaceAlpha', 0.25);    % transparency: lower = more see-through
end

%% Add MRI slice planes as anatomical reference
% Three orthogonal slices through the MRI centre
mid = round(size(mri_normalised.anatomy) / 2);

% Build a quick helper to plot one slice as a texture
anatomy_norm = double(mri_normalised.anatomy);
anatomy_norm = anatomy_norm / max(anatomy_norm(:));   % normalise to [0,1]

function plot_mri_slice(anatomy, transform, dim, slice_idx)
    sz = size(anatomy);
    switch dim
        case 1   % sagittal
            slice = squeeze(anatomy(slice_idx, :, :));
            [Y, Z] = meshgrid(1:sz(2), 1:sz(3));
            X      = repmat(slice_idx, size(Y));
        case 2   % coronal
            slice = squeeze(anatomy(:, slice_idx, :));
            [X, Z] = meshgrid(1:sz(1), 1:sz(3));
            Y      = repmat(slice_idx, size(X));
        case 3   % axial
            slice = squeeze(anatomy(:, :, slice_idx));
            [X, Y] = meshgrid(1:sz(1), 1:sz(2));
            Z      = repmat(slice_idx, size(X));
    end
    % Transform voxel coords to MRI world coords
    coords     = [X(:), Y(:), Z(:), ones(numel(X),1)]';
    coords_mni = (transform * coords)';
    Xw = reshape(coords_mni(:,1), size(X));
    Yw = reshape(coords_mni(:,2), size(Y));
    Zw = reshape(coords_mni(:,3), size(Z));
    surf(Xw, Yw, Zw, repmat(slice', [1 1 3]), ...   % RGB grey
        'EdgeColor', 'none', 'FaceAlpha', 0.6);
    colormap(gray);
end

plot_mri_slice(anatomy_norm, mri_normalised.transform, 1, mid(1));  % sagittal
plot_mri_slice(anatomy_norm, mri_normalised.transform, 2, mid(2));  % coronal
plot_mri_slice(anatomy_norm, mri_normalised.transform, 3, mid(3));  % axial

%% Overlay dipoles as spheres
for i = length(EEG2.dipfit.model)
    pos = EEG2.dipfit.model(i).posxyz;
    if all(pos == 0) || any(isnan(pos)), continue; end

    % Draw a sphere at dipole location
    [sx, sy, sz] = sphere(12);
    r_sphere     = 4;   % sphere radius in mm
    surf(sx*r_sphere + pos(1), sy*r_sphere + pos(2), sz*r_sphere + pos(3), ...
        'FaceColor', 'w', 'EdgeColor', 'none', 'FaceAlpha', 0.9);

    % Label
    text(pos(1)+5, pos(2), pos(3), sprintf('IC%d', i), ...
        'Color', 'w', 'FontSize', 8, 'FontWeight', 'bold');
end

%% Lighting and view
lighting gouraud;
material dull;
camlight('headlight');
light('Position', [ 1  1  1], 'Style', 'infinite');
light('Position', [-1 -1 -1], 'Style', 'infinite');

axis equal;
axis off;
view(3);          % default 3D perspective
rotate3d on;      % enable mouse rotation

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title(sprintf('AAL Atlas — %s | %s', SUBJ, SPEECH_TYPE), ...
    'Color', 'w', 'FontSize', 12);