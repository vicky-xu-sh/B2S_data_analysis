% process_mri_create_headmodel.m

%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================

% CHANGE THESE
SUBJ        = 'subj-06';

% set env for openmeeg
setenv('PATH', [getenv('PATH') ':/Users/vickyxu/Documents/MATLAB/Toolboxes/OpenMEEG-2.4.1-MacOSX/bin/']);

% add fieldtrip path
ft_defaults;

BASE_PATH  = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/data';
RAW_PATH   = fullfile(BASE_PATH, '01_raw/', SUBJ);
OUTPUT_DIR = fullfile(BASE_PATH, '02_interim_local', SUBJ, 'custom_headmodel');

if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
    fprintf('Created output directory: %s\n', OUTPUT_DIR);
else
    fprintf('Output directory already exists: %s\n', OUTPUT_DIR);
end

%% Load raw subject MRI

fprintf('\n--- Loading raw MRI ---\n');
raw_mri_filename_pattern = fullfile(RAW_PATH, '*.nii.gz');

raw_mri_file_matches = dir(raw_mri_filename_pattern);
if ~isempty(raw_mri_file_matches)
    [~, mri_filename, ~] = fileparts(raw_mri_file_matches(1).name);
    raw_mri_file = fullfile(RAW_PATH, [mri_filename, '.gz']);
else
    error('Raw MRI .nii.gz file not found in: %s', RAW_PATH);
end

fprintf('Reading raw MRI file: %s\n', raw_mri_file);
mri = ft_read_mri(raw_mri_file);
disp(mri);

% filename = [SUBJ, '_mri.mat'];
% save(fullfile(OUTPUT_DIR, filename), 'mri');
% fprintf('Saved raw MRI as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Plot raw MRI

fprintf('\n--- Plotting raw MRI ---\n');
cfg        = [];
cfg.method = 'ortho';
cfg.dataname  = sprintf('%s: Raw MRI', SUBJ);
figure;
ft_sourceplot(cfg, mri);

%% Transform MRI to CTF coordinate system

fprintf('\n--- Realigning MRI to CTF coordinate system ---\n');

% REPLACE the fiducials with subject-specific voxel coordinates
% subj-02
% raw_nas = [113 280 175];
% raw_lpa = [22  164 135];
% raw_rpa = [203 163 135];

% subj-03
% raw_nas = [126 210 65];
% raw_lpa = [203 104 42];
% raw_rpa = [56  105 42];

% subj-04
% raw_nas = [44  129  85];
% raw_lpa = [139 106  22];
% raw_rpa = [139 117 138];

% subj-05
% raw_nas = [129 208 57];
% raw_lpa = [197 117 30];
% raw_rpa = [63 131 21];

% subj-06
raw_nas = [114 269 147];
raw_lpa = [31 159 124];
raw_rpa = [191 153 118];

% subj-07
% raw_nas = [110 277 182];
% raw_lpa = [23 162 124];
% raw_rpa = [205 165 122];


fprintf('Fiducials (voxel space):\n');
fprintf('  NAS: [%d %d %d]\n', raw_nas(1), raw_nas(2), raw_nas(3));
fprintf('  LPA: [%d %d %d]\n', raw_lpa(1), raw_lpa(2), raw_lpa(3));
fprintf('  RPA: [%d %d %d]\n', raw_rpa(1), raw_rpa(2), raw_rpa(3));

cfg              = [];
cfg.method       = 'fiducial';
cfg.fiducial.nas = raw_nas;
cfg.fiducial.lpa = raw_lpa;
cfg.fiducial.rpa = raw_rpa;
cfg.coordsys     = 'ctf';

mri_realigned = ft_volumerealign(cfg, mri);
fprintf('Realigned MRI to CTF coordinate system.\n');

% filename = [SUBJ, '_mri_realigned.mat'];
% save(fullfile(OUTPUT_DIR, filename), 'mri_realigned');
% fprintf('Saved realigned MRI as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Plot raw fiducials overlaid on MRI (in voxel space)

fprintf('\n--- Plotting fiducials in voxel space ---\n');

% Use identity transform so ft_sourceplot interprets locations as voxels. 
mri2           = mri;
mri2.transform = eye(4);

cfg2                     = [];
cfg2.method              = 'ortho';
cfg2.locationcoordinates = 'voxel';

cfg2.location = raw_nas;
cfg2.dataname    = sprintf('%s: Nasion (voxel)', SUBJ);
figure; ft_sourceplot(cfg2, mri2);

cfg2.location = raw_rpa;
cfg2.dataname    = sprintf('%s: RPA (voxel)', SUBJ);
figure; ft_sourceplot(cfg2, mri2);

cfg2.location = raw_lpa;
cfg2.dataname    = sprintf('%s: LPA (voxel)', SUBJ);
figure; ft_sourceplot(cfg2, mri2);

%% Bias correction

fprintf('\n--- Applying bias field correction ---\n');
cfg        = [];
mri_unbiased = ft_volumebiascorrect(cfg, mri_realigned);
fprintf('Bias correction complete.\n');

cfg_plot        = [];
cfg_plot.method = 'ortho';
figure;
ft_sourceplot(cfg_plot, mri_unbiased);

filename = [SUBJ, '_mri_unbiased.mat'];
save(fullfile(OUTPUT_DIR, filename), 'mri_unbiased');
fprintf('Saved bias-corrected MRI as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Segmentation

fprintf('\n--- Segmenting MRI into brain / skull / scalp ---\n');

cfg        = [];
cfg.output = {'brain', 'skull', 'scalp'};

% Uncomment and adjust thresholds only if the mesh looks wrong:
% cfg.scalpthreshold = 0.08;
% cfg.brainthreshold = 0.45;
% (skull thickness can also be adjusted inside the FieldTrip module if needed)

mri_segmented_3_compartment = ft_volumesegment(cfg, mri_unbiased);
fprintf('Segmentation complete.\n');
ft_checkdata(mri_segmented_3_compartment, 'feedback', 'yes');

% Visualise segmented compartments

fprintf('\n--- Plotting segmented compartments ---\n');
segmentedmri_indexed         = ft_checkdata(mri_segmented_3_compartment, ...
                                             'segmentationstyle', 'indexed');
segmentedmri_indexed.anatomy = mri_unbiased.anatomy;

cfg_seg              = [];
cfg_seg.method       = 'ortho';
cfg_seg.anaparameter = 'anatomy';
cfg_seg.funparameter = 'tissue';
cfg_seg.funcolormap  = [0 0 0; 1 0 0; 0 1 0; 0 0 1];  % background / scalp / skull / brain
cfg_seg.dataname     = sprintf('%s: Segmented compartments (R=scalp, G=skull, B=brain)', SUBJ);
figure;
ft_sourceplot(cfg_seg, segmentedmri_indexed);

%% Save segmentation result once everything looks good

filename = [SUBJ, '_3com_segmentedmri_from_mri_unbiased.mat'];
save(fullfile(OUTPUT_DIR, filename), 'mri_segmented_3_compartment');
fprintf('Saved 3-compartment segmented MRI as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Construct surface meshes

% Notes on choosing number of vertices: The output consists of surfaces represented 
% by points or vertices that are connected in triangles. The tissues from which the 
% surfaces are created have to be specified and also the number of vertices for each 
% tissue. Since the potential changes the most rapidly on the outside of the brain 
% (or inside of the skull), we want that surface to be the most detailed. 
% The potential does not change rapidly over the scalp, so that can remain relatively coarse. 
% It is common to use the ratio 3/2/1 for the brain/skull/scalp

fprintf('\n--- Constructing surface meshes ---\n');

cfg             = [];
cfg.tissue      = {'brain', 'skull', 'scalp'};
cfg.numvertices = [3000, 2000, 1000];

mesh = ft_prepare_mesh(cfg, mri_segmented_3_compartment);
fprintf('Mesh construction complete. Vertices — brain: %d, skull: %d, scalp: %d\n', ...
        size(mesh(1).pos,1), size(mesh(2).pos,1), size(mesh(3).pos,1));

%% Visualise meshes

fprintf('\n--- Plotting surface meshes ---\n');
figure;
ft_plot_mesh(mesh(1), 'facecolor', 'r', 'facealpha', 1.0, 'edgecolor', 'k', 'edgealpha', 1.0);
hold on;
ft_plot_mesh(mesh(2), 'facecolor', 'g', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);
ft_plot_mesh(mesh(3), 'facecolor', 'b', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);
hold off;
legend({'Brain', 'Skull', 'Scalp'}, 'Location', 'northeast');
title(sprintf('%s: Surface meshes (R=brain, G=skull, B=scalp)', SUBJ));


%% Create volume conduction model — BEM (bemcp)

fprintf('\n--- Building BEM head model (bemcp) ---\n');
cfg        = [];
cfg.method = 'bemcp';
headmodel_bemcp  = ft_prepare_headmodel(cfg, mesh);
fprintf('bemcp head model ready.\n');

figure;
ft_plot_headmodel(headmodel_bemcp, 'facealpha', 0.6);
title(sprintf('%s: Head model (bemcp)', SUBJ));

%% Create volume conduction model — BEM (openmeeg)

fprintf('\n--- Building BEM head model (openmeeg) ---\n');
cfg                = [];
cfg.method         = 'openmeeg';
headmodel_openmeeg = ft_prepare_headmodel(cfg, mesh);
fprintf('openmeeg head model ready.\n');

figure;
ft_plot_headmodel(headmodel_openmeeg, 'facealpha', 0.6);
title(sprintf('%s: Head model (openmeeg)', SUBJ));

%% Save the headmodels

filename = [SUBJ, '_headmodel_bemcp.mat'];
save(fullfile(OUTPUT_DIR, filename), 'headmodel_bemcp');
fprintf('Saved bemcp head model as: %s\n', fullfile(OUTPUT_DIR, filename));

filename = [SUBJ, '_headmodel_openmeeg.mat'];
save(fullfile(OUTPUT_DIR, filename), 'headmodel_openmeeg');
fprintf('Saved openmeeg head model as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Normalise MRI to MNI space

fprintf('\n--- Normalising MRI to MNI space ---\n');
cfg            = [];
mri_normalised = ft_volumenormalise(cfg, mri_unbiased);
fprintf('MNI normalisation complete.\n');

filename = [SUBJ, '_mri_normalised.mat'];
save(fullfile(OUTPUT_DIR, filename), 'mri_normalised');
fprintf('Saved MNI-normalised MRI as: %s\n', fullfile(OUTPUT_DIR, filename));

mri_normalised.coordsys = 'mni';
cfg_mni              = [];
cfg_mni.method       = 'ortho';
cfg_mni.anaparameter = 'anatomy';
cfg_mni.location     = [0 0 0];    % MNI origin
cfg_mni.title        = sprintf('%s: MNI-normalised MRI (origin [0 0 0])', SUBJ);
figure;
ft_sourceplot(cfg_mni, mri_normalised);

%% Helper function: transform fiducials to CTF mm space

function chanlocs = get_transformed_fid(transformed_mri, raw_nas, raw_lpa, raw_rpa)
% GET_TRANSFORMED_FID  Convert voxel fiducial coordinates to CTF mm space.
%
%   chanlocs = get_transformed_fid(transformed_mri, raw_nas, raw_lpa, raw_rpa)
%
%   Inputs:
%     transformed_mri      - ft MRI structure with a valid .transform field
%     raw_nas / lpa / rpa  - fiducial positions in MRI voxel space [1x3]
%
%   Output:
%     chanlocs - struct array with fields: labels, X, Y, Z (CTF mm)

nas_ctf = ft_warp_apply(transformed_mri.transform, raw_nas, 'homogenous');
lpa_ctf = ft_warp_apply(transformed_mri.transform, raw_lpa, 'homogenous');
rpa_ctf = ft_warp_apply(transformed_mri.transform, raw_rpa, 'homogenous');

chanlocs           = struct();
chanlocs(1).labels = 'Nasion';
chanlocs(1).X      = nas_ctf(1);
chanlocs(1).Y      = nas_ctf(2);
chanlocs(1).Z      = nas_ctf(3);

chanlocs(2).labels = 'LPA';
chanlocs(2).X      = lpa_ctf(1);
chanlocs(2).Y      = lpa_ctf(2);
chanlocs(2).Z      = lpa_ctf(3);

chanlocs(3).labels = 'RPA';
chanlocs(3).X      = rpa_ctf(1);
chanlocs(3).Y      = rpa_ctf(2);
chanlocs(3).Z      = rpa_ctf(3);

fprintf('Transformed fiducials (CTF mm space):\n');
fprintf('  Nasion: [%.2f %.2f %.2f]\n', nas_ctf(1), nas_ctf(2), nas_ctf(3));
fprintf('  LPA:    [%.2f %.2f %.2f]\n', lpa_ctf(1), lpa_ctf(2), lpa_ctf(3));
fprintf('  RPA:    [%.2f %.2f %.2f]\n', rpa_ctf(1), rpa_ctf(2), rpa_ctf(3));

end

%% Compute and save transformed fiducials

fprintf('\n--- Computing transformed fiducials (CTF mm space) ---\n');
chanlocs = get_transformed_fid(mri_unbiased, raw_nas, raw_lpa, raw_rpa);

filename = [SUBJ, '_fid_chanlocs.mat'];
save(fullfile(OUTPUT_DIR, filename), 'chanlocs');
fprintf('Saved transformed fiducials as: %s\n', fullfile(OUTPUT_DIR, filename));

%% Plot transformed fiducials overlaid on bias-corrected MRI

fprintf('\n--- Plotting transformed fiducials in CTF mm space ---\n');

cfg_fid              = [];
cfg_fid.method       = 'ortho';
cfg_fid.anaparameter = 'anatomy';

cfg_fid.location = [chanlocs(1).X, chanlocs(1).Y, chanlocs(1).Z];
cfg_fid.dataname    = sprintf('%s: Nasion (CTF mm)', SUBJ);
figure; ft_sourceplot(cfg_fid, mri_unbiased);

cfg_fid.location = [chanlocs(2).X, chanlocs(2).Y, chanlocs(2).Z];
cfg_fid.dataname    = sprintf('%s: LPA (CTF mm)', SUBJ);
figure; ft_sourceplot(cfg_fid, mri_unbiased);

cfg_fid.location = [chanlocs(3).X, chanlocs(3).Y, chanlocs(3).Z];
cfg_fid.dataname    = sprintf('%s: RPA (CTF mm)', SUBJ);
figure; ft_sourceplot(cfg_fid, mri_unbiased);

fprintf('\n=== Head model pipeline complete for %s ===\n', SUBJ);




%% =========================================================================
% Set the headmodel to be plotted below
% ==========================================================================

headmodel = headmodel_openmeeg;
HEADMODEL_TYPE = 'openmeeg';


%% =========================================================================
%  Visualize headmodel + MRI slices (sanity check — CTF space)
% ==========================================================================

fprintf('[VIZ] Plotting headmodel with MRI overlay...\n');

% --- Check required variables are in workspace ---
if ~exist('headmodel', 'var') || ~exist('mri_unbiased', 'var') 
    error('[ERROR] Missing variable — make sure headmodel, mri_unbiased are loaded.');
end

% --- Verify MRI anatomy field ---
if ~isfield(mri_unbiased, 'anatomy') || isempty(mri_unbiased.anatomy)
    error('[ERROR] mri_unbiased.anatomy is missing or empty.');
end
fprintf('  → anatomy size:  %s\n',   mat2str(size(mri_unbiased.anatomy)));
fprintf('  → MRI transform:\n');     disp(mri_unbiased.transform);

% --- Create figure and bring to front ---
fh = figure('Color', 'k', ...
            'Name',  sprintf('%s — Headmodel + MRI (%s)', SUBJ, HEADMODEL_TYPE), ...
            'Units', 'normalized', ...
            'Position', [0.1 0.1 0.75 0.75]);
figure(fh);   % raise to front
clf;          % clear in case figure handle was reused
hold on;

% --- Plot headmodel BEM surfaces ---
% Layers: [1] scalp  [2] skull  [3] brain  (order depends on your BEM)
n_layers     = length(headmodel.bnd);
layer_colors = [0.8 0.7 0.6;   % scalp — skin tone
                0.5 0.5 0.5;   % skull — grey
                0.4 0.7 0.9];  % brain — light blue
layer_alpha  = [0.12, 0.10, 0.25];
layer_names  = {'Scalp', 'Skull', 'Brain'};

for k = 1:n_layers
    ft_plot_mesh(headmodel.bnd(k), ...
        'facecolor', layer_colors(min(k, size(layer_colors,1)), :), ...
        'edgecolor', 'none', ...
        'facealpha', layer_alpha(min(k, length(layer_alpha))));
    fprintf('  → Plotted BEM layer %d / %d  (%s)\n', k, n_layers, layer_names{min(k,3)});
    drawnow;   % flush each layer so you see progress
end

% --- Normalise MRI anatomy for display ---
anatomy_disp = double(mri_unbiased.anatomy);
anatomy_disp = anatomy_disp ./ max(anatomy_disp(:));

% --- Compute slice centre in world coordinates ---
mid_vox = round(size(anatomy_disp) / 2);
mid_pos = ft_warp_apply(mri_unbiased.transform, mid_vox);
fprintf('  → Slice centre: voxel [%d %d %d]  →  world [%.1f %.1f %.1f] mm\n', ...
    mid_vox(1), mid_vox(2), mid_vox(3), mid_pos(1), mid_pos(2), mid_pos(3));

% --- Plot three orthogonal MRI slices ---
slice_orientations = {[1 0 0], [0 1 0], [0 0 1]};
slice_labels       = {'Sagittal', 'Coronal', 'Axial'};

for s = 1:3
    fprintf('  → Plotting %s slice...', slice_labels{s});
    ft_plot_slice(anatomy_disp, ...
        'transform',   mri_unbiased.transform, ...
        'location',    mid_pos, ...
        'orientation', slice_orientations{s}, ...
        'resolution',  1, ...       % increase to 2-3 if slow
        'colormap',    gray(256), ...
        'facealpha',   0.85);
    drawnow;
    fprintf(' done.\n');
end

% --- Lighting, view, interactivity ---
axis equal;
axis off;
view(45, 20);    % slightly angled starting view
rotate3d on;

drawnow;
shg;   % "show graph" — brings figure to front after all rendering

fprintf('[VIZ] Done — use mouse to rotate figure.\n\n');

