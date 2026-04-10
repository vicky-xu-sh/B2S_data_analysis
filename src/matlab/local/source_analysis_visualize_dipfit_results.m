% source_analysis_visualize_dipfit_results.m

%% =========================================================================
%  CONFIGURATION — update paths and subject info
% ==========================================================================

% CHANGE THESE
SUBJ        = 'subj-02';
RV_THRES = 0.15;

BASE_PATH     = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/data';
SPOKEN_PATH = fullfile(BASE_PATH, '05_source_analysis', SUBJ, 'spoken');
IMAGINED_PATH = fullfile(BASE_PATH, '05_source_analysis', SUBJ, 'imagined');

% Launch EEGLAB 
eeglab;
global ALLEEG EEG CURRENTSET;

fprintf('\n========================================================\n');
fprintf('  Subject:     %s\n', SUBJ);
fprintf('  Spoken data source analysis dir:   %s\n', SPOKEN_PATH);
fprintf('  Imagined data source analysis dir:  %s\n', IMAGINED_PATH);
fprintf('========================================================\n\n');

%% =========================================================================
%  Load datasets with DIPFIT results
% ==========================================================================

headmodel_types = {'bemcp', 'openmeeg'};
speech_types = {'sp', 'im'};

for cond_idx = 1:2
    if strcmp(speech_types{cond_idx}, 'sp')
        INPUT_DIR = SPOKEN_PATH;
    else
        INPUT_DIR = IMAGINED_PATH;
    end
    for hm_idx = 1:2
        % DIPFIT results in ctf space
        setname  = [SUBJ, '_pilot_', speech_types{cond_idx}, '_cleaned_2ndICA_epoched_dipfit_ctf_',headmodel_types{hm_idx},'_headmodel'];
        filename = [setname, '.set'];
        
        fprintf('Loading dataset: %s\n', filename);
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

        % DIPFIT results in MNI space
        setname  = [SUBJ, '_pilot_', speech_types{cond_idx}, '_cleaned_2ndICA_epoched_dipfit_ctf_',headmodel_types{hm_idx},'_headmodel_warptoMNI'];
        filename = [setname, '.set'];
        
        fprintf('Loading dataset: %s\n', filename);
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
    end
end


%% Run the following interactively

%% =========================================================================
%  Plot dipoles in MNI space
% ==========================================================================

% fprintf('Plotting dipoles in MNI space...\n');
% dipplot(EEG.dipfit.model(1:length(EEG.dipfit.model)), ...
%     'coordformat', 'MNI', ...
%     'transform',   [], ...
%     'mri',         mri_normalised, ...
%     'num',         'on', ...
%     'axistight',   'on', ...
%     'pointout',    'on', ...
%     'normlen',     'on');

fprintf('Plotting dipoles in MNI space...\n');
dipplot(EEG.dipfit.model(1:length(EEG.dipfit.model)), ...
    'coordformat', 'MNI', ...
    'transform',   [], ...
    'num',         'on', ...
    'axistight',   'on', ...
    'pointout',    'on', ...
    'normlen',     'on');

%% =========================================================================
%  Plot a single dipole in CTF space (sanity check vs. MNI plot)
% ==========================================================================

% EDIT: change i to the IC you want to inspect (and make sure EEG is the
% correct dataset)
i = 2;

fprintf('Plotting IC %d in CTF space (sanity check)...\n', i);
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