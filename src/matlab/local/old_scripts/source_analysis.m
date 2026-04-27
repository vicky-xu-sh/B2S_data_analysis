% Source analysis

eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

% CHANGE THIS
SPEECH_TYPE = 'sp';
SUBJ = 'subj-02';

if SPEECH_TYPE == 'sp'
    dataset_path = [dataset_path,'/',SUBJ,'/spoken'];  % make sure datapath exists
else 
    dataset_path = [dataset_path,'/',SUBJ,'/imagined'];
end


if exist(dataset_path, 'dir') == 7
    disp('Dataset path exists.');
else
    disp('Dataset path does not exist.');
end

% MRI images and headmodel path

ft_defaults % fieldtrip

custom_headmodel_path = ['/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets/',SUBJ,'/custom_headmodel'];

if exist(custom_headmodel_path, 'dir') == 7
    disp('Headmodel path exists.');
else
    disp('Headmodel path does not exist.');
end

%% Load epoched dataset

setname = [SUBJ,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname, '.set'];
EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
% updates data structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw; % refresh GUI

%% Create custom head model from MRI (inside DIPFIT), Run DIPFIT autofit (instead of coaurse fit + fine fit)

% EEG = pop_dipfit_headmodel(EEG, '/Users/vickyxu/Desktop/fsl_files_subj-02/T1_biascorr_to_std_sub.nii', 'datatype','EEG','plotfiducial',{'nasion','lpa','rpa'});
% subj-03 spoken
% EEG = pop_dipfit_settings( EEG, 'coordformat','MNI','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[162:184] );
% subj-03 imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','MNI','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[67 182 183 187 188:216] );


%% Use headmodel built from fieldtrip (coordinate system of realigned or resliced/unbiased mri and headmodel should be CTF)

headmodel_path = [custom_headmodel_path, '/', SUBJ, '_headmodel_bemcp.mat'];
mri_path = [custom_headmodel_path, '/', SUBJ, '_mri_unbias.mat']; % CHANGE
chanlocs_path = [custom_headmodel_path, '/', SUBJ, '_fid_chanlocs.mat'];

load(headmodel_path, 'headmodel');
load(mri_path, 'mri_unbias'); mri = mri_unbias;
load(chanlocs_path, 'chanlocs');

EEG.dipfit.hdmfile = headmodel;
EEG.dipfit.mrifile = mri;
EEG.dipfit.chanfile = chanlocs;
EEG.dipfit.coordformat = 'ctf'; % or 'SCS' as Brainstorm calls it
EEG.dipfit.coord_transform = [];

% EEG = pop_dipfit_settings(EEG); % interactive window

% subj-02 spoken
EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform', [-2 0 23 -0.142526 -0.00426732 0.117257 10.3 10.3 10.3 ] ,'chanomit', [210 211 217:245] );
% OLDEEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[-1.28 -0.6 23 -0.1585 0.1 0.18 10 10 10] ,'chanomit', [210 211 217:245] );

% subj-02 imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[-1.28 -0.6 23 -0.1585 0.1 0.18 10 10 10] ,'chanomit', [204 205 211:240] );

% subj-03 spoken
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[162:184] );

% subj-03 imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[67 182 183 187 188:216] );

%% Standard template headmodel

% subj-02 spoken
EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform', ...
    [1.5982 -19.98 -11.8991 0.05 0.1428 -1.454 11.0134 11.0134 11.0134] ,'chanomit',[210 211 217 218:245] );

% subj-02 imagined
% EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform', ...
%     [1.5982 -19.98 -11.8991 0.05 0.1428 -1.454 11.0134 11.0134 11.0134] ,'chanomit',[204 205 211:240] );


%% Run DIPFIT autofit (instead of coaurse fit + fine fit)

% CHANGE THIS
num_comps = 32;

EEG = pop_multifit(EEG, [1:num_comps] ,'threshold',100);

% Save dipole fitted set (CHANGE)
setname = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_standard']; % if using template/standard headmodel
% setname = [SUBJ, '_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf'];

filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI


%% Plot individual dipoles 
% This should be in subject ctf space, however, there might be bugs in
% dipplot that it only plots in MNI. (Need to further verify)

% % Plotting dipoles with residual var <= 30%, pointing out, normalized length
% pop_dipplot( EEG, [1:num_comps],'rvrange',[0 30],'num','on','axistight','on','pointout','on','normlen','on');

% % Using a different way to call the function (and including the headmesh,
% % but dipplot has bugs in plotting mesh)
% 
% dipplot(EEG.dipfit.model([1:num_comps]), 'transform', [], ...
%     'mri',EEG.dipfit.mrifile,'meshdata', EEG.dipfit.hdmfile.bnd(3), ...
%     'num','on','axistight','on','pointout','on','normlen','on', 'mesh', 'on');


%% Warp dipole results to MNI (explore results in MNI)

% load normalised mri
norm_mri_path = [custom_headmodel_path, '/', SUBJ, '_mri_normalised.mat'];
load(norm_mri_path, 'mri_normalised');

model_mni = EEG.dipfit.model;

for i = 1:length(model_mni)
    if ~isempty(model_mni(i).posxyz)
        % % Apply the non-linear warp using the params from ft_volumenormalise
        % model_mni(i).posxyz = ft_warp_apply(mri_normalised.params, model_mni(i).posxyz, 'individual2sn');

        % STEP A: Apply the initial rigid-body alignment
        % This moves the dipole from subject space to the pre-aligned space
        pos_prealigned = ft_warp_apply(mri_normalised.initial, model_mni(i).posxyz, 'homogeneous');
        
        % STEP B: Apply the non-linear warp
        % This moves it from pre-aligned space to MNI space
        model_mni(i).posxyz = ft_warp_apply(mri_normalised.params, pos_prealigned, 'individual2sn');
    end
end


%% Look up the atlas (atlas only supports case where dipole results are in MNI or spherical)

EEG2 = EEG;
EEG2.dipfit.model = model_mni;
EEG2.dipfit.coordformat = 'MNI';
EEG2.dipfit.coord_transform = [];
EEG2.dipfit.mrifile = mri_normalised;
EEG2.dipfit.hdmfile = []; % the headmodel is not in MNI space (remove that to avoid future confusion)

EEG2 = eeg_compatlas(EEG2);

% Plot the dipoles (in MNI)
dipplot(EEG2.dipfit.model([1:length(EEG2.dipfit.model)]), 'coordformat', 'MNI', 'transform', [], 'mri', mri_normalised,'num','on','axistight','on','pointout','on','normlen','on');

%% Save MNI space result 

setname = [SUBJ,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf_warptoMNI'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG2, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI


%% Plot summary (all dipoles in one snapshot)

% ICs_to_plot = [4,7,11,12,13,14,16];
% ICs_to_plot = [2,5,8:14];
% ICs_to_plot = [3,10,12,14,15,17,19,20];
% 
% pop_dipplot( EEG, ICs_to_plot, ...
%     'summary','on','num','on', ...
%     'drawedges','on','cornermri','on', ...
%     'axistight','on','pointout','on','normlen','on');

%% Print the DK atlas lookup results for dipole location

% ICs = [4 7 11 12:14 16];
ICs = [2:3:8 9:14];
for i = ICs
    fprintf('Comp %d %s\n', i, EEG.dipfit.model(i).areadk)
end

%% Plot dipoles using ft_plot_dipole

figure
hold on

i = 3;
ft_plot_dipole(EEG.dipfit.model(i).posxyz, EEG.dipfit.model(i).momxyz, 'color', 'g', 'unit', 'mm')

pos = EEG.dipfit.model(i).posxyz + 2;
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, 'location', pos, 'orientation', [1 0 0], 'resolution', 0.1)
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, 'location', pos, 'orientation', [0 1 0], 'resolution', 0.1)
ft_plot_slice(EEG.dipfit.mrifile.anatomy, 'transform', EEG.dipfit.mrifile.transform, 'location', pos, 'orientation', [0 0 1], 'resolution', 0.1)

ft_plot_crosshair(pos, 'color', [1 1 1]/2);

axis tight
axis off

view(12, -10)

%% AAL atlas lookup

function region = lookup_aal_region(mni_coord, atlas)
    cfg        = [];
    cfg.roi    = mni_coord;
    cfg.atlas  = atlas;
    cfg.output = 'single';
    labels = ft_volumelookup(cfg, atlas);
    [~, idx] = max(labels.count);
    if isempty(idx) || labels.count(idx) == 0
        region = 'Outside atlas';
    else
        region = labels.name{idx};
    end
end

[ftver, ftpath] = ft_version;
atlas = ft_read_atlas([ftpath '/template/atlas/aal/ROI_MNI_V4.nii']);

fprintf('\n--- Dipole AAL Lookup ---\n');
for i = 1:length(EEG.dipfit.model)

    if EEG.dipfit.model(i).rv > 0.15
        fprintf('IC %02d → poor fit (rv=%.2f), skipped\n', ...
            i, EEG.dipfit.model(i).rv);
        continue
    end

    mni_coord = EEG.dipfit.model(i).posxyz;

    % Suppress all ft_volumelookup console output
    evalc('region = lookup_aal_region(mni_coord, atlas);');

    fprintf('IC %02d | rv=%.2f | [%6.1f %6.1f %6.1f] → %s\n', ...
        i, EEG.dipfit.model(i).rv, ...
        mni_coord(1), mni_coord(2), mni_coord(3), region);
end









%% Trying fieldtrip source analysis

% Step 1: Convert EEGLAB → FieldTrip
data_ft = eeglab2fieldtrip(EEG, 'preprocessing', 'none');
%% 

% Step 2: Load your subject-specific head model
headmodel_path = [custom_headmodel_path, '/', SUBJ, '_headmodel_bemcp.mat'];
load(headmodel_path, 'headmodel');

% headmodel_path = [custom_headmodel_path, '/', SUBJ, '_headmodel_openmeeg.mat'];
% load(headmodel_path, 'headmodel_openmeeg');
% headmodel = headmodel_openmeeg;

%% Step 3: Get electrode positions in FieldTrip format

% Convert with fiducials included
elec_raw = eeglab2fieldtrip(EEG, 'chanloc_withfid');
elec     = elec_raw.elec;   % unwrap nested structure

% Separate EEG channels from fiducials
fid_labels = {'Nz', 'LPA', 'RPA'};
fid_idx    = ismember(elec.label, fid_labels);
eeg_idx    = ~fid_idx;

% EEG electrodes
elec_eeg          = elec;
elec_eeg.elecpos  = elec.elecpos(eeg_idx, :);
elec_eeg.pnt      = elec.pnt(eeg_idx, :);
elec_eeg.label    = elec.label(eeg_idx);

% Fiducials
elec_fid          = elec;
elec_fid.elecpos  = elec.elecpos(fid_idx, :);
elec_fid.pnt      = elec.pnt(fid_idx, :);
elec_fid.label    = elec.label(fid_idx);

% Fix units: EGI exports in cm, FieldTrip headmodel is in mm
elec_eeg.elecpos  = elec_eeg.elecpos * 10;
elec_eeg.pnt      = elec_eeg.pnt * 10;
elec_fid.elecpos  = elec_fid.elecpos * 10;
elec_fid.pnt      = elec_fid.pnt * 10;

% Add required FieldTrip fields
elec_eeg.unit     = 'mm';
elec_eeg.coordsys = 'ctf';

fprintf('EEG electrodes: %d\n', sum(eeg_idx));
fprintf('Fiducials:      %d\n', sum(fid_idx));

% EEG electrode position ranges
fprintf('X range: %.1f to %.1f mm\n', min(elec_eeg.elecpos(:,1)), max(elec_eeg.elecpos(:,1)));
fprintf('Y range: %.1f to %.1f mm\n', min(elec_eeg.elecpos(:,2)), max(elec_eeg.elecpos(:,2)));
fprintf('Z range: %.1f to %.1f mm\n', min(elec_eeg.elecpos(:,3)), max(elec_eeg.elecpos(:,3)));

% Also check headmodel scalp vertex ranges
fprintf('\nScalp X range: %.1f to %.1f mm\n', min(headmodel.bnd(end).pos(:,1)), max(headmodel.bnd(end).pos(:,1)));
fprintf('Scalp Y range: %.1f to %.1f mm\n', min(headmodel.bnd(end).pos(:,2)), max(headmodel.bnd(end).pos(:,2)));
fprintf('Scalp Z range: %.1f to %.1f mm\n', min(headmodel.bnd(end).pos(:,3)), max(headmodel.bnd(end).pos(:,3)));


%% Realign electrodes

% Rename to FieldTrip standard if needed
elec_fid.label = {'Nasion', 'LPA', 'RPA'};

% Load your previously saved CTF fiducials
load([custom_headmodel_path, '/', SUBJ, '_fid_chanlocs.mat'], 'chanlocs');

% Build FieldTrip-compatible fiducial struct
headshape_fid.label = {'Nasion'; 'LPA'; 'RPA'};
headshape_fid.pos   = [
    chanlocs(1).X  chanlocs(1).Y  chanlocs(1).Z;   % Nasion
    chanlocs(2).X  chanlocs(2).Y  chanlocs(2).Z;   % LPA
    chanlocs(3).X  chanlocs(3).Y  chanlocs(3).Z;   % RPA
];

% Combine EEG electrodes with fiducial labels for alignment
elec_withfid          = elec_eeg;
elec_withfid.elecpos  = [elec_eeg.elecpos;  elec_fid.elecpos];   % numeric → ; is fine
elec_withfid.pnt      = [elec_eeg.pnt;      elec_fid.pnt];       % numeric → ; is fine
elec_withfid.label    = [elec_eeg.label(:); elec_fid.label(:)];  % cell → need (:)

% Align using fiducials
cfg                  = [];
cfg.method           = 'fiducial';
cfg.target           = headshape_fid;
cfg.elec             = elec_withfid;

% FieldTrip expects fiducial fields to contain
% the LABEL STRINGS that match entries in elec.label
cfg.fiducial         = {'Nasion', 'LPA', 'RPA'};  % {nas, lpa, rpa} order

elec_aligned = ft_electroderealign(cfg);

%% Project onto scalp surface

cfg             = [];
cfg.method      = 'project';
cfg.headshape   = headmodel.bnd(end);   % scalp = outermost boundary
cfg.elec        = elec_aligned;
elec_projected  = ft_electroderealign(cfg);


%% Visualize with all head model layers + electrodes
figure;

% Plot all compartments with different colors and transparency
% headmodel.bnd order matches your tissuelabel: csf, gray, scalp, skull, white
% We want brain (gray/white) and scalp at minimum

% Scalp — outermost
ft_plot_mesh(headmodel.bnd(end), 'facealpha', 0.1, ...
    'edgecolor', 'none', 'facecolor', [1 1 0]);   % yellow
hold on;

% Skull
ft_plot_mesh(headmodel.bnd(end-1), 'facealpha', 0.1, ...
    'edgecolor', 'none', 'facecolor', [0 1 1]);   % cyan

% Brain (innermost — CSF or gray depending on your model)
ft_plot_mesh(headmodel.bnd(1), 'facealpha', 0.4, ...
    'edgecolor', 'none', 'facecolor', [0.8 0.2 0.2]);  % red

% Electrodes
ft_plot_sens(elec_projected, 'style', '.k', 'label', 'on');

% Fiducials
plot3(headshape_fid.pos(1,1), headshape_fid.pos(1,2), headshape_fid.pos(1,3), ...
    'g*', 'MarkerSize', 20, 'LineWidth', 3);   % Nasion
plot3(headshape_fid.pos(2,1), headshape_fid.pos(2,2), headshape_fid.pos(2,3), ...
    'b*', 'MarkerSize', 20, 'LineWidth', 3);   % LPA
plot3(headshape_fid.pos(3,1), headshape_fid.pos(3,2), headshape_fid.pos(3,3), ...
    'b*', 'MarkerSize', 20, 'LineWidth', 3);   % RPA

title('Full head model + electrodes');
legend('Scalp', 'Skull', 'Brain', 'Electrodes', 'Nasion', 'LPA/RPA');
view(135, 30);   % 3D perspective view

%% Create dipole source model

cfg = [];
cfg.resolution  = 7.5;
cfg.threshold   = 0.1;
cfg.smooth      = 5;
cfg.headmodel   = headmodel;
cfg.inwardshift = 1; % shifts dipoles away from surfaces
sourcemodel = ft_prepare_sourcemodel(cfg);

%% Visualize source model
figure;

% Plot source grid points (inside brain only)
ft_plot_mesh(sourcemodel.pos(sourcemodel.inside,:), ...
    'vertexsize', 10, ...
    'vertexcolor', 'r');
hold on;

% Plot scalp surface for context
ft_plot_mesh(headmodel.bnd(end), ...
    'vertexcolor', 'none', ...
    'facecolor', [1 1 0], ...
    'facealpha', 0.2, ...
    'edgealpha', 0.1);

% Plot brain surface (innermost boundary)
ft_plot_mesh(headmodel.bnd(1), ...
    'vertexcolor', 'none', ...
    'facecolor', [0.8 0.2 0.2], ...
    'facealpha', 0.2, ...
    'edgealpha', 0.1);

title(sprintf('Source model: %d inside / %d total grid points', ...
    sum(sourcemodel.inside), length(sourcemodel.inside)));
view(135, 30);
lighting gouraud;
camlight;

%% 

filename = [SUBJ,'_sourcemodel_openmeeg.mat'];
save(fullfile(custom_headmodel_path, filename), 'sourcemodel');

%% Compute leadfield

cfg                 = [];
cfg.sourcemodel     = sourcemodel;
cfg.headmodel       = headmodel;
cfg.elec            = elec_projected;   % aligned electrodes
cfg.reducerank      = 3;               
leadfield_bem = ft_prepare_leadfield(cfg);

% Sanity check output
fprintf('Leadfield computed for %d grid points\n', sum(leadfield_bem.inside));
fprintf('Leadfield matrix size per point: %d x %d\n', ...
    size(leadfield_bem.leadfield{find(leadfield_bem.inside,1)}, 1), ...
    size(leadfield_bem.leadfield{find(leadfield_bem.inside,1)}, 2));


%% 

filename = [SUBJ,'_leadfield_openmeeg.mat'];
save(fullfile(custom_headmodel_path, filename), 'leadfield_bem');


%% Get component into fieldtrip struct

comp = eeglab2fieldtrip(EEG, 'comp');


%% Define face/neck channels to exclude
exclude_chans = [218, 219, 225:240, 241:255];
exclude_labels = arrayfun(@(x) sprintf('E%d', x), exclude_chans, 'UniformOutput', false);

% Remove from elec_eeg.label
keep_idx = ~ismember(elec_eeg.label, exclude_labels);
clean_labels = elec_eeg.label(keep_idx);

fprintf('Channels before exclusion: %d\n', length(elec_eeg.label));
fprintf('Channels excluded:         %d\n', sum(~keep_idx));
fprintf('Channels remaining:        %d\n', length(clean_labels));

%% Dipole fitting for IC 03

cfg                 = [];
cfg.numdipoles      = 1;
cfg.headmodel       = headmodel;
cfg.sourcemodel     = leadfield_bem;    % use precomputed leadfield
cfg.elec            = elec_projected;
cfg.nonlinear       = 'yes';            % nonlinear optimization
cfg.component       = 3;               % IC 03 specifically
cfg.channel         = clean_labels;

dip = ft_dipolefitting(cfg, comp);

fprintf('IC 03 dipole position (CTF mm): [%.1f %.1f %.1f]\n', dip.dip.pos);
fprintf('Residual variance: %.3f\n', dip.dip.rv);

%% Plot dipole location

figure;
ft_plot_mesh(headmodel.bnd(1));
alpha 0.7;
ft_plot_dipole(dip.dip.pos(1,:), mean(dip.dip.mom(1:3,:),2), 'color', 'b','unit','mm')


%% Try Minimum norm estimation

% Extract IC03 from comp struct and make timelock

ic_num = 3;
n_trials = length(comp.trial);

comp_sensor.trial   = cell(1, n_trials);
comp_sensor.time    = comp.time;
comp_sensor.label   = elec_eeg.label;  
comp_sensor.fsample = EEG.srate;

for t = 1:n_trials
    comp_sensor.trial{t} = comp.topo(:, ic_num) * comp.trial{t}(ic_num, :);
end

% Verify dimensions are consistent
fprintf('Label count:       %d\n', length(comp_sensor.label));
fprintf('Trial chan count:   %d\n', size(comp_sensor.trial{1}, 1));
fprintf('Timepoints:       %d\n', size(comp_sensor.trial{1}, 2));

%% Timelock analysis
cfg                     = [];
cfg.covariance          = 'yes';
cfg.covariancewindow    = 'prestim';
timelock_ic03 = ft_timelockanalysis(cfg, comp_sensor);

%% MNE source analysis

cfg                     = [];
cfg.method              = 'mne';                    % specify minimum norm estimate as method
cfg.latency             = [0.024 0.026];            % latency of interest
cfg.grid                = leadfield_bem;            % the precomputed leadfield
cfg.headmodel           = headmodel;            
cfg.mne.prewhiten       = 'yes';                    % prewhiten data
cfg.mne.lambda          = 3;                        % regularisation parameter
cfg.mne.scalesourcecov  = 'yes';                    % scaling the source covariance matrix
cfg.channel             = clean_labels;
minimum_norm_bem        = ft_sourceanalysis(cfg, timelock_ic03);

%% 

mri_filename = [SUBJ,'_mri_unbias.mat'];
load(fullfile(custom_headmodel_path, mri_filename), 'mri_unbias');

cfg            = [];
cfg.parameter  = 'avg.pow';
interpolate    = ft_sourceinterpolate(cfg, minimum_norm_bem , mri_unbias);

cfg = [];
cfg.method        = 'ortho';
cfg.funparameter  = 'pow';
ft_sourceplot(cfg, interpolate);





%% Save set to contain only brain ICs

% Find the Brain ICs (dipfit RV <15% + >50% prob being brain)
% Get all residual variances as a vector
allRV = [EEG.dipfit.model.rv];

% Find indices where RV < 0.15
goodDipoleIdx = find(allRV < 0.15);

% Display the indices
disp('Component indices with RV < 0.15:');
disp(goodDipoleIdx);

% brain ICs to have prob > 0.5
ic_probs = EEG.etc.ic_classification.ICLabel.classifications;
all_prob_brain_ICs = find(ic_probs(:,1) > 0.5);

disp('Component indices brain prob > 0.5');
disp(all_prob_brain_ICs);

% find the intersection
ICs_to_keep = intersect(goodDipoleIdx, all_prob_brain_ICs);
disp('Components to keep');
disp(ICs_to_keep)

all_ICs = 1:length([EEG.dipfit.model.rv]);
ICs_to_reject = setdiff(all_ICs, ICs_to_keep);

% Mark non-brain ICs for rejection
EEG.reject.gcompreject(ICs_to_reject) = 1;

EEG = pop_subcomp(EEG, [], 1);
EEG = pop_reref( EEG, []);

setname = [SUBJ,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_brain_only'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI