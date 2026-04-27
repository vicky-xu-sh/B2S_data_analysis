%% Load path

ft_defaults

custom_headmodel_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

SUBJ = 'subj-02';

custom_headmodel_path = [custom_headmodel_path,'/',SUBJ,'/custom_headmodel'];

if exist(custom_headmodel_path, 'dir') == 7
    disp('Path exists.');
else
    disp('Path does not exist.');
end

%% Load raw subject mri 

mri = ft_read_mri('/Users/vickyxu/Desktop/B2S/raw_EEG_data/subj-02/DICOM_VBRAIN_3DT1_0.8mm_20230209130618_201.nii.gz');

filename = [SUBJ,'_mri.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri');

disp(mri) % displays dim, anatomy, hdr, transform, and unit
%       dim: [256 256 160]
%   anatomy: [256×256×160 double]
%       hdr: [1×1 struct]
% transform: [4×4 double]
%      unit: 'mm'

%% Plot raw mri

cfg = [];
cfg.method = 'ortho';
ft_sourceplot(cfg, mri)

%% Transform mri to ctf coordinate system

% REPLACE the fiducials
% subj-02
raw_nas = [113 280 175];
raw_lpa = [22 164 135];
raw_rpa = [203 163 135];

% subj-03
% raw_nas = [126 210 65];
% raw_lpa = [203 104 42];
% raw_rpa = [56 105 42];

% subj-04
% raw_nas = [44 129 85];
% raw_lpa = [139 106 22];
% raw_rpa = [139 117 138];

cfg = [];
cfg.method = 'fiducial';

cfg.fiducial.nas = raw_nas;
cfg.fiducial.lpa = raw_lpa;
cfg.fiducial.rpa = raw_rpa;

cfg.coordsys = 'ctf'; % the desired coordinate system
mri_realigned = ft_volumerealign(cfg, mri)

filename = [SUBJ,'_mri_realigned.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_realigned');

%% Plot fiducials

cfg2 = [];
cfg2.locationcoordinates = 'voxel'; % treat the location as voxel coordinates
mri2 = mri;
mri2.transform = eye(4);
mri2.transform(:,4) = 1;

cfg2.location = cfg.fiducial.nas
ft_sourceplot(cfg2, mri2);

cfg2.location = cfg.fiducial.rpa
ft_sourceplot(cfg2, mri2);

cfg2.location = cfg.fiducial.lpa
ft_sourceplot(cfg2, mri2);

%% Reslice to have have isotropic voxels (if needed)
% also aligns the voxels with the axes of the coordinate system, 
% thereby avoiding it being plotted upside down later in the pipeline

% cfg = [];
% cfg.method = 'linear';
% mri_resliced = ft_volumereslice(cfg, mri_realigned);
% 
% cfg = [];
% cfg.method = 'ortho';
% ft_sourceplot(cfg, mri_resliced)

%% Trying to see if unbias would fix (if needed) -> we should be doing this

mri_to_be_correct = mri_realigned; % CHANGE

cfg = [];
mri_unbias = ft_volumebiascorrect(cfg, mri_to_be_correct);

ft_sourceplot(cfg, mri_unbias)

filename = [SUBJ,'_mri_unbias.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_unbias');


%% Load unbiased and normalised MRI (FSL output)
% 
% mri_mni_biascorr = ft_read_mri('/Users/vickyxu/Desktop/fsl_files_subj-02/T1_biascorr_to_std_sub.nii.gz');
% mri_mni_biascorr.coordsys = 'mni';
% 
% filename = [SUBJ,'_mri_mni_biascorr.mat'];
% save(fullfile(custom_headmodel_path, filename), 'mri_mni_biascorr');

%% Segmentation

cfg           = [];
cfg.output    = {'brain', 'skull', 'scalp'};

% Change the threshold values to fix weird head mesh etc. (IF NEEDED)
% cfg.scalpthreshold
% cfg.brainthreshold = 0.5;

% Manually change the skull thickness in the fieldtrip module (IF NEEDED)

mri_segmented_3_compartment  = ft_volumesegment(cfg, mri_unbias); % CHANGE to the mri needs to be segmented

ft_checkdata(mri_segmented_3_compartment, 'feedback', 'yes') % display some information about the segmentation

filename = [SUBJ,'_3com_segmentedmri_from_mri_unbiased.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_segmented_3_compartment');

%% Another segmentation method from TPM (not working)
% 
% cfg           = [];
% cfg.spmmethod = 'new';
% cfg.output    = {'tpm'};
% 
% segment_tpm   = ft_volumesegment(cfg, mri_mni_biascorr);
% cfg = [];
% cfg.output    = {'brain', 'skull', 'scalp'};
% 
% segmentedmri = ft_volumesegment(cfg, segment_tpm);
% 
% ft_checkdata(segmentedmri, 'feedback', 'yes') % display some information about the segmentation


%% Visualize the segmented brain, skull, and scalp

segmentedmri_indexed = ft_checkdata(mri_segmented_3_compartment, 'segmentationstyle', 'indexed')

segmentedmri_indexed.anatomy = mri_unbias.anatomy; % CHANGE IF NEEDED

cfg = [];
cfg.method = 'ortho';
cfg.anaparameter = 'anatomy';
cfg.funparameter = 'tissue';
cfg.funcolormap = [
  0 0 0
  1 0 0
  0 1 0
  0 0 1
  ];
ft_sourceplot(cfg, segmentedmri_indexed)

%% 5 Compartment segmentation

cfg = [];
cfg.output         = {'scalp', 'skull', 'csf', 'gray', 'white'};
cfg.brainsmooth    = 1;
cfg.scalpthreshold = 0.11;
cfg.skullthreshold = 0.15;
cfg.brainthreshold = 0.15;
mri_segmented_5_compartment = ft_volumesegment(cfg, mri_unbias);

filename = [SUBJ,'_5com_segmentedmri_from_mri_unbiased.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_segmented_5_compartment');


%% Visualize 

segmentedmri_indexed = ft_checkdata(mri_segmented_5_compartment, 'segmentationstyle', 'indexed')

segmentedmri_indexed.anatomy = mri_unbias.anatomy; % CHANGE IF NEEDED

tissue_colors_with_bg = [
    0.0  0.0  0.0;   % index 0 = background → BLACK
    0.0  0.4  0.8;   % index 1 = csf        → blue
    0.8  0.2  0.2;   % index 2 = gray       → red
    1.0  1.0  0.0;   % index 3 = scalp      → yellow
    0.0  1.0  1.0;   % index 4 = skull      → cyan
    0.9  0.9  0.9;   % index 5 = white      → light gray
];

cfg = [];
cfg.method        = 'ortho';
cfg.anaparameter  = 'anatomy';
cfg.funparameter  = 'tissue';
cfg.funcolormap = tissue_colors_with_bg;
cfg.funcolorlim = [-0.5 5.5];   
cfg.opacitylim    = [0 5];
cfg.opacitymap    = 'rampup';    % transparent where seg=0 (background)
ft_sourceplot(cfg, segmentedmri_indexed);


%% Construct meshes

% Notes on choosing number of vertices: The output consists of surfaces represented 
% by points or vertices that are connected in triangles. The tissues from which the 
% surfaces are created have to be specified and also the number of vertices for each 
% tissue. Since the potential changes the most rapidly on the outside of the brain 
% (or inside of the skull), we want that surface to be the most detailed. 
% The potential does not change rapidly over the scalp, so that can remain relatively coarse. 
% It is common to use the ratio 3/2/1 for the scalp/skull/brain

cfg = [];
cfg.tissue      = {'brain', 'skull', 'scalp'};
cfg.numvertices = [3000 2000 1000];
mesh = ft_prepare_mesh(cfg, mri_segmented_3_compartment);

%% Visualize

% individually
% figure
% ft_plot_mesh(mesh(1), 'facecolor', 'none'); % brain
% view([0 -1 0]); % from the right side
% 
% figure
% ft_plot_mesh(mesh(2), 'facecolor', 'none'); % skull
% view([0 -1 0]); % from the right side
% 
% figure
% ft_plot_mesh(mesh(3), 'facecolor', 'none'); % scalp
% view([0 -1 0]); % from the right side

% together
figure
ft_plot_mesh(mesh(1), 'facecolor','r', 'facealpha', 1.0, 'edgecolor', 'k', 'edgealpha', 1);
hold on
ft_plot_mesh(mesh(2), 'facecolor','g', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);
hold on
ft_plot_mesh(mesh(3), 'facecolor','b', 'facealpha', 0.4, 'edgecolor', 'k', 'edgealpha', 0.1);


%% Create a volume conduction model (BEM)
% the mesh needs to be properly closed and nested (the ft_plot_headmodel
% will check that)

% cfg        = [];
% cfg.method = 'bemcp'; % You can also specify 'openmeeg', 'dipoli', or another method
% headmodel  = ft_prepare_headmodel(cfg, mesh);
% 
% figure;
% ft_plot_headmodel(headmodel, 'facealpha', 0.6);
% 
% filename = [SUBJ,'_headmodel_bemcp.mat'];
% save(fullfile(custom_headmodel_path, filename), 'headmodel');

cfg        = [];
cfg.method = 'openmeeg'; % You can also specify 'openmeeg', 'dipoli', or another method
headmodel_openmeeg  = ft_prepare_headmodel(cfg, mesh);

figure;
ft_plot_headmodel(headmodel_openmeeg, 'facealpha', 0.6);
%% 

filename = [SUBJ,'_headmodel_openmeeg.mat'];
save(fullfile(custom_headmodel_path, filename), 'headmodel_openmeeg');

%% Normalise MRI 

cfg = [];
mri_normalised = ft_volumenormalise(cfg, mri_unbias); % CHANGE IF NEEDED

filename = [SUBJ,'_mri_normalised.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_normalised');

cfg = [];
mri_normalised.coordsys = 'mni';
cfg.method       = 'ortho';
cfg.anaparameter = 'anatomy';
cfg.location     = [0 0 0];
ft_sourceplot(cfg, mri_normalised);

%% Get transformed fiducials 

function chanlocs = get_transformed_fid(transformed_mri, raw_nas, raw_lpa, raw_rpa)

nas_ctf = ft_warp_apply(transformed_mri.transform, raw_nas, 'homogenous');
lpa_ctf = ft_warp_apply(transformed_mri.transform, raw_lpa, 'homogenous');
rpa_ctf = ft_warp_apply(transformed_mri.transform, raw_rpa, 'homogenous');

chanlocs = [];
chanlocs.labels = 'Nasion';
chanlocs.X      = nas_ctf(1);
chanlocs.Y      = nas_ctf(2);
chanlocs.Z      = nas_ctf(3);
chanlocs(end+1).labels = 'LPA';
chanlocs(end).X = lpa_ctf(1);
chanlocs(end).Y = lpa_ctf(2);
chanlocs(end).Z = lpa_ctf(3);
chanlocs(end+1).labels = 'RPA';
chanlocs(end).X = rpa_ctf(1);
chanlocs(end).Y = rpa_ctf(2);
chanlocs(end).Z = rpa_ctf(3);

disp(chanlocs(1))
disp(chanlocs(2))
disp(chanlocs(3))

end

chanlocs = get_transformed_fid(mri_unbias, raw_nas, raw_lpa, raw_rpa);
filename = [SUBJ,'_fid_chanlocs.mat'];
save(fullfile(custom_headmodel_path, filename), 'chanlocs');


%% Plot the transformed fiducials

cfg2 = [];
cfg2.method       = 'ortho';
cfg2.anaparameter = 'anatomy';

% Plot Nasion
cfg2.location = [chanlocs(1).X chanlocs(1).Y chanlocs(1).Z];
ft_sourceplot(cfg2, mri_unbias);
title('Nasion');

% Plot LPA
cfg2.location = [chanlocs(2).X chanlocs(2).Y chanlocs(2).Z];
ft_sourceplot(cfg2, mri_unbias);
title('LPA');

% Plot RPA
cfg2.location = [chanlocs(3).X chanlocs(3).Y chanlocs(3).Z];
ft_sourceplot(cfg2, mri_unbias);
title('RPA');

%% tutorial fix for weird head mesh 

% cfg          = [];
% cfg.output   = {'brain','skull','scalp'};
% cfg.scalpthreshold = 0.2; % changing threshold
% cfg.scalpsmooth  = 'no';
% segmentedmri = ft_volumesegment(cfg, mri_resliced);
% 
% cfg             = [];
% cfg.tissue      = {'brain','skull','scalp'};
% cfg.numvertices = [3000 2000 1000];
% bnd             = ft_prepare_mesh(cfg, segmentedmri);
% 
% figure;
% ft_plot_mesh(bnd(3), 'facecolor',[0.4 0.4 0.4]);
% view([0 0]);

%% Create mesh and head model (FEM)

cfg = [];
cfg.shift  = 0.3;
cfg.method = 'hexahedral';
cfg.resolution = 1; % this is in mm
mesh_fem = ft_prepare_mesh(cfg,mri_segmented_5_compartment);

cfg = [];
cfg.method = 'simbio';
cfg.conductivity = [0.43 0.0024 1.79 0.14 0.33]; % same as tissuelabel in vol_simbio
cfg.tissuelabel = {'scalp', 'skull', 'csf', 'gray', 'white'};
headmodel_fem = ft_prepare_headmodel(cfg, mesh_fem);


%% Visualize

% csf: 1, gm: 2, scalp: 3, skull: 4, wm: 5
ts = 3;
figure
mesh2 =[];
mesh2.hex = headmodel_fem.hex(headmodel_fem.tissue==ts,:); %mesh2.hex(1:size(mesh2.hex),:);
mesh2.pos = headmodel_fem.pos;
mesh2.tissue = headmodel_fem.tissue(headmodel_fem.tissue==ts,:); %mesh.tissue(1:size(mesh2.hex),:);

mesh_ed = mesh2edge(mesh2);
patch('Faces',mesh_ed.poly,...
  'Vertices',mesh_ed.pos,...
  'FaceAlpha',.5,...
  'LineStyle', 'none',...
  'FaceColor',[1 1 1],...
  'FaceLighting', 'gouraud');

xlabel('coronal');
ylabel('sagital');
zlabel('axial')
camlight;
axis on;



