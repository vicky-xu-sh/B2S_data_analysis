%% Load path

ft_defaults

custom_headmodel_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

SUBJ = 'subj-04';
SESS = 'sess-01';

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

disp(mri) 
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
% raw_nas = [102 278 174];
% raw_lpa = [22 164 135];
% raw_rpa = [203 163 135];

% subj-03
% raw_nas = [126 210 65];
% raw_lpa = [203 104 42];
% raw_rpa = [56 105 42];

% subj-04
raw_nas = [44 129 85];
raw_lpa = [139 106 22];
raw_rpa = [139 117 138];

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

%% Trying to see if unbias would fix (if needed)

% mri_to_be_correct = mri_realigned; % CHANGE
% 
% cfg = [];
% mri_unbias = ft_volumebiascorrect(cfg, mri_to_be_correct);
% 
% ft_sourceplot(cfg, mri_unbias)
% 
% filename = [SUBJ,'_mri_unbias.mat'];
% save(fullfile(custom_headmodel_path, filename), 'mri_unbias');


%% Segmentation

cfg           = [];
cfg.output    = {'brain', 'skull', 'scalp'};

% Change the threshold values to fix weird head mesh etc. (IF NEEDED)
% cfg.scalpthreshold
% cfg.brainthreshold = 0.5;

% Manually change the skull thickness in the fieldtrip module (IF NEEDED)

segmentedmri  = ft_volumesegment(cfg, mri_realigned); % CHANGE to the mri needs to be segmented

ft_checkdata(segmentedmri, 'feedback', 'yes') % display some information about the segmentation

filename = [SUBJ,'_segmentedmri_from_mri_realigned.mat'];
save(fullfile(custom_headmodel_path, filename), 'segmentedmri');

%% Another segmentation method from TPM (not working)

% cfg           = [];
% cfg.spmmethod = 'new';
% cfg.output    = {'tpm'};
% segment_tpm   = ft_volumesegment(cfg, mri_unbias);
% cfg = [];
% cfg.output    = {'brain', 'skull', 'scalp'};
% 
% segmentedmri = ft_volumesegment(cfg, segment_tpm);
% 
% ft_checkdata(segmentedmri, 'feedback', 'yes') % display some information about the segmentation


%% Visualize the segmented brain, skull, and scalp

segmentedmri_indexed = ft_checkdata(segmentedmri, 'segmentationstyle', 'indexed')

segmentedmri_indexed.anatomy = mri_realigned.anatomy; % CHANGE IF NEEDED

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
mesh = ft_prepare_mesh(cfg, segmentedmri);

% Also tried iso2mesh, but doesn't really work
% cfg = [];
% cfg.method = 'iso2mesh';
% cfg.tissue      = {'brain', 'skull', 'scalp'};
% cfg.numvertices = [3000 2000 1000];
% mesh = ft_prepare_mesh(cfg, segmentedmri);

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


%% Create a volume conduction model
% the mesh needs to be properly closed and nested (the ft_plot_headmodel
% will check that)

cfg        = [];
cfg.method = 'bemcp'; % You can also specify 'openmeeg', 'dipoli', or another method
headmodel  = ft_prepare_headmodel(cfg, mesh);

ft_plot_headmodel(headmodel, 'unit', 'mm')

filename = [SUBJ,'_headmodel_bemcp.mat'];
save(fullfile(custom_headmodel_path, filename), 'headmodel');

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

%% Get transformed fiducials (Need to verify the method correctness)

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

chanlocs = get_transformed_fid(mri_realigned, raw_nas, raw_lpa, raw_rpa);
filename = [SUBJ,'_fid_chanlocs.mat'];
save(fullfile(custom_headmodel_path, filename), 'chanlocs');


%% Plot the transformed fiducials

cfg2 = [];
cfg2.locationcoordinates = 'voxel'; % treat the location as voxel coordinates
mri2 = mri_realigned;
mri2.transform = eye(4);
mri2.transform(:,4) = 1;

cfg2.location = [chanlocs(1).X chanlocs(1).Y chanlocs(1).Z];
ft_sourceplot(cfg2, mri2);

cfg2.location = [chanlocs(2).X chanlocs(2).Y chanlocs(2).Z];
ft_sourceplot(cfg2, mri2);

cfg2.location = [chanlocs(3).X chanlocs(3).Y chanlocs(3).Z];
ft_sourceplot(cfg2, mri2);

%% Normalise MRI 

cfg = [];
mri_normalised = ft_volumenormalise(cfg, mri_realigned); % CHANGE IF NEEDED

filename = [SUBJ,'_mri_normalised.mat'];
save(fullfile(custom_headmodel_path, filename), 'mri_normalised');

ft_sourceplot(cfg, mri_normalised);