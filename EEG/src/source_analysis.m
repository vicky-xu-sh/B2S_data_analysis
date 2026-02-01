% Source analysis

eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

% CHANGE THIS
SPEECH_TYPE = 'sp';
SUBJ = 'subj-02';
SESS = 'sess-02';

if SPEECH_TYPE == 'sp'
    dataset_path = [dataset_path,'/',SUBJ,'/',SESS,'/spoken'];  % make sure datapath exists
else 
    dataset_path = [dataset_path,'/',SUBJ,'/',SESS,'/imagined'];
end


if exist(dataset_path, 'dir') == 7
    disp('Path exists.');
else
    disp('Path does not exist.');
end

%% Load epoched dataset

setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched'];
filename = [setname, '.set'];
EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
% updates data structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw; % refresh GUI

%% Create custom head model from MRI (inside DIPFIT), Run DIPFIT autofit (instead of coaurse fit + fine fit)

% EEG = pop_dipfit_headmodel(EEG, '/Users/vickyxu/Desktop/B2S/raw_EEG_data/subj-03/3DT1_LB_FLIP.nii', 'datatype','EEG','plotfiducial',{'nasion','lpa','rpa'});
% subj-03 spoken
% EEG = pop_dipfit_settings( EEG, 'coordformat','MNI','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[162:184] );
% subj-03 imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','MNI','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[67 182 183 187 188:216] );


%% Use headmodel built from fieldtrip (coordinate system of realigned or resliced/unbiased mri and headmodel should be CTF)

headmodel_path = [dataset_path, '/../../custom_headmodel/', SUBJ, '_headmodel_bemcp.mat'];
mri_path = [dataset_path, '/../../custom_headmodel/', SUBJ, '_mri_unbias.mat']; % CHANGE
chanlocs_path = [dataset_path, '/../../custom_headmodel/', SUBJ, '_fid_chanlocs.mat'];

load(headmodel_path, 'headmodel');
load(mri_path, 'mri_unbias'); mri = mri_unbias;
% load(mri_path, 'mri_realigned'); mri = mri_realigned;
load(chanlocs_path, 'chanlocs');

EEG.dipfit.hdmfile = headmodel;
EEG.dipfit.mrifile = mri;
EEG.dipfit.chanfile = chanlocs;
EEG.dipfit.coordformat = 'ctf'; % or 'SCS' as Brainstorm calls it
EEG.dipfit.coord_transform = [];

EEG = pop_dipfit_settings(EEG); % interactive window

% subj-02, sess-02, spoken
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[-1.28 -0.6 23 -0.1585 0.1 0.18 10 10 10] ,'chanomit', [210 211 217:245] );
% subj-02, sess-02, imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[-1.28 -0.6 23 -0.1585 0.1 0.18 10 10 10] ,'chanomit', [204 205 211:240] );

% subj-03 spoken
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[162:184] );

% subj-03 imagined
% EEG = pop_dipfit_settings( EEG, 'coordformat','ctf','coord_transform',[10 -2 23 0.0064942 0.14616 -0.071998 9.98 9.98 9.98] ,'chanomit',[67 182 183 187 188:216] );

%% Standard template headmodel

% subj-02, sess-02, spoken
% EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform', ...
%     [1.5982 -19.98 -11.8991 0.05 0.1428 -1.454 11.0134 11.0134 11.0134] ,'chanomit',[210 211 217 218:245] );

% subj-02, sess-02, imagined
% EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform', ...
%     [1.5982 -19.98 -11.8991 0.05 0.1428 -1.454 11.0134 11.0134 11.0134] ,'chanomit',[204 205 211:240] );


%% Run DIPFIT autofit (instead of coaurse fit + fine fit)

% CHANGE THIS
num_comps = 33;

EEG = pop_multifit(EEG, [1:num_comps] ,'threshold',100);

% Save dipole fitted set
% setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE,
%           '_cleaned_2ndICA_epoched_dipfit_standard']; % if using template/standard headmodel
setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI


%% Plot individual dipoles 
% This hould be in subject ctf space, however, there might be bugs in
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
norm_mri_path = [dataset_path, '/../../custom_headmodel/', SUBJ, '_mri_normalised.mat'];
load(norm_mri_path, 'mri_normalised');

model_mni = EEG.dipfit.model;

for i = 1:length(model_mni)
    if ~isempty(model_mni(i).posxyz)
        % Apply the non-linear warp using the params from ft_volumenormalise
        model_mni(i).posxyz = ft_warp_apply(mri_normalised.params, model_mni(i).posxyz, 'individual2sn');
    end
end

%% Look up the atlas (atlas only supports case where dipole results are in MNI or spherical)

EEG2 = EEG;
EEG2.dipfit.model = model_mni;
EEG2.coordormat = 'MNI';
EEG2.coord_transform = [];
EEG2.mrifile = mri_normalised;

EEG2 = eeg_compatlas(EEG2);

% Plot the dipoles (in MNI)
dipplot(EEG2.dipfit.model([1:num_comps]), 'coordformat', 'MNI', 'transform', [], 'mri', mri_normalised,'num','on','axistight','on','pointout','on','normlen','on');

%% Save MNI space result (don't know if the warping is good/correct, need to further verify)

setname = [SUBJ,'_',SESS,'_pilot_', SPEECH_TYPE, '_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf_warptoMNI'];
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

%% Plot the DK atlas lookup results for dipole location

% ICs = [4 7 11 12:14 16];
ICs = [2:3:8 9:14];
for i = ICs
    fprintf('Comp %d %s\n', i, EEG.dipfit.model(i).areadk)
end