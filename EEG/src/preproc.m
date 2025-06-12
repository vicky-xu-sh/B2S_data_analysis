% Preprocessing

eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets';

SPEECH_TYPE = 'im';
%% Import/load dataset

% === Import mff and save as dataset (Only run this if not saved as dataset) ===
% import mff and save as dataset
EEG = pop_mffimport({'/Users/vickyxu/Desktop/B2S/Final_datafiles/B2S_Pilot1_Imagined2_20241119_024607.mff'},{'classid','code','description','label','mffkeys','name'},0,0);
EEG.setname = ['pilot_',SPEECH_TYPE,'_raw'];
filename = [EEG.setname, '.set'];
EEG = pop_saveset(EEG, 'filename', filename, 'filepath', '/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets');
eeglab redraw; % refresh GUI
EEG_orig = EEG; % keep a copy of the original

% % === Load the raw dataset ===
% filename = ['pilot_',SPEECH_TYPE,'_raw.set'];
% EEG = pop_loadset('filename', filename, 'filepath', dataset_path);
% % updates data structure
% [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
% eeglab redraw; % refresh GUI
% EEG_orig = EEG; % keep a copy of the original


%% Import voice onset/offset events (run only for overt speech)
% Import events
speech_events_path = '/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/overt_audio_processing/speech_events.txt';
EEG = pop_importevent(EEG, 'event', speech_events_path, 'fields', {'latency' 'type'}, 'timeunit', 1);

% Check events
pop_eegplot(EEG, 1, 1, 1);

%% Visualize raw dataset
figure;  title('Raw data power spectra');
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 200],...
    'electrodes','off');

%% Highpass at 1 or 2 Hz and remove line noise

if strcmp(SPEECH_TYPE, 'sp')
    hp_cutoff = 2;
else 
    hp_cutoff = 1;
end

EEG = pop_eegfiltnew(EEG, 'locutoff', hp_cutoff); 

% % Remove 60 Hz line noise and its harmonics
EEG = pop_eegfiltnew(EEG, 'locutoff',59,'hicutoff',61,'revfilt',1); % set 'plotfreqz' to 1 if want freq/phase response graph
EEG = pop_eegfiltnew(EEG, 'locutoff',119,'hicutoff',121,'revfilt',1);
EEG = pop_eegfiltnew(EEG, 'locutoff',179,'hicutoff',181,'revfilt',1);

% Visualize
figure; title(sprintf('Power spectra after %d Hz high-pass and line noise removed',hp_cutoff));
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 250],...
    'electrodes','off');

%% Remove bad channels 

EEG = pop_clean_rawdata(EEG, ...
    'FlatlineCriterion',5, ...
    'ChannelCriterion',0.8, ...
    'LineNoiseCriterion',4, ...
    'Highpass','off', ...
    'BurstCriterion','off', ...
    'WindowCriterion','off', ...
    'BurstRejection','off', ...
    'Distance','Euclidian');

% % Visualize the removed channels
removed_chans = EEG_orig.chanlocs(~EEG.etc.clean_channel_mask); 
removed_labels = {removed_chans.labels};

% Create a subset EEG to visualize the removed channels
EEG_removed_only = pop_select(EEG_orig, 'channel', removed_labels);
pop_eegplot(EEG_removed_only, 1, 1, 1);  % Plot removed channels only

figure; 
title('Power spectra after filtered and bad channels removed');
pop_spectopo(EEG, 1, ...
    [0 EEG.pnts], ...
    'EEG' , 'freq', [20 50 100], ...
    'freqrange',[0.1 250],...
    'electrodes','off');

% save set
setname = ['pilot_', SPEECH_TYPE, '_raw_',num2str(hp_cutoff),'hz_hp_badchan_removed'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Re-reference to average and resample

EEG = pop_reref( EEG, []);
EEG = pop_resample( EEG, 500);

%% Manually inpspect the data and reject bad data segments

% OVERT (SPOKEN DATA)
% Aggressive rejection (some trial samples are removed)
% EEG = eeg_eegrej( EEG, [11 277;26672 27449;75704 76504;83332 84133;119318 120017;134020 134706;139461 139999;156043 156609;163150 163665;181993 182452;190899 191468;234690 235274]);

% Only reject beginning and end of data recording
% EEG = eeg_eegrej( EEG, [11 277; 234690 235274]);

% SAVE DATASET AFTER REJECTION

%% Define PCA function

function run_PCA(EEG)
% Reshape EEG data to 2D (channels x time)
data = double(EEG.data);
data_2d = reshape(data, EEG.nbchan, []);

% Center data
data_2d = data_2d - mean(data_2d, 2);

% Run SVD (PCA)
[U, S, ~] = svd(data_2d, 'econ');
explained = diag(S).^2 / sum(diag(S).^2);
cumulative_explained = cumsum(explained);

n_pca = find(cumulative_explained >= 0.99, 1);
fprintf('Number of PCs that explain 99%% variance: %d\n', n_pca);

n_pca = find(cumulative_explained >= 0.999, 1);
fprintf('Number of PCs that explain 99.9%% variance: %d\n', n_pca);

% Plot cumulative explained variance
figure;
plot(cumulative_explained * 100, 'LineWidth', 2); % convert to percent
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance (%)');
title('PCA: Cumulative Explained Variance');
grid on;

hold on;
yline(99, '--r', '99% threshold');
hold on;
yline(99.9, '--g', '99.9% threshold');
end

%% Run PCA to get num PCs that explain 99% var

run_PCA(EEG);

%% Run ICA (AMICA) 

% type “help runamica15()” for a full list and explanation of the parameters
% define parameters
numprocs = 1;       % # of nodes (default = 1)
max_threads = 4;    % # of threads per node
num_models = 1;     % # of models of mixture ICA
max_iter = 2000;    % max number of learning steps
% ===== EDIT THIS PARAMETER ==== %
pcakeep = 74;       % EDIT NUM of PCs to keep

% run amica
outdir = [ pwd filesep 'amicaouttmp' filesep ];
[weights,sphere,mods] = runamica15(EEG.data, 'num_models',num_models, 'outdir',outdir, ...
    'numprocs', numprocs, 'max_threads', max_threads, 'max_iter',max_iter, 'pcakeep', pcakeep);

% Assign weights and sphere to EEG
EEG.icaweights = weights;
EEG.icasphere = sphere;

% Store number of components
EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;  % assuming all channels used

% save set
setname = ['pilot_', SPEECH_TYPE, '_raw_',num2str(hp_cutoff),'hz_hp_badchan_removed_reref_resampled_seg_removed_PCA',num2str(pcakeep),'_AMICA'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Prepare analysis dataset (bandpass 1-200Hz)

% save ICA EEG set
EEG_ICA = EEG;

% Filtering Bandpass 1-200 Hz and remove line noise

% Band-pass at 1-200 Hz
EEG = pop_eegfiltnew(EEG_orig, 1, 200);

% Remove 60 Hz line noise and its harmonics
EEG = pop_eegfiltnew(EEG, 'locutoff',59,'hicutoff',61,'revfilt',1); 
EEG = pop_eegfiltnew(EEG, 'locutoff',119,'hicutoff',121,'revfilt',1);
EEG = pop_eegfiltnew(EEG, 'locutoff',179,'hicutoff',181,'revfilt',1);

% Remove the same bad channels
removed_chans = EEG.chanlocs(~EEG_ICA.etc.clean_channel_mask); 
removed_labels = {removed_chans.labels};
EEG = pop_select(EEG, 'nochannel', removed_labels);

% check channel consistency
if EEG_ICA.nbchan ~= EEG.nbchan
    error('Channel count does not match between datasets.');
elseif ~isequal({EEG_ICA.chanlocs.labels}, {EEG.chanlocs.labels})
    error('Channel labels or order do not match — cannot safely transfer ICA.');
else
    disp('Channel count and order are consistent.');
end

% re-referenve and resample
EEG = pop_reref( EEG, []);
EEG = pop_resample( EEG, 500);

% remove the same bad data segments

% OVERT/SPOKEN DATASET
% EEG = eeg_eegrej( EEG, [11 277; 234690 235274]);
% EEG = eeg_eegrej( EEG, [11 277;26672 27449;75704 76504;83332 84133;119318 120017;134020 134706;139461 139999;156043 156609;163150 163665;181993 182452;190899 191468;234690 235274]);

% COVERT/IMAGINED DATASET
EEG = eeg_eegrej( EEG, [1 427;237514 238432]);

%% Transfer ICA weights to 1-200Hz dataset for analysis

% ica_weights: [n_ICs × n_PCA_components]
% icasphere: [n_PCA_components × n_channels]

% Reconstruct the full unmixing matrix W
W = EEG_ICA.icaweights * EEG_ICA.icasphere;  % shape: [n_ICs × n_channels]

% Transfer ICA to EEG_analysis
EEG.icaweights  = W;
EEG.icawinv     = pinv(W);
EEG.icasphere   = eye(size(W, 2));  % full-rank, no sphering
EEG.icachansind = 1:size(W, 2);
EEG.icaact      = [];

% Check 
EEG = eeg_checkset(EEG);

% Run ICLabel classification
EEG = pop_iclabel(EEG, 'lite');
% view components
pop_viewprops(EEG, 0, 1:size(EEG.icaweights,1), {'freqrange' [2 200]});

setname = ['pilot_', SPEECH_TYPE, '_bp_1_200hz_bad_data_removed_full_transferred_ICs'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Mark non-brain ICs to be rejected (but don't reject yet)

% Extract ICLabel probabilities
% Columns order (default ICLabel output):
% 1: Brain, 2: Muscle, 3: Eye, 4: Heart, 5: Line Noise, 6: Channel Noise, 7: Other

ic_probs = EEG.etc.ic_classification.ICLabel.classifications;

% Define muscle ICs to have prob > 0.5
all_muscle_ICs = find(ic_probs(:,2) > 0.5);

% The 7th column corresponds to the "other" class
other_probs = ic_probs(:,7);

% Find indices where "other" is the most likely class
[~, max_classes] = max(ic_probs, [], 2);
other_ICs = find(max_classes == 7);

% find all ICs with brain probability < thresh
thresh = 0.05;
non_brain_ICs = find(ic_probs(:,1) < thresh);
fprintf('\nNon-brain ICs: %s\n', num2str(non_brain_ICs'));

% find ICs that are non brain but not marked as other
ICs_to_reject = setdiff(non_brain_ICs, other_ICs);
fprintf('\nNon-brain but not marked as others ICs: %s\n', num2str(ICs_to_reject'));

% Mark non-brain ICs for rejection
EEG.reject.gcompreject(ICs_to_reject) = 1;

rejected_muscle_ICs = intersect(all_muscle_ICs, ICs_to_reject);
fprintf('\nRejected muscle ICs: %s\n', num2str(rejected_muscle_ICs'));

% visualize the components, do manual rejections
pop_selectcomps(EEG, [1:size(EEG.icaweights,1)] );

%% 

% save set
setname = ['pilot_', SPEECH_TYPE, '_bp_1_200hz_bad_data_removed_full_transferred_ICs_mark_non_brain'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% subtract rejected ICs

EEG = pop_subcomp(EEG, [], 1);

%% Interpolate removed bad channels and mark Nz/RPA/LPA channels

EEG = pop_interp(EEG, EEG_orig.chanlocs, 'spherical');
EEG = pop_chanedit(EEG, 'changefield',{258,'labels','Nz'},'changefield',{260,'labels','RPA'},'changefield',{259,'labels','LPA'});
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

%% re-reference to average

EEG = pop_reref( EEG, []);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

%% Run PCA to determine # PCs to keep

run_PCA(EEG);

%% Second ICA run

numprocs = 1;       % # of nodes (default = 1)
max_threads = 4;    % # of threads per node
num_models = 1;     % # of models of mixture ICA
max_iter = 2000;    % max number of learning steps
% ===== EDIT THIS PARAMETER ==== %
pcakeep = 22;       % EDIT NUM of PCs to keep

% run amica
outdir = [ pwd filesep 'amicaouttmp' filesep ];
[weights,sphere,mods] = runamica15(EEG.data, 'num_models',num_models, 'outdir',outdir, ...
    'numprocs', numprocs, 'max_threads', max_threads, 'max_iter',max_iter, 'pcakeep', pcakeep);

% Assign weights and sphere to EEG
EEG.icaweights = weights;
EEG.icasphere = sphere;

% Store number of components
EEG.icawinv = pinv(EEG.icaweights * EEG.icasphere);
EEG.icachansind = 1:EEG.nbchan;  % assuming all channels used

% Run ICLabel classification
EEG = pop_iclabel(EEG, 'default');
% view components
pop_viewprops(EEG, 0, 1:size(EEG.icaweights,1), {'freqrange' [2 200]});

%% save
setname = ['pilot_', SPEECH_TYPE, '_bp_1_200hz_bad_data_removed_cleaned_interpolated_2ndICA'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% DIPFIT

EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat',...
                           'chanfile','standard_1005.elc','coordformat','MNI','coord_transform',...
                           [-0.10139 -20.0657 -6.2232 0.12425 0.00053703 -1.5713 10.4359 10 10.0369] ,...
                           'chanomit',[17 32 33 43 44 48 49 56 57 63 68 73 81 88 94 99 107 113:6:125 126:128] );
% coarse fit
EEG = pop_dipfit_gridsearch(EEG, [1:22] ,[-85     -77.6087     -70.2174     -62.8261     -55.4348     -48.0435     -40.6522     -33.2609     -25.8696     -18.4783      -11.087     -3.69565      3.69565       11.087      18.4783      25.8696      33.2609      40.6522      48.0435      55.4348      62.8261      70.2174      77.6087           85] ,[-85     -77.6087     -70.2174     -62.8261     -55.4348     -48.0435     -40.6522     -33.2609     -25.8696     -18.4783      -11.087     -3.69565      3.69565       11.087      18.4783      25.8696      33.2609      40.6522      48.0435      55.4348      62.8261      70.2174      77.6087           85] ,[0      7.72727      15.4545      23.1818      30.9091      38.6364      46.3636      54.0909      61.8182      69.5455      77.2727           85] ,0.4);

% run fine fit manually 

% save dipole fitted set
setname = ['pilot_', SPEECH_TYPE, '_cleaned_2ndICA_dipole_fit'];
filename = [setname, '.set'];
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET+1,...
    'setname',setname, ...
    'savenew',fullfile(dataset_path, filename), ...
    'gui','off'); 
eeglab('redraw'); % refresh GUI

%% Epoch

% speech/overt
% EEG = pop_epoch( EEG, {  'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _guSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _miSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _muSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _siSP_[]_ECI TCP-IP 55513'...
%                          'EVNT_STIM_    _suSP_[]_ECI TCP-IP 55513'  }, ...
%                          [-0.5 1.5], ...
%                          'newname', 'pilot_sp_cleaned_2ndICA_dipfit_voice_seg_marked_dipfit_epoched', ...
%                          'epochinfo', 'yes');
% 
% % save to disk
% filename = 'pilot_sp_cleaned_2ndICA_dipfit_voice_seg_marked_dipfit_epoched.set';
% EEG = pop_saveset(EEG, 'filename', filename, 'filepath', dataset_path);

% covert

% mark non-brain
ICs_to_ignore = [5,8,10,11,13,15,17,19:22];
EEG.reject.gcompreject(ICs_to_ignore) = 1;

setname = ['pilot_', SPEECH_TYPE, '_cleaned_2ndICA_dipole_fit_epoched'];
EEG = pop_epoch( EEG, {  'EVNT_STIM_    _giIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _guIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _miIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _muIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _siIM_[]_ECI TCP-IP 55513'...
                         'EVNT_STIM_    _suIM_[]_ECI TCP-IP 55513'  }, ...
                         [-0.5 1.5], ...
                         'newname', setname, ...
                         'epochinfo', 'yes');

% save to disk
filename = [setname, '.set'];
EEG = pop_saveset(EEG, 'filename', filename, 'filepath', dataset_path);