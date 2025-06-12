eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets';

EEG = pop_loadset('filename', 'dipfit_brain_only_voice_seg_marked.set', 'filepath', dataset_path);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw; % refresh GUI

%% 

% Convert EEG to FieldTrip format
data = eeglab2fieldtrip(EEG, 'preprocessing', 'none');

% Manually add sampleinfo
data.sampleinfo = zeros(length(data.trial), 2);
for i = 1:length(data.trial)
    data.sampleinfo(i,:) = [1 size(data.trial{i}, 2)];
end
%% 

% Make sure electrode labels match
% elec = ft_read_sens('standard_1005.elc'); % or use EEG.chanlocs
data.label = elec.label;

% Check leadfield alignment
% load('your_leadfield_file.mat'); % from DIPFIT grid creation

% freq analysis (eLORETA needs freq-domain input)
cfg = [];
cfg.method = 'mtmfft';
cfg.output = 'pow';
cfg.foilim = [8 12]; % example for alpha
cfg.taper = 'hanning';
cfg.pad = 'nextpow2';
freq = ft_freqanalysis(cfg, data);

% Source analysis
cfg = [];
cfg.method = 'eloreta';
cfg.frequency = freq.freq;
cfg.grid = EEG.dipfit.grid;   % includes leadfield + positions + inside voxels
cfg.headmodel = EEG.dipfit.vol;
cfg.elec = EEG.dipfit.elec;
source = ft_sourceanalysis(cfg, freq);