% ERSP plotting experiment

% Assume EEG dataset is loaded

EEG.icaact = (EEG.icaweights * EEG.icasphere) * EEG.data(EEG.icachansind, :);
% Reshape to 3D: [ICs x time x trials]
EEG.icaact = reshape(EEG.icaact, size(EEG.icaweights, 1), EEG.pnts, EEG.trials);


%% 

comp_num = 7;
max_time_ms = 1498;

freqs = exp(linspace(log(4), log(160), 100));

% Construct cycles vector matched to freqs
cycles = zeros(size(freqs));
for i = 1:length(freqs)
    if freqs(i) <= 45
        % linear scale from 1 to 8 in the lower part
        cycles(i) = 1 + (8 - 1) * (log(freqs(i)) - log(4)) / (log(45) - log(4));
    else
        cycles(i) = 8;
    end
end


timefreqs = [400  8;    % [time_ms  freq_Hz]
             350 14;
             500 24;
             1050 11];

[ersp,itc,powbase,times,freqs_out,erspboot,itcboot,tfdata] = newtimef( ...
    EEG.icaact(comp_num, :, :), ...
    EEG.pnts, [-500 max_time_ms], EEG.srate, cycles, ...
    'topovec', EEG.icawinv(:,comp_num), 'elocs', EEG.chanlocs, 'chaninfo', EEG.chaninfo, ...
    'baseline',[-450 0], 'freqs', freqs, ... 
    'freqscale', 'log', 'plotphase', 'off', 'ntimesout', 200, ...
    'plotitc', 'off', 'plotersp', 'on', ...   % <- now it draws the ERSP figure
    'basenorm', 'on', 'trialbase', 'full');




%% --- Your wavelet settings ---
freqs = exp(linspace(log(4), log(160), 100));
cycles = zeros(size(freqs));
for i = 1:length(freqs)
    if freqs(i) <= 45
        cycles(i) = 1 + (8 - 1) * (log(freqs(i)) - log(4)) / (log(45) - log(4));
    else
        cycles(i) = 8;
    end
end

%% --- Extract IC activation ---
comp_num = 7; % example
EEG = eeg_checkset(EEG, 'ica');

EEG.icaact = (EEG.icaweights * EEG.icasphere) * EEG.data(EEG.icachansind, :);
% Reshape to 3D: [ICs x time x trials]
EEG.icaact = reshape(EEG.icaact, size(EEG.icaweights, 1), EEG.pnts, EEG.trials);

useChans = EEG.icachansind;                  
A = EEG.icawinv(useChans, comp_num);          % [channels x 1]
S = squeeze(EEG.icaact(comp_num, :, :));      % [pnts x trials]
[pnts, ntrials] = size(S);
nCh = length(A);

%% --- Reconstruct IC -> channel space ---
X = zeros(nCh, pnts, ntrials);
for tr = 1:ntrials
    X(:,:,tr) = A * S(:, tr).';
end

%% --- Build a temporary EEG object ---
EEG_ic              = EEG;
EEG_ic.data         = X;
EEG_ic.nbchan       = nCh;
EEG_ic.chanlocs     = EEG.chanlocs(useChans);
EEG_ic.icaact       = [];
EEG_ic.icaweights   = [];
EEG_ic.icasphere    = [];
EEG_ic.icawinv      = [];
EEG_ic.icachansind  = [];

%% --- Compute ERSP per channel (no plotting) ---
for ch = 1:nCh
    [ersp,~,~,times,freqs_out,erspboot,~] = newtimef( ...
        squeeze(EEG_ic.data(ch,:,:)), EEG_ic.pnts, ...
        [EEG_ic.xmin*1000 EEG_ic.xmax*1000], EEG_ic.srate, cycles, ...
        'freqs', freqs, 'freqscale','log', ...
        'baseline',[-450 0], 'plotersp','off','plotitc','off');

    if ch == 1
        allersp     = zeros([size(ersp) nCh]);
        alltimes    = zeros([size(times) nCh]);
        allfreqs    = zeros([size(freqs_out) nCh]);
        allerspboot = zeros([size(erspboot) nCh]);
    end

    allersp(:,:,ch)     = ersp;
    alltimes(:,:,ch)    = times;
    allfreqs(:,:,ch)    = freqs_out;
    allerspboot(:,:,ch) = erspboot;
end

%% --- Now use TFTOTO for multi-topo plotting ---
timefreqs = [400 8; 350 14; 500 24; 1050 11];

figure;
tftopo(allersp, alltimes(:,:,1), allfreqs(:,:,1), ...
    'chanlocs', EEG_ic.chanlocs, ...
    'timefreqs', timefreqs, ...
    'signifs',  allerspboot, ...
    'sigthresh',[6], ...
    'mode', 'ave', ...
    'limits', [nan nan nan 35 -1.5 1.5])

sgtitle(['IC ' num2str(comp_num) ' projected to scalp']);
