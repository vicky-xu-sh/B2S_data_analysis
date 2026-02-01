eeglab; % launch EEGLAB
dataset_path = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets';

%% Load dataset

EEG = pop_loadset('filename', 'pilot_sp_raw.set', 'filepath', dataset_path);
% updates data structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw; % refresh GUI


%% Import voice onset/offset events
% Import events
speech_events_path = '../../overt_audio_processing/speech_events.txt';
EEG = pop_importevent(EEG, 'event', speech_events_path, 'fields', {'latency' 'type'}, 'timeunit', 1);

% Check events
pop_eegplot(EEG, 1, 1, 1);


% Align with the cleaned EEG dataset
EEG = pop_resample( EEG, 500);
EEG = eeg_eegrej( EEG, [11 277;26672 27449;75704 76504;83332 84133;119318 120017;134020 134706;139461 139999;156043 156609;163150 163665;181993 182452;190899 191468;234690 235274]);
%% 

% Set output filename
filename = '/Users/vickyxu/Desktop/B2S/B2S_data_analysis/overt_audio_processing/subj-02_sess-01/speech_onset_offset_events_clean_EEG_aligned.txt';
fid = fopen(filename, 'w');

% Loop through EEG events
for i = 1:length(EEG.event)
    eventType = EEG.event(i).type;
    
    % Check if event is 'onset' or 'offset'
    if ischar(eventType) && (strcmp(eventType, 'onset') || strcmp(eventType, 'offset'))
        % Convert latency (in samples) to seconds
        time_sec = EEG.event(i).latency / EEG.srate;
        fprintf(fid, '%.6f\t%s\n', time_sec, eventType);
    end
end

% Close file
fclose(fid);