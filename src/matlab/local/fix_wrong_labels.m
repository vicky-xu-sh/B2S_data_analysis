%% =========================================================================
%  FIX WRONG EVENT LABELS (SP → IM)
%  NOTE: subj-XX im dataset had stimulus events incorrectly marked as SP
% ==========================================================================

for i = 1:length(EEG.event)
    % Fix event type string e.g. 'EVNT_STIM_    _giSP_[]_ECI TCP-IP 55513'
    EEG.event(i).type  = strrep(EEG.event(i).type,  'SP_[]', 'IM_[]');
    % Fix label field e.g. 'giSP' -> 'giIM'
    if isfield(EEG.event, 'label')
        EEG.event(i).label = strrep(EEG.event(i).label, 'SP', 'IM');
    end
end

% Verify fix
fprintf('Unique event types after fix:\n');
disp(unique({EEG.event.type})');
if isfield(EEG.event, 'label')
    fprintf('Unique event labels after fix:\n');
    disp(unique({EEG.event.label})');
end