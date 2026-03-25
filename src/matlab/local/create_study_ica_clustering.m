% Create a study to compare the brain sources of covert vs overt condition

STUDY = [];
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
[STUDY ALLEEG] = std_editset( STUDY, [], 'commands', ...
    { ...
        {'index',1,'load','/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets/subj-03/sess-01/spoken/subj-03_sess-01_pilot_sp_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf.set'}, ...
        {'index',2,'load','/Users/vickyxu/Desktop/B2S/B2S_data_analysis/EEG/datasets/subj-03/sess-01/imagined/subj-03_sess-01_pilot_im_cleaned_2ndICA_epoched_dipfit_bemcp_headmodel_realigned_mri_ctf.set'}, ...
        {'index',1,'subject','S03-1'},{'index',2,'subject','S03-2'}, ...
        {'index',1,'session',1}, ...
        {'index',2,'session',1} ...
    },'updatedat','off' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);
[STUDY ALLEEG] = std_editset( STUDY, ALLEEG, 'name','subj-03_spoken_vs_imagined','commands',{{'index',1,'comps',[4 7 11 12:14 16] }, {'index',2,'comps',[2:3:8 9:14] }},'updatedat','off','rmclust','on' );
[STUDY ALLEEG] = std_checkset(STUDY, ALLEEG);

CURRENTSTUDY = 1; EEG = ALLEEG; CURRENTSET = [1:length(EEG)];
[STUDY, ALLEEG] = std_checkset(STUDY, ALLEEG);
eeglab redraw;

%% Cluster ICA components

[STUDY, ALLEEG] = std_precomp(STUDY, ALLEEG, 'components','savetrials','on','recompute','on','spec','on','specparams',{'specmode','fft','logtrials','off'});
[STUDY ALLEEG] = std_preclust(STUDY, ALLEEG, 1,{'dipoles','weight',1},{'moments','weight',1});

[STUDY] = pop_clust(STUDY, ALLEEG, 'algorithm','kmeans','clus_num',  8 , 'outliers',  3 );


%% COORMAP commands

CORRMAP=corrmap(STUDY,ALLEEG,1,4,'th','0.8','ics',2);
CORRMAP.output
CORRMAP.output.ics{2}
CORRMAP.output.sets{2}
CORRMAP.corr
CORRMAP.corr.sets{2}
CORRMAP.corr.ics{2}
CORRMAP.corr.abs_values{2}
