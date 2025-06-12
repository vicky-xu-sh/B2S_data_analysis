% EEGLAB history file generated on the 20-May-2025
% ------------------------------------------------
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG = pop_loadset('filename','pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA.set','filepath','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets/');
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
EEG=pop_chanedit(EEG, 'lookup','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/AdultAverageNet256_v1.sfp');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
pop_eegplot( EEG, 1, 1, 1);
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'overwrite','on','gui','off'); 
EEG=pop_chanedit(EEG, 'changefield',{258,'labels','Nz'},'changefield',{258,'X','0'},'changefield',{258,'Y','85'},'changefield',{258,'Z','-40'},'changefield',{259,'labels','LPA'},'changefield',{259,'X','-85'},'changefield',{259,'Y','0'},'changefield',{259,'Z','-40'},'changefield',{260,'labels','RPA'},'changefield',{260,'X','85'},'changefield',{260,'Y','0'},'changefield',{260,'Z','-40'},'lookup','/Users/vickyxu/Documents/MATLAB/Toolboxes/eeglab2025.0.0/plugins/Fieldtrip-lite20240111/template/electrode/GSN-HydroCel-256.sfp');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG=pop_chanedit(EEG, 'changefield',{258,'X','0'},'changefield',{258,'Y','85'},'changefield',{258,'Z','-40'},'changefield',{258,'X',''},'changefield',{258,'Y','0'},'changefield',{258,'X','10'},'changefield',{258,'Z','-2'},'changefield',{258,'X','0'},'changefield',{258,'Y','1'},'changefield',{258,'Z','0'},'changefield',{258,'X','1'},'changefield',{258,'Y','0'});
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 2,'retrieve',1,'study',0); 
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 1,'retrieve',2,'study',0); 
EEG = pop_loadset('filename','pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA.set','filepath','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets/');
[ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
EEG = pop_interp(EEG, ALLEEG(1).chanlocs, 'spherical');
[ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 3,'setname','pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA_interpolated','savenew','pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA_interpolated.set','gui','off'); 
EEG=pop_chanedit(EEG, []);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG=pop_chanedit(EEG, 'changefield',{258,'labels','Nz'},'changefield',{260,'labels','RFA'},'changefield',{260,'labels','RPA'},'changefield',{259,'labels','LPA'});
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = pop_dipfit_settings( EEG, 'hdmfile','standard_vol.mat','mrifile','standard_mri.mat','chanfile','standard_1005.elc','coordformat','MNI','coord_transform',[-0.10139 -20.0657 -6.2232 0.12425 0.00053703 -1.5713 10.4359 10 10.0369] ,'chanomit',[17 32 33 43 44 48 49 56 57 63 68 73 81 88 94 99 107 113:6:125 126:128] );
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
pop_prop( EEG, 0, 2, NaN, {'freqrange',[2 200] });
figure;pop_topoplot(EEG, 0, 2,'pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA_interpolated',[1 1] ,0,'electrodes','off');
EEG = pop_dipfit_gridsearch(EEG, [1:23] ,[-85     -77.6087     -70.2174     -62.8261     -55.4348     -48.0435     -40.6522     -33.2609     -25.8696     -18.4783      -11.087     -3.69565      3.69565       11.087      18.4783      25.8696      33.2609      40.6522      48.0435      55.4348      62.8261      70.2174      77.6087           85] ,[-85     -77.6087     -70.2174     -62.8261     -55.4348     -48.0435     -40.6522     -33.2609     -25.8696     -18.4783      -11.087     -3.69565      3.69565       11.087      18.4783      25.8696      33.2609      40.6522      48.0435      55.4348      62.8261      70.2174      77.6087           85] ,[0      7.72727      15.4545      23.1818      30.9091      38.6364      46.3636      54.0909      61.8182      69.5455      77.2727           85] ,0.4);
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
pop_dipplot( EEG, [2 3 5 6:10 14 21] ,'mri','standard_mri.mat','normlen','on');
pop_topoplot(EEG, 0, [1:23] ,'pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA_interpolated',[5 5] ,0,'electrodes','off');
pop_topoplot(EEG, 0, [1:23] ,'pilot_sp_bp_1_200hz_bad_data_removed_cleaned_2ndICA_interpolated',[5 5] ,1,'electrodes','off');
% === History not supported for manual dipole fitting ===
EEG = pop_saveset( EEG, 'filename','dipole_fit.set','filepath','/Users/vickyxu/Desktop/B2S/B2S-EEG-Analysis/datasets/');
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);
eeglab redraw;
