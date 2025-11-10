import os
import scipy.io as sio
from main_preprocessing import Preprocessing
import mne
import pandas as pd
import lmdb
import pickle
import numpy as np


mped_channels = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M1','TP7', 'CP5', 'CP3',
    'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2','P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]

useless_ch = ['M1', 'M2']
# %%
root_path = './MPED/raw_EEG_signals'
file_list = os.listdir(root_path)
file_list.remove('raw_EEG_Label.mat')
file_list.sort()
print(file_list)
print(len(file_list))

label_each_trial = [0, 2, 1, 4, 3, 5, 6, 7, 8, 1, 2, 3, 8, 6, 7, 4, 5, 0, 1, 5, 6, 2, 2, 1, 8, 7, 6, 4, 4, 3, 5, 3, 7, 8]
using_trials = [2,3,4,5,6,7,8,10,11,12,14,15, 16, 17, 19, 20,21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32, 33]
print(label_each_trial)
#0-resting status, 1-neutral, 2-joy, 3-funny, 4-angry, 5-fear, 6-disgust, 7-sadness
Arousal = [-1, 4, 6, 5, 6, 6, 5, 5, -1]
Valence = [-1, 4, 7, 6, 3, 3, 2, 3, -1]

info = mne.create_info(ch_names=mped_channels, sfreq=1000, ch_types='eeg')

files_dict = {
    'train': file_list[:],
}
print(files_dict)

dataset = {
    'train': list(),
}

db_path = './processed_data/MPED'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=15 * 1024 * 1024 * 1024)  # 15GB
eeg_duration = 10

for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 10
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')
    # else:
    #     eeg_duration = 5
    for file_name in files_dict[files_key]:
        file_path = os.path.join(root_path, file_name)
        if file_name.endswith('.mat'):
            continue
        mat_list = os.listdir(file_path)
        sub_data = {}
        exclude_keys = ['__header__', '__version__', '__globals__']
        for mat_file in mat_list:
            mat_path = os.path.join(file_path, mat_file)
            data = sio.loadmat(mat_path)
            for key in data:
                if key not in exclude_keys:
                    sub_data[key] = data[key]
        for trial_idx in using_trials:
            trial_data = sub_data[f'raw_eeg{trial_idx}'][:64]
            print(f'Processing {file_name} trial {trial_idx} data shape: {trial_data.shape}')
            raw = mne.io.RawArray(trial_data, info)
            raw.drop_channels(useless_ch)
            raw.resample(200)
            processed = Preprocessing(raw=raw)
            processed.band_pass_filter(0.3, 49);
            processed.eeg_ica()
            processed.average_ref();
            data_i = processed.raw.get_data(units='V')
            samples = data_i.reshape(62, -1, 200)
            eeg_len = samples.shape[1]
            label = label_each_trial[trial_idx - 1]
            arousal = Arousal[label]
            valence = Valence[label]
            print('-'*20, trial_idx,label, arousal,valence)
            for j in range(eeg_len // eeg_duration):
                sample = samples[:, eeg_duration * j:eeg_duration * (j + 1), :]
                sample_key = f'{file_name}-{trial_idx}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label, 'arousal': arousal,'valence': valence,
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()


#
