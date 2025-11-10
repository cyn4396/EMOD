import os

from scipy.io import loadmat
import lmdb
import pickle
import numpy as np
from channel_idx import dreamer_channels
import mne
from main_preprocessing import Preprocessing

# %%
data_path = './DREAMER/DREAMER.mat'
num_subs = 23
num_trials = 18

files_dict = {
    'train': np.arange(0,23),
}

dataset = {
    'train': list(),
}



eeg_duration = 10
frequency = 128
info = mne.create_info(ch_names=dreamer_channels, sfreq=128, ch_types='eeg')

db_path = './processed_data/DREAMER'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=1500*1024*1024)

for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 10
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')

    for sub_id in files_dict[files_key]:
        sub_data = loadmat(data_path)['DREAMER'][0, 0]['Data'][0, sub_id]
        for trialid in range(num_trials):

            eeg_data = sub_data['EEG'][0, 0]['stimuli'][0, 0][trialid, 0]
            eeg_data = eeg_data.transpose(1, 0)
            print(eeg_data.shape)


            eeg_baseline = sub_data['EEG'][0, 0]['baseline'][0, 0][trialid, 0]
            eeg_baseline = eeg_baseline.transpose(1, 0)
            baseline_mean = np.mean(eeg_baseline, axis=1, keepdims=True)
            eeg_data = eeg_data - baseline_mean

            raw = mne.io.RawArray(eeg_data, info, verbose=False)
            processed = Preprocessing(raw=raw)
            processed.band_pass_filter(0.3, 49);
            # processed.bad_channels_interpolate(thresh1=3, proportion=0.3)
            processed.eeg_ica()
            processed.average_ref();
            eeg_data = processed.raw.get_data(units='V')

            eeg_data = eeg_data.reshape(eeg_data.shape[0],-1,frequency)
            arousal = sub_data['ScoreArousal'][0, 0][trialid, 0]
            valence = sub_data['ScoreValence'][0, 0][trialid, 0]
            arousal = (arousal-1)*2
            valence = (valence-1)*2

            for j in range(eeg_data.shape[1] // eeg_duration):
                sample = eeg_data[:,eeg_duration * j:eeg_duration * (j + 1)]
                print(sample.shape)
                print(np.max(sample , axis=1), np.min(sample , axis=1))
                sample_key = f'{sub_id}-{trialid}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'valence': valence,'arousal': arousal
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()