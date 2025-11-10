import scipy
from scipy import signal
import os
import lmdb
import pickle
import numpy as np
import mne
from main_preprocessing import Preprocessing
import pandas as pd

"""
SEED-V Dataset Preprocessing Script

This preprocessing pipeline is based on the implementation from the CBraMod project:

- Paper: Wu, J., et al. (2024). 
         "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding".
         arXiv preprint arXiv:2412.07236.
- URL: https://arxiv.org/abs/2412.07236
- Related Code: https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_SEEDV.py
"""


useless_ch = ['M1', 'M2', 'VEO', 'HEO']

df = pd.read_excel('./SEEDIV/data/Channel Order.xlsx', header=None)
ch_names = df[0].to_list()
info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')

trials_of_sessions = {
    '1': {'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
          'end': [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]},

    '2': {'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
          'end': [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]},

    '3': {'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
          'end': [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]},
}
labels_of_sessions = {
    '1': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0, ],
    '2': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
    '3': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0, ],
}
# 0: Disgust, 1: Fear, 2: Sad, 3:Neutral, 4:Happy


root_dir = './SEEDV/raw_data'
files = [file for file in os.listdir(root_dir)]
files = sorted(files, key=lambda x: int(x.split('_')[0]))
print(files)
print(len(files))

trials_split = {
    'train': range(5),
    'val': range(5, 10),
    'test': range(10, 15),
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

# db = lmdb.open('/data/datasets/BigDownstream/SEED-V/processed', map_size=15614542346)

eeg_duration = 1
db_path = './processed_data/SEEDV'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=15614542346)


for file in files:
    raw = mne.io.read_raw_cnt(os.path.join(root_dir, file), preload=True)
    raw.drop_channels(useless_ch)
    raw.resample(200)
    processed = Preprocessing(raw=raw)
    processed.band_pass_filter(0.3, 49);
    processed.average_ref();
    data_matrix = processed.raw.get_data(units='uV')
    session_index = file.split('_')[1]
    data_trials = [
        data_matrix[:,
        trials_of_sessions[session_index]['start'][j] * 200:trials_of_sessions[session_index]['end'][j] * 200]
        for j in range(15)]
    labels = labels_of_sessions[session_index]
    for files_key in trials_split.keys():
        for index in trials_split[files_key]:
            data_trial = data_trials[index]

            label = labels[index]
            print(data_trial.shape)
            data = data_trial.reshape(62, -1, 200)
            eeg_len = data.shape[1]
            for i in range(eeg_len // eeg_duration):
                sample = data[:, eeg_duration * i:eeg_duration * (i + 1), :]
                print(sample.shape, label)
                sample_key = f'{file}-{index}-{i}'
                data_dict = {
                    'sample': sample, 'label': label
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()