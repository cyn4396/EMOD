import os
import scipy.io as sio
from main_preprocessing import Preprocessing
import mne
import pandas as pd
import lmdb
import pickle
import numpy as np

# %%
file_path = './SEED-VII/EEG_preprocessed'
files = [file for file in os.listdir(file_path)]
files = sorted(files,key=lambda x: (int(x.split('.')[0])))
print(files)


label = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0,
         4, 3, 5, 1, 6, 6, 1, 5, 3, 4, 4, 3, 5, 1, 6, 6, 1, 5, 3, 4,
         0, 6, 2, 5, 4, 4, 5, 2, 6, 0, 0, 6, 2, 5, 4, 4, 5, 2, 6, 0,
         2, 3, 5, 6, 0, 0, 6, 5, 3, 2, 2, 3, 5, 6, 0, 0, 6, 5, 3, 2]
# label 0-6:Happy, Neutral, Disgust, Sad, Anger, Fear, Surprise
Arousal = [6, 4, 5, 5, 6, 6, 6]
Valence = [7, 4, 2, 3, 3, 3, 4]

trial_idx = [i for i in range(80)]
files_dict = {
    'train': files[:],
}


dataset = {
    'train': list(),
}

eeg_duration = 10
df = pd.read_excel('./SEED-VII/Channel Order.xlsx', header=None)
ch_names = df[0].to_list()
info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')

db_path = './processed_data/SEEDVII'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=50*1024*1024*1024)

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        data_path = os.path.join(file_path, file)
        print(data_path)
        data = sio.loadmat(data_path)
        for trial_id in trial_idx:
            trial_data = data[str(trial_id+1)]
            print(trial_data.shape)
            raw = mne.io.RawArray(trial_data, info)
            processed = Preprocessing(raw=raw)
            processed.band_pass_filter(0.3, 49);
            processed.eeg_ica()
            processed.average_ref();
            data_i = processed.raw.get_data(units='V')
            samples = data_i.reshape(62, -1, 200)
            eeg_len = samples.shape[1]
            print(trial_id, eeg_len)
            for j in range(eeg_len // eeg_duration):
                sample = samples[:, eeg_duration * j:eeg_duration * (j + 1), :]
                print(sample.shape)
                sample_key = f'{file}-{trial_id}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label[trial_id], 'arousal': Arousal[label[trial_id]],'valence': Valence[label[trial_id]],
                }

                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()