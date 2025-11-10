import os
import scipy.io as sio
from channel_idx import deap_channels
from main_preprocessing import Preprocessing
import mne
import pandas as pd
import lmdb
import pickle
import numpy as np

# %%
root_dir = './DEAP/data_preprocessed_python'
files = [file for file in os.listdir(root_dir)]
files = sorted(files,key=lambda x: (int(x[1:3])))
print(files)
num_trials = 40
files_dict = {
    'train': files[:],

}
print(files_dict)
dataset = {
    'train': list(),
}

eeg_duration = 10
baseline_duration = 3  # 秒


db_path = './processed_data/DEAP'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=3500*1024*1024)

for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 10
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')
    # else:
    #     eeg_duration = 5
    print(f'Processing {files_key} for {files_dict[files_key]}')
    for file in files_dict[files_key]:
        data_path = os.path.join(root_dir, file)
        print(data_path)
        data_label = pickle.load(open(data_path, 'rb'), encoding='latin1')
        data = data_label['data']

        data = data.reshape(40,40,63,128)
        data = data.transpose(0, 2, 1, 3)
        label_valence = data_label['labels'][:,1]
        label_arousal = data_label['labels'][:,0]
        data = data.transpose(0,2,1,3)
        data = data[:,:32].reshape(40,32,-1)
        print(data.shape)

        for i in range(num_trials):
            samples = data[i]
            info = mne.create_info(ch_names=deap_channels,sfreq=128, ch_types='eeg')
            raw = mne.io.RawArray(samples, info, verbose=False)
            processed = Preprocessing(raw=raw)
            processed.band_pass_filter(0.3, 49);
            processed.eeg_ica()
            processed.average_ref();
            samples = processed.raw.get_data(units='V')
            samples = samples.reshape(32, -1, 128)
            baseline_values = np.mean(samples[:, :baseline_duration], axis=1)
            samples = samples - baseline_values[:, np.newaxis, :]
            samples = samples[:, baseline_duration:]
            for j in range(data.shape[1] // eeg_duration):
                sample = samples[:,eeg_duration * j:eeg_duration * (j + 1)]
                # print(sample.shape, label_valence[i], label_arousal[i])
                sample_key = f'{file}-{i}-{j}'
                data_dict = {
                    'sample': sample, 'valence': label_valence[i]-1,'arousal': label_arousal[i]-1
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()