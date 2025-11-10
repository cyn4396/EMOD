import os
import scipy.io as sio
from main_preprocessing import Preprocessing
import mne
import pandas as pd
import lmdb
import pickle
import numpy as np

# %%
file_path = './SEED/Preprocessed_EEG'
file_list = os.listdir(file_path)
file_list = sorted(file_list, key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))
print(file_list)

label = [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]
print(label)

files_dict = {
    'train': file_list[0::3],
    'val':   file_list[1::3],
    'test':  file_list[2::3],
}
print(files_dict)

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

eeg_duration = 5
df = pd.read_excel('./SEED/channel-order.xlsx', header=None)
ch_names = df[0].to_list()
info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')

db_path = './processed_data/SEED'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=15614542346)


for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 5
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')
    else:
        eeg_duration = 5
    for file in files_dict[files_key]:
        data_path = os.path.join(file_path, file)
        print(data_path)
        data = sio.loadmat(data_path)
        i = 0
        for keys in data:
            if 'eeg' in keys:
                raw = mne.io.RawArray(data[keys], info)
                processed = Preprocessing(raw=raw)
                processed.band_pass_filter(0.3, 49);
                processed.average_ref();
                data_i = raw.get_data(units='V')[:, :-1]
                samples = data_i.reshape(62, -1, 200)
                eeg_len = samples.shape[1]
                print(keys, eeg_len)
                for j in range(eeg_len // eeg_duration):
                    sample = samples[:, eeg_duration * j:eeg_duration * (j + 1), :]
                    sample_key = f'{file}-{i}-{j}'
                    print(sample_key)
                    data_dict = {
                        'sample': sample, 'label': label[i]
                    }
                    txn = db.begin(write=True)
                    txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                    txn.commit()
                    dataset[files_key].append(sample_key)
                i+=1

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()