import os
import scipy.io as sio
import lmdb
import pickle
import pandas as pd
import mne
from main_preprocessing import Preprocessing
import numpy as np


df = pd.read_excel('./SEEDIV/data/Channel Order.xlsx', header=None)
ch_names = df[0].to_list()
info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types='eeg')


labels_of_sessions = {
    '1': [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    '2': [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    '3': [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

#(Label) 0: neutral, 1: sad, 2: fear, 3: happy
Arousal = [4, 5, 6, 6]
Valence = [4, 3, 3, 7]


root_dir = './SEEDIV/data/eeg_raw_data/'
file_list = []
for i in range(1,4):
    file_path = os.path.join(root_dir, str(i))
    files = [os.path.join(str(i),file) for file in os.listdir(file_path)]
    files = sorted(files, key=lambda x: (int(x.split('/')[1].split('_')[0]), x.split('/')[0]))
    file_list += files
file_list = sorted(file_list, key=lambda x: int(x.split('/')[1].split('_')[0]))
print(file_list)

files_dict = {
    'train':file_list[:],
    # 'val':file_list[3*9:3*12],
    # 'test':file_list[3*12:],
}
print(files_dict)
#

#
dataset = {
    'train': list(),
    # 'val': list(),
    # 'test': list(),
}
#
#
eeg_duration = 10
db_path = './processed_data/SEEDIV'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=15614542346)

for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 10
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')

    for file in files_dict[files_key]:
        print(file)
        data_path = os.path.join(root_dir, file)
        print(data_path)
        data = sio.loadmat(data_path)
        i = 0
        for keys in data:
            if 'eeg' in keys:
                raw = mne.io.RawArray(data[keys], info)
                processed = Preprocessing(raw=raw)
                processed.band_pass_filter(0.3, 49);
                processed.eeg_ica()
                processed.average_ref();
                data_i = processed.raw.get_data(units='V')[:, :-1]
                samples = data_i.reshape(62, -1, 200)
                eeg_len = samples.shape[1]
                label = labels_of_sessions[file[0]][i]
                print(keys, eeg_len,label)
                for j in range(eeg_len // eeg_duration):
                    sample = samples[:, eeg_duration * j:eeg_duration * (j + 1), :]
                    # print(np.max(sample, axis=1), np.min(sample, axis=1))
                    sample_key = f'{file}-{i}-{j}'
                    print(sample_key)
                    data_dict = {
                        'sample': sample, 'label': label, 'arousal': Arousal[label],'valence': Valence[label],
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