from scipy import signal
import os
import lmdb
import pickle
import numpy as np
from scipy.io import loadmat
import pandas as pd

label_each_trial = np.array([0]*8+[1]*8+[2]*16)
print(label_each_trial)

len_each_trial = [6000, 8700, 8700, 9000, 8700, 9000, 9000, 8400, 6600, 6900,
        9000, 9300, 7200, 8400, 8700, 8100, 8700, 8400, 8100, 9000,
        9300, 7800, 9600, 7200, 7800, 6600, 9000, 6900, 7200, 6900,
        6000, 8700]

root_path = './MME/raw_data'
file_list = os.listdir(root_path)
file_list = sorted(file_list, key=lambda x: int(x))
print(file_list)


frequency = 300
eeg_duartion = 10

sub_dict = {
    'train': file_list[:],
}

dataset = {
    'train': list(),
}
#
db_path = './processed_data/MME'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=4 * 1024 * 1024 * 1024)
#
eeg_duration = 10
#
for files_key in sub_dict.keys():
    for sub_path in sub_dict[files_key]:

        sub_file_path = os.path.join(root_path, sub_path,'datas.mat')
        av_path = os.path.join(root_path, sub_path, sub_path, 'Arousal_Valence.csv')
        print(sub_file_path)
        data_sub = loadmat(sub_file_path)['eeg_datas']
        av_df = pd.read_csv(av_path, header=None)
        data_sub = data_sub[:18,:]
        print(data_sub.shape)

        trial_start = 0
        for trial_id in range(32):
            trial_av = av_df[av_df[0] == trial_id]
            arousal_label = trial_av[1].values[0]
            valence_label = trial_av[2].values[0]
            data_trial = data_sub[:,trial_start:trial_start+len_each_trial[trial_id]]
            trial_start += len_each_trial[trial_id]
            trial_len = data_trial.shape[1]//frequency
            for i in range(trial_len//eeg_duration):
                sample_key = f'sub-{sub_path}_trial-{trial_id}_segment-{i}'
                sample = data_trial[:, i*eeg_duration*frequency:(i+1)*eeg_duration*frequency]
                sample = sample.reshape(-1,eeg_duartion,frequency)
                print(np.max(sample, axis=1), np.min(sample, axis=1))
                label = label_each_trial[trial_id]
                print(label, arousal_label, valence_label, trial_id, i, sample_key)
                data_dict = {
                    'sample': sample, 'label': label,'arousal': arousal_label-1, 'valence': valence_label-1,
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
