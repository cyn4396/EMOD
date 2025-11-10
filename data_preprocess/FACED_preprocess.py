from scipy import signal
import os
import lmdb
import pickle
import numpy as np
from scipy.io import loadmat

"""
FACED Dataset Preprocessing Script

This preprocessing pipeline is based on the implementation from the CBraMod project:

- Paper: Wu, J., et al. (2024). 
         "CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding".
         arXiv preprint arXiv:2412.07236.
- URL: https://arxiv.org/abs/2412.07236
- Related Code: https://github.com/wjq-learning/CBraMod/blob/main/preprocessing/preprocessing_faced.py
"""

labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])
root_dir = './FACED/Processed_data'
files = [file for file in os.listdir(root_dir)]
files = sorted(files)

files_dict = {
    'train':files[:80],
    'val':files[80:100],
    'test':files[100:],
}

dataset = {
    'train': list(),
    'val': list(),
    'test': list(),
}

db_path = './processed_data/FACED'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=12 * 1024 * 1024 * 1024)

eeg_duration = 10

for files_key in files_dict.keys():
    for file in files_dict[files_key]:
        f = open(os.path.join(root_dir, file), 'rb')
        array = pickle.load(f)
        print(array.shape)
        eeg = array.reshape(28, 32, 30, 250)

        sub_id = file[3:6]

        for i, (samples, label) in enumerate(zip(eeg, labels)):
            # sample_min = samples.min()
            # sample_max = samples.max()
            #
            # samples = (samples - sample_min)/ (sample_max - sample_min)


            for j in range(30//eeg_duration):
                sample = samples[:, eeg_duration*j:eeg_duration*(j+1), :]
                sample_key = f'{file}-{i}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'label': label,
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)


txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()
