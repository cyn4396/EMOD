import os
import scipy.io as sio
import numpy as np
from main_preprocessing import Preprocessing
import mne
import lmdb
import pickle
from channel_idx import amigos_channels


"""
AMIGOS Dataset Preprocessing Script

This preprocessing pipeline is implemented based on the official code provided in:

- Paper: Zhang, L., et al. (2024).
         "TorchEEGEMO: A Deep Learning Toolbox Towards EEG-based Emotion Recognition".
         *Expert Systems with Applications*, Elsevier.
- URL: https://www.sciencedirect.com/science/article/pii/S0957417424004159
- Related Project: https://github.com/torcheeg/torcheeg/tree/main
"""


root_path = './AMIGOS/data_preprocessed'
file_list = os.listdir(root_path)
file_list.sort()
skipped_sub = [9, 12, 21, 22, 23, 24, 33]
filtered_filelist = [file for file in file_list if int(file[19:21]) not in skipped_sub]
file_list = filtered_filelist
print(len(file_list))

files_dict = {
    'train': file_list[:],
}

dataset = {
    'train': list(),
}

frequency = 128
num_videos = 16
num_channel = 14
eeg_duration = 10
baseline_duration = 5  # Baseline duration in seconds
baseline_samples = baseline_duration * frequency

db_path = './processed_data/AMIGOS'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=4000*1024*1024)

for files_key in files_dict.keys():
    if files_key == 'train':
        eeg_duration = 10
        print(f'Processing {files_key} files with duration {eeg_duration} seconds')
    # else:
    #     eeg_duration = 5
    for file in files_dict[files_key]:

        data_path = os.path.join(root_path, file)
        print(data_path)
        data_label = mat_data = sio.loadmat(data_path)
        joined_data = data_label['joined_data']
        emo_label = data_label['labels_selfassessment']

        for i in range(num_videos):
            video_data = joined_data[0][i].T
            video_data = video_data[:num_channel]
            info = mne.create_info(ch_names=amigos_channels, sfreq=128, ch_types='eeg')
            raw = mne.io.RawArray(video_data, info, verbose=False)
            processed = Preprocessing(raw=raw)
            processed.band_pass_filter(0.3, 49);
            processed.eeg_ica()
            processed.average_ref();
            samples = processed.raw.get_data(units='V')

            baseline = np.mean(video_data[:baseline_samples], axis=1)
            video_data = video_data - baseline[:,np.newaxis]
            video_data = video_data[:, baseline_samples:]

            eeg_len = video_data.shape[1]
            label_valence = emo_label[0,i][0,1]
            label_arousal = emo_label[0,i][0,0]
            for j in range(eeg_len // (eeg_duration*frequency)):
                sample = video_data[:, eeg_duration*frequency * j:eeg_duration*frequency * (j + 1)]

                sample = sample.reshape(sample.shape[0], eeg_duration, frequency)
                # print(sample.shape)
                # print(np.max(sample , axis=1), np.min(sample , axis=1))
                sample_key = f'{file}-{i}-{j}'
                print(sample_key)
                data_dict = {
                    'sample': sample, 'valence': label_valence-1,'arousal': label_arousal-1
                }
                txn = db.begin(write=True)
                txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
                txn.commit()
                dataset[files_key].append(sample_key)
#
txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()