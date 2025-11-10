import os
from main_preprocessing import Preprocessing
from scipy.io import loadmat
import lmdb
import pickle
import numpy as np
import xmltodict
import glob
import mne
from channel_idx import hci_channels
# from

# %%


"""
MAHNOB-HCI Dataset Preprocessing Script

This preprocessing pipeline is implemented based on the official code provided in:

- Paper: Zhang, L., et al. (2024).
         "TorchEEGEMO: A Deep Learning Toolbox Towards EEG-based Emotion Recognition".
         *Expert Systems with Applications*, Elsevier.
- URL: https://www.sciencedirect.com/science/article/pii/S0957417424004159
- Related Project: https://github.com/torcheeg/torcheeg/tree/main
"""


root_path = './HCI-tagging/Sessions'
records = os.listdir(root_path)
print(records)

dataset = {
    'train': list(),
}


frequency = 128
sampling_rate = 128
num_channel = 32
eeg_duration = 10


db_path = './processed_data/MAHNOB'
if not os.path.exists(db_path):
    os.makedirs(db_path)
db = lmdb.open(db_path, map_size=4000*1024*1024)

for i in range(len(records)):
    record = records[i]
    trial_dir = os.path.join(root_path, record)
    # record the common meta info for the trial
    label_file = os.path.join(trial_dir, 'session.xml')
    # print(label_file)
    with open(label_file) as f:
        label_info = xmltodict.parse('\n'.join(f.readlines()))
    # print(label_info)
    subid = label_info['session']['subject']['@id']
    files_key = 'train'

    if '@feltArsl' not in label_info['session']:
        continue
    arousal = label_info['session']['@feltArsl']
    arousal = int(arousal)-1
    valence = label_info['session']['@feltVlnc']
    valence = int(valence)-1
    label = label_info['session']['@feltEmo']
    label = int(label)
    if label==11:
        label=7
    elif label==12:
        label=8
    print('-'*30,arousal,valence)
    # extract signals
    sample_file = glob.glob(str(os.path.join(trial_dir, '*.bdf')))[0]
    num_baseline = 30
    baseline_chunk_size = 128
    raw = mne.io.read_raw_bdf(sample_file,
                              preload=True,
                              stim_channel='Status')
    events = mne.find_events(raw, stim_channel='Status')
    montage = mne.channels.make_standard_montage(kind='biosemi32')
    raw.set_montage(montage, on_missing='ignore')
    print(raw.ch_names[:num_channel])
    # pick channels
    raw.pick_channels(raw.ch_names[:num_channel])
    processed = Preprocessing(raw)
    processed.band_pass_filter(0.3, 49);
    # processed.bad_channels_interpolate(thresh1=3, proportion=0.3)
    processed.eeg_ica()
    processed.average_ref();
    raw = processed.raw

    start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
    # extract baseline signals
    trial_baseline_raw = raw.copy().crop(raw.times[0],
                                         raw.times[end_samp])
    trial_baseline_raw = trial_baseline_raw.resample(sampling_rate)
    # trial_baseline_sample = trial_baseline_raw.to_data_frame().to_numpy(
    # )[:, 1:].swapaxes(1, 0)  # channel(32), timestep(30 * 128)
    trial_baseline_sample = trial_baseline_raw.get_data(units='uV')
    print(trial_baseline_sample.shape)
    trial_baseline_sample = trial_baseline_sample[:, :num_baseline *
                                                      baseline_chunk_size]
    trial_baseline_sample = trial_baseline_sample.reshape(
        num_channel, num_baseline,
        baseline_chunk_size).mean(axis=1)  # channel(32), timestep(128)
    # extract experimental signals
    trial_raw = raw.copy().crop(raw.times[start_samp],
                                raw.times[end_samp])
    trial_raw = trial_raw.resample(sampling_rate)

    # trial_samples = trial_raw.to_data_frame().to_numpy()[:,
    #                 1:].swapaxes(1, 0)
    trial_samples = trial_raw.get_data(units='uV')
    baseline_mean = trial_baseline_sample.mean(axis=1)  # shape: (num_channel,)
    trial_samples -= baseline_mean[:, None]
    trial_samples = trial_samples*0.1

    for j in range(trial_samples.shape[1] // (eeg_duration*frequency)):
        sample = trial_samples[:,eeg_duration * frequency * j:eeg_duration * frequency * (j + 1)]
        sample = sample.reshape(num_channel, -1, frequency)
        print(f'Sample shape: {sample.shape}')
        sample_key = f'{i}-{j}'
        data_dict = {
            'sample': sample, 'label': label,'valence': valence,'arousal': arousal
        }
        txn = db.begin(write=True)
        txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
        txn.commit()
        dataset[files_key].append(sample_key)

txn = db.begin(write=True)
txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset))
txn.commit()
db.close()