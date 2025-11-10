import numpy as np
import mne
from mne.preprocessing import ICA


class Preprocessing():
    """
    EEG Signal Preprocessing Pipeline

    This preprocessing implementation is inspired by the dataset and methodology described in:

    - Paper: Zheng, W.L., & Lu, B.L. (2023).
             "A Large Finer-grained Affective Computing EEG Dataset".
             *Scientific Data*, Nature Portfolio.
    - URL: https://www.nature.com/articles/s41597-023-02650-w
    - Related Project DOI: https://doi.org/10.7303/syn50614194
    """
    def __init__(self, raw, nchns=62, freq=200):
        # Modify the montage
        self.nchns = nchns
        self.freq = freq
        chn_names = raw.info['ch_names']
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage,on_missing='ignore')
        # Match the corresponding montage to their index
        self.montage_index = dict(zip(np.arange(self.nchns), chn_names))
        # split out the data matrix
        self.raw = raw
        # Ptr operation
        self.data = self.raw.get_data()

    def band_pass_filter(self, l_freq, h_freq):
        # The default filter is a FIR filter,
        # therefore,  the input [l_freq,h_freq] is
        # [lower pass-band edge, higher pass-band edge]
        self.raw.filter(l_freq, h_freq)

    # Should cut the data into pieces than do the interpolation
    def bad_channels_interpolate(self, thresh1=None, thresh2=None, proportion=0.3):
        data = self.raw.get_data()
        # We found that the data shape of epochs is 3 dims
        # print(data.shape)
        if len(data.shape) > 2:
            data = np.squeeze(data)
        Bad_chns = []
        value = 0
        # Delete the much larger point
        if thresh1 != None:
            md = np.median(np.abs(data))
            value = np.where(np.abs(data) > (thresh1 * md), 0, 1)[0]
        if thresh2 != None:
            value = np.where((np.abs(data)) > thresh2, 0, 1)[0]
        # Use the standard to pick out the bad channels
        Bad_chns = np.argwhere((np.mean((1 - value), axis=0) > proportion))
        if Bad_chns.size > 0:
            self.raw.info['bads'].extend([self.montage_index[str(bad)] for bad in Bad_chns])
            print('Bad channels: ', self.raw.info['bads'])
            self.raw = self.raw.interpolate_bads()
        else:
            print('No bad channel currently')

    # You can manually exclude the ICA elements
    # Our auto preprocessing method is more effective for eye blink removal
    def eeg_ica(self, check_ica=None):
        ica = ICA(max_iter='auto', method='fastica')
        raw_ = self.raw.copy()
        ica.fit(self.raw)
        eog_chs = self.raw.ch_names
        if 'Fp1' in eog_chs:
            ch_name1 = 'Fp1'
        elif 'FP1' in eog_chs:
            ch_name1 = 'FP1'
        else:
            ch_name1 = 'AF3'
        if 'Fp2' in eog_chs:
            ch_name2 = 'Fp2'
        elif 'FP2' in eog_chs:
            ch_name2 = 'FP2'
        else:
            ch_name2 = 'AF4'

        # Plot different elements of the signals
        eog_indices1, eog_score1 = ica.find_bads_eog(self.raw, ch_name=ch_name1)
        eog_indices2, eog_score2 = ica.find_bads_eog(self.raw, ch_name=ch_name2)
        eog_indices = list(set(eog_indices1 + eog_indices2))
        ica.exclude = eog_indices

        if check_ica == True:
            print('Already use:', eog_indices)
            ica.plot_sources(raw_)
            ica.plot_components()
            eog_indices = input("Exclude ?")
            eog_indices = eog_indices.split(" ")
            eog_indices = list(map(int, eog_indices))
            ica.exclude = eog_indices

        ica.apply(self.raw)

    def average_ref(self):
        self.raw.set_eeg_reference(ref_channels='average')