# 各个数据集的通道列表
faced_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
                  'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4',
                  'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2', 'A1', 'A2']

seed_channels = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6',
                 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3',
                 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5',
                 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2']

deap_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7',
                 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4',
                 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']

amigos_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6',
                   'F4', 'F8', 'AF4']

dreamer_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6',
                    'F4', 'F8', 'AF4']

hci_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3',
    'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2',
    'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']


mped_channels = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4',
    'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3',
    'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz',
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
    'CB1', 'O1', 'Oz', 'O2', 'CB2'
]
mme_channels = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "Cz", "C3", "C4", "T3", "T4", "P3", "P4", "T5", "T6", "O1", "O2"]


def get_channel_idx():
    # 所有通道合并，统一大小写
    all_channels = (faced_channels + seed_channels + deap_channels +
                    amigos_channels + dreamer_channels + hci_channels+mped_channels+mme_channels)

    # 标准化名称为大写，去重、排序
    unique_channels = sorted(set(ch.upper() for ch in all_channels))

    # 编号字典
    channel_to_id = {channel: idx+1 for idx, channel in enumerate(unique_channels)}
    return channel_to_id

channel_to_id = get_channel_idx()
faced_channel_ids = [channel_to_id[channel.upper()] for channel in faced_channels]
seed_channel_ids = [channel_to_id[channel.upper()] for channel in seed_channels]
deap_channel_ids = [channel_to_id[channel.upper()] for channel in deap_channels]
amigos_channel_ids = [channel_to_id[channel.upper()] for channel in amigos_channels]
dreamer_channel_ids = [channel_to_id[channel.upper()] for channel in dreamer_channels]
hci_channel_ids = [channel_to_id[channel.upper()] for channel in hci_channels]
mped_channels_ids = [channel_to_id[channel.upper()] for channel in mped_channels]
mme_channels_ids = [channel_to_id[channel.upper()] for channel in mme_channels]


if __name__ == '__main__':
    channel_to_id = get_channel_idx()
    print(channel_to_id)
    print(len(mped_channels))
    faced_channels = ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 'FC6',
                      'Cz', 'C3', 'C4', 'T7', 'T8', 'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4',
                      'P7', 'P8', 'PO3', 'PO4', 'Oz', 'O1', 'O2', 'A1', 'A2']

    faced_channel_ids = [channel_to_id[channel.upper()] for channel in faced_channels]
    print("FACED：", faced_channel_ids)

