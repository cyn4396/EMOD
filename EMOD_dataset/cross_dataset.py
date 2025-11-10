import pickle
import lmdb
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
import random
from data_preprocess.channel_idx import faced_channel_ids, seed_channel_ids, deap_channel_ids, \
    amigos_channel_ids, dreamer_channel_ids, hci_channel_ids,mped_channels_ids,mme_channels_ids

def apply_transform(data, channels, transform):
    idx = random.randint(0, len(transform) - 1)
    t = transform[idx]
    data, channels = t(data, channels)
    return data, channels

class EEGContrastiveDataset(Dataset):
    def __init__(self, datasets, transform=None, repeat=1):
        """
        datasets: dict, 格式如 {'FACED': (data1, labels1), 'SEED': (data2, labels2)}
        transform: 可选的对样本进行aug的函数
        """
        self.datasets = datasets
        self.transform = transform

        # 存储处理后的数据
        self.dataset = {}
        self.num_classes = 9
        self.repeat = repeat
        self.sample_num = 1
        # 遍历每个数据集，将数据按标签分类
        for dataset_name, dataset in datasets.items():
            dataset_entries = {label: [] for label in range(9)}  # 按标签分类的数据
            for i in range(len(dataset)):
                data, arousal, valence, channels = dataset[i]  # 通过 __getitem__ 获取数据和标签
                label = int((valence) // 3)+int((arousal)// 3)*3
                dataset_entries[label].append([data,arousal,valence,channels])  # 根据标签将数据添加到对应的分类中
            # 存储数据集名称、标签和样本
            self.dataset[dataset_name] = dataset_entries

    def __len__(self):
        # 返回所有数据集的样本总数
        total_samples = 0
        for dataset_name, entries in self.dataset.items():
            total_samples += sum(len(data_list) for data_list in entries.values())
        return total_samples * self.repeat // (len(self.dataset.keys()))

    def __getitem__(self, idx):
        all_data = []
        all_arousal = []
        all_valence = []
        all_channels = []
        label_idx = idx % self.num_classes
        for dataset_name, entries in self.dataset.items():
            # 按标签从数据集中选择样本
            while len(entries[label_idx]) == 0:
                label_idx = (label_idx + 1) % self.num_classes
            label_i_data = entries[label_idx]
            # 随机选择一个样本
            indices = torch.randint(0, len(label_i_data), (1,))
            data, arousal, valence, channels = label_i_data[indices.item()]
            # 转为 Tensor
            data = torch.from_numpy(data).float()
            arousal = torch.tensor(arousal, dtype=torch.long)
            valence = torch.tensor(valence, dtype=torch.long)
            channels = torch.tensor(channels, dtype=torch.long)

            # 如果需要 reshape 或 transform
            if self.transform is not None:
                data, channels = apply_transform(data, channels, self.transform)
            else:
                data = data.reshape(data.shape[0], -1)  # e.g. (32, 30*250)
            all_data.append(data)
            all_arousal.append(arousal)
            all_valence.append(valence)
            all_channels.append(channels)

        return all_data, all_arousal, all_valence, all_channels

class EmoDataset_Contra(Dataset):
    def __init__(
            self,
            data_dir='./processed_data/Faced',
            mode='train',
            channel_idxs=None,
    ):
        super().__init__()

        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.channel_idxs = channel_idxs

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        arousal = pair['arousal']
        valence = pair['valence']

        arousal = int(arousal)
        valence = int(valence)

        channel_idxs = self.channel_idxs
        return data, arousal, valence, channel_idxs

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_arousal = np.array([x[1] for x in batch])
        y_valence = np.array([x[2] for x in batch])
        channel_idx = np.array([x[3] for x in batch])

        x_data = torch.from_numpy(x_data).float()
        y_arousal = torch.from_numpy(y_arousal).long()
        y_valence = torch.from_numpy(y_valence).long()
        channel_idx = torch.from_numpy(channel_idx).long()

        return x_data, y_arousal, y_valence, channel_idx

# def

def get_cross_dataset():

    DEAP_train_data = EmoDataset_Contra(data_dir='./processed_data/DEAP', mode='train',channel_idxs=deap_channel_ids)
    SEEDIV_train_data = EmoDataset_Contra(data_dir='./processed_data/SEEDIV', mode='train',channel_idxs=seed_channel_ids)
    SEEDVII_train_data = EmoDataset_Contra(data_dir='./processed_data/SEEDVII', mode='train',channel_idxs=seed_channel_ids)
    AMIGOS_train_data = EmoDataset_Contra(data_dir='./processed_data/AMIGOS', mode='train',channel_idxs=amigos_channel_ids)
    DREAMER_train_data = EmoDataset_Contra(data_dir='./processed_data/DREAMER', mode='train',channel_idxs=dreamer_channel_ids)
    HCI_train_data = EmoDataset_Contra(data_dir='./processed_data/MAHNOB', mode='train',channel_idxs=hci_channel_ids)
    MPED_train_data = EmoDataset_Contra(data_dir='./processed_data/MPED', mode='train',channel_idxs=mped_channels_ids)
    MME_train_data = EmoDataset_Contra(data_dir='./processed_data/MME', mode='train',channel_idxs=mme_channels_ids)


    dataset_train = {'DEAP': DEAP_train_data,'AMIGOS': AMIGOS_train_data,'MME': MME_train_data,
                     'SEEDVII': SEEDVII_train_data,'DREAMER': DREAMER_train_data,'HCI': HCI_train_data,
                    'SEEDIV': SEEDIV_train_data, 'MPED': MPED_train_data}


    print(f'using {len(dataset_train.keys())} datasets: {dataset_train.keys()}')

    return dataset_train

def get_num_samples(dataset_name):
    """
    获取所有数据集的样本总数
    :return: int, 样本总数
    """
    dataset = EmoDataset_Contra(data_dir=f'./processed_data/{dataset_name}', mode='train',channel_idxs=deap_channel_ids)
    return len(dataset)


if __name__ == '__main__':
    for dataset_name in ['Faced','DEAP', 'SEEDIV','SEEDVII',  'AMIGOS',  'MPED', 'DREAMER','MAHNOB', 'MME']:
        num_samples = get_num_samples(dataset_name)
        print(f"{dataset_name} dataset has {num_samples} samples.")

    dataset_train= get_cross_dataset()
    pretrain_data = EEGContrastiveDataset(datasets=dataset_train,transform=None)

    train_loader = DataLoader(
        pretrain_data,
        batch_size=4*9,
        shuffle=True,
        num_workers=16,
        drop_last=True
    )



