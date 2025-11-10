import pickle
import lmdb
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
from EMOD_dataset.cross_dataset import EmoDataset_Contra
import torch
from data_preprocess.channel_idx import faced_channel_ids, seed_channel_ids, deap_channel_ids, amigos_channel_ids, dreamer_channel_ids, hci_channel_ids
class EmoDataset(Dataset):
    def __init__(
            self,
            data_dir='./processed_data/FACED',
            channel_idx=None,
            mode='train',
    ):
        super().__init__()
        self.db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
        self.channel_idx = channel_idx

    def __len__(self):
        return len((self.keys))

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
        data = pair['sample']
        data = data.reshape(data.shape[0],-1)
        label = pair['label']
        channel_idx = self.channel_idx
        return data, label, channel_idx

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        channel_idx = np.array([x[2] for x in batch])
        x_data = torch.from_numpy(x_data).float()
        y_label = torch.from_numpy(y_label).long()
        channel_idx = torch.from_numpy(channel_idx).long()
        return x_data, y_label, channel_idx

class LoadDataset(object):
    def __init__(self, datasets_dir, channel_idx, batch_size=32,train_ratio=1.0):
        self.batch_size = batch_size
        self.datasets_dir = datasets_dir
        self.channel_idx = channel_idx
        self.train_ratio = train_ratio

    def get_data_loader(self):
        train_set = EmoDataset(self.datasets_dir, self.channel_idx,mode='train')
        train_collate_fn = train_set.collate
        val_set = EmoDataset(self.datasets_dir, self.channel_idx, mode='val')
        test_set = EmoDataset(self.datasets_dir, self.channel_idx, mode='test')

        if self.train_ratio < 1.0:
            total_len = len(train_set)
            selected_len = int(total_len * self.train_ratio)
            indices = np.random.choice(total_len, selected_len, replace=False)
            train_set = Subset(train_set, indices)


        print(len(train_set), len(val_set), len(test_set))
        print(len(train_set)+len(val_set)+len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.batch_size,
                collate_fn=train_collate_fn,
                shuffle=True,
                num_workers=32,

            ),
            'val': DataLoader(
                val_set,
                batch_size=self.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                num_workers=32,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                num_workers=32,
            ),
        }
        return data_loader

class LoadDataset_viz(object):
    def __init__(self, datasets_dir, channel_idx, batch_size=32,train_ratio=1.0):
        self.batch_size = batch_size
        self.datasets_dir = datasets_dir
        self.channel_idx = channel_idx
        self.train_ratio = train_ratio

    def get_data_loader(self):
        train_set = EmoDataset_Contra(self.datasets_dir, channel_idxs=self.channel_idx)
        train_collate_fn = train_set.collate

        if self.train_ratio < 1.0:
            total_len = len(train_set)
            selected_len = int(total_len * self.train_ratio)
            indices = np.random.choice(total_len, selected_len, replace=False)
            train_set = Subset(train_set, indices)

        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.batch_size,
                collate_fn=train_collate_fn,
                shuffle=True,
                num_workers=32,

            ),
        }
        return data_loader

if __name__ == '__main__':
    datasets_dir = './processed_data/SEED'
    dataset = LoadDataset(datasets_dir, channel_idx=faced_channel_ids, batch_size=32)
    data_loader = dataset.get_data_loader()
    train_loader = data_loader['train']
    val_loader = data_loader['val']


