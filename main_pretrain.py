from torch.utils.data import DataLoader
from utils.contrastive_learning import contrastive_train
from model.EMOD import EMOD_pretrain
from utils.train_func import setup_seed, SaveCheckpointsCallback
import torch
from utils.sup_con import SupConLoss, SupConLoss_dist_soft
from utils.data_augment import add_gaussian_noise, channel_mask, time_mask, eeg_filter, shuffle_channels ,no_aug, time_flip
from EMOD_dataset import get_cross_dataset, EEGContrastiveDataset
import argparse
import os

parser = argparse.ArgumentParser(description='EEG Contrastive Learning')

parser.add_argument('--eeg_duration', type=int, default=10, help='EEG segment duration (seconds)')
parser.add_argument('--batch_size', type=int, default=36, help='Batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--n_filters', type=int, default=128, help='Number of filters in CNN')
parser.add_argument('--random_seed', type=int, default=2025, help='Random seed')
parser.add_argument('--device_id', type=int, default=0, help='GPU device id')
parser.add_argument('--transformer_depth', type=int, default=3, help='Depth of the transformer')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')


args = parser.parse_args()

for k, v in sorted(vars(args).items()):
    print(k, '=', v)

batch_size = args.batch_size
lr = args.lr
wd = args.wd
num_epochs = args.num_epochs
n_filters = args.n_filters
random_seed = args.random_seed
device_id = args.device_id
setup_seed(random_seed)

device = (
    f"cuda:{device_id}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)


transform = [
    lambda x,y: no_aug(x,y),
    lambda x, y: add_gaussian_noise(x, y, noise_factor=0.1),
    lambda x,y: no_aug(x,y),
    lambda x,y: shuffle_channels(x,y, prob=0.5),
    lambda x,y: no_aug(x,y),
    lambda x,y: time_mask(x,y, mask_prob=0.2),
]

dataset_train = get_cross_dataset()
contrastive_dataset_train = EEGContrastiveDataset(datasets=dataset_train, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(
    contrastive_dataset_train,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    drop_last=True
)

for i, (data, arousal, valence, channels) in enumerate(train_loader):
    for j in range(len(data)):
        print(data[j].shape)
        print(j, arousal[j].shape)
        print(j, valence[j].shape)
        print(j, channels[j].shape)
    break
print(len(train_loader))


# 训练模型
model = EMOD_pretrain(n_filters=n_filters, depth=args.transformer_depth, dropout=args.dropout)

cp_path = './check_points/Pretraining'
save_model_name = f'contrastive_{len(transform)-1}aug_batch{batch_size}_{args.temperature}_soft_label_maxdist{args.max_dist}'
best_name = f'{save_model_name}_best_model.ckpt'
last_name = f'{save_model_name}_last_model.ckpt'
print(cp_path)
print(best_name)



# 使用多卡
device_ids = [0,1,2,3]
model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-7)
loss = SupConLoss_dist_soft(temperature=args.temperature)


save_ckpt_callback = SaveCheckpointsCallback(cp_path, save_step=1, save_best_only=True,
                                                 file_name=best_name)

record_dict = contrastive_train(model, loss, train_loader, num_epochs, optimizer, scheduler,
                    device=device, save_ckpt_callback=save_ckpt_callback, max_norm=3.0)
# save_last_model
torch.save(model.state_dict(), os.path.join(cp_path, last_name))
