from EMOD_dataset import LoadDataset
from utils.train_func import train, setup_seed, evaluate, SaveCheckpointsCallback
from model.EMOD import EMOD
import argparse
import os
import torch
from data_preprocess.channel_idx import faced_channel_ids,seed_channel_ids

parser = argparse.ArgumentParser(description='EEG Contrastive Learning')
parser.add_argument('--dataset', type=str, default='Faced', help='Dataset name')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epdochs')
parser.add_argument('--n_filters', type=int, default=128, help='Number of filters in CNN')
parser.add_argument('--random_seed', type=int, default=2025, help='Random seed')
parser.add_argument('--device_id', type=int, default=1, help='GPU device id')
parser.add_argument('--num_classes', type=int, default=9, help='Number of classes')
parser.add_argument('--load_pretrain_model', type=int, default=1, help='Whether to load pretrained model')
parser.add_argument('--transformer_depth', type=int, default=3, help='Depth of the transformer')
parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
parser.add_argument('--backbone_lr', type=float, default=0.5, help='Learning rate of backbone model')
args = parser.parse_args()


for k, v in sorted(vars(args).items()):
    print(k, '=', v)

# 设置设备
device = (
    f"cuda:{args.device_id}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")
datasets_dir = f'./processed_data/{args.dataset}'
print(f"Using dataset: {datasets_dir}")
random_seed = args.random_seed
device = (
    f"cuda:{args.device_id}"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(device)



lr = args.lr
wd = args.wd
num_epochs = args.num_epochs
n_filters = args.n_filters
num_classes = args.num_classes

if 'Faced' in args.dataset:
    channel_ids = faced_channel_ids
    frequency = 250
elif 'SEED' in args.dataset:
    channel_ids = seed_channel_ids
    frequency = 200
else:
    channel_ids = None
    frequency = 128


setup_seed(random_seed)
print(f"---------Using random seed: {random_seed}-----------")
dataset = LoadDataset(datasets_dir,channel_idx=channel_ids,batch_size=args.batch_size,train_ratio=1.0)
data_loader = dataset.get_data_loader()

train_loader = data_loader['train']
val_loader = data_loader['val']
test_loader = data_loader['test']

for i, (data, label, channels_idx) in enumerate(train_loader):
    channels = data.shape[1]
    eeg_len = data.shape[2]
    print(i, data.shape, label.shape, channels_idx.shape)
    print(label)
    break

for i, (data, label, channels_idx) in enumerate(val_loader):
    channels = data.shape[1]
    eeg_len = data.shape[2]
    print(i, data.shape, label.shape, channels_idx.shape)
    # print(label)
    break

model = EMOD(n_filters=n_filters, depth=args.transformer_depth, dropout=args.dropout, num_classes=num_classes, frequency=frequency,
             channels=channels, eeg_len=eeg_len,dataset=args.dataset)


optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)



if args.load_pretrain_model:
    cp_path = './Pretraining_weights'
    ckpt_name = 'EMOD_pretrained.ckpt'
    state_dict = torch.load(os.path.join(cp_path, ckpt_name), map_location=device)
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print('load pretrained model from ', os.path.join(cp_path, ckpt_name))
    optimizer = torch.optim.Adam([
        {"params": model.embd.parameters(), "lr": lr* args.backbone_lr},
        {"params": model.axial_transformer.parameters(), "lr": lr* args.backbone_lr},
        {"params": model.pos_embeddings.parameters(), "lr": lr* args.backbone_lr},
        {"params": model.classifier.parameters(), "lr": lr},
    ], lr=lr, weight_decay=wd)

    opt_param_ids = set(id(p) for group in optimizer.param_groups for p in group['params'])
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in opt_param_ids:
            print(f"⚠️ Parameter {name} is NOT in optimizer!")

cp_path = f'./check_points/{args.dataset}'
if not os.path.exists(cp_path):
    os.makedirs(cp_path)

best_name = f'EMOD_{args.dataset}_load{args.load_pretrain_model}_lr{args.lr}_wd{args.wd}.ckpt'
save_ckpt_callback = SaveCheckpointsCallback(cp_path, save_step=1, save_best_only=True,
                                             file_name=best_name)

if args.dataset == 'Faced':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-6)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-8)

record_dict = train(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, scheduler,
                    device=device, save_ckpt_callback=save_ckpt_callback, max_norm=1.0,)


# Evaluate the best model
# load_best_model
print(f'-------best val model---------')
model.load_state_dict(torch.load(os.path.join(cp_path, best_name), map_location=device))
model.to(device)
loss, acc, c,balanced_acc,kappa, weighted_f1 = evaluate(model, test_loader, loss_fn, device=device)
print(args.dataset, loss, acc)
print(c)
print(f'Balanced Accuracy: {balanced_acc:.4f}')
print(f'Kappa: {kappa:.4f}')
print(f'Weighted F1 Score: {weighted_f1:.4f}')
