import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, balanced_accuracy_score, cohen_kappa_score, f1_score
import os
import random
from tqdm import tqdm

def setup_seed(seed=7):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

class SaveCheckpointsCallback:
    def __init__(self, save_dir, save_step=5000, save_best_only=True, file_name='best'):
        """
        Save checkpoints each save_epoch epoch.
        We save checkpoint by epoch in this implementation.
        Usually, training scripts with pytorch evaluating model and save checkpoint by step.

        Args:
            save_dir (str): dir to save checkpoint
            save_epoch (int, optional): the frequency to save checkpoint. Defaults to 1.
            save_best_only (bool, optional): If True, only save the best model or save each model at every epoch.
        """
        self.save_dir = save_dir
        self.save_step = save_step
        self.save_best_only = save_best_only
        self.best_metrics = -1
        self.file_name = file_name

        # mkdir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def __call__(self, step, state_dict, metric=None):
        if step % self.save_step > 0:
            return

        if self.save_best_only:
            assert metric is not None
            if metric >= self.best_metrics:
                # save checkpoints
                torch.save(state_dict, os.path.join(self.save_dir, self.file_name))
                # update best metrics
                self.best_metrics = metric
        else:
            torch.save(state_dict, os.path.join(self.save_dir, f"step{step}_{self.file_name}"))


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model on validation/test set."""
    model.eval()
    loss_list, pred_list, label_list = [], [], []

    with torch.no_grad():
        for datas, labels, channel_idx in dataloader:
            datas, labels, channel_idx = datas.to(device), labels.to(device), channel_idx.to(device)

            logits = model(datas, channel_idx)
            loss = loss_fn(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            loss_list.append(loss.item())
            pred_list.extend(preds.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

    return (
        np.mean(loss_list),
        accuracy_score(label_list, pred_list),
        confusion_matrix(label_list, pred_list),
        balanced_accuracy_score(label_list, pred_list),
        cohen_kappa_score(label_list, pred_list),
        f1_score(label_list, pred_list, average='weighted'),
    )


def predict(model, dataloader, device):
    model.eval()
    pred_list, label_list = [], []

    with torch.no_grad():
        for datas, labels, channels in dataloader:
            datas, labels, channels = datas.to(device), labels.to(device), channels.to(device)

            logits = model(datas, channels)
            preds = torch.argmax(logits, dim=-1)

            pred_list.extend(preds.cpu().numpy())
            label_list.extend(labels.cpu().numpy())

    return label_list, pred_list

def train(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, scheduler, device,
          save_ckpt_callback=None, max_norm=1.0):

    record_dict = {"train": [], "val": []}
    global_step = 0
    model.to(device)

    for epoch_id in range(num_epochs):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        num_batches = len(train_loader)

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch_id}", unit="batch")

        for batch_id, (datas, labels, channels) in enumerate(train_pbar):
            datas, labels, channels = datas.to(device), labels.to(device), channels.to(device)

            optimizer.zero_grad()
            logits = model(datas, channels)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            preds = torch.argmax(logits, dim=-1)
            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())

            total_loss += loss.item()
            total_acc += acc
            global_step += 1

            # 更新 tqdm 进度条的描述信息
            train_pbar.set_postfix(loss=loss.item(), acc=acc)

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        record_dict["train"].append({"loss": avg_loss, "acc": avg_acc, "step": epoch_id})

        # Validate
        val_loss, val_acc, c, ba, kappa, wf1 = evaluate(model, val_loader, loss_fn, device)
        record_dict["val"].append({
            "loss": val_loss, "acc": val_acc, "step": epoch_id,
            "ba": ba, "kappa": kappa, "wf1": wf1
        })

        if save_ckpt_callback:
            save_ckpt_callback(global_step, model.state_dict(), metric=kappa)

        print(f"[Epoch {epoch_id}] Train Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, BACC: {ba:.4f}, Kappa: {kappa:.4f}, WF1: {wf1:.4f}")

    return record_dict
