import numpy as np
import torch


def forward_concat_batch(model, datas, channels, device, use_projection=False):
    """
    Performs forward pass for a list of (data, channel) pairs,
    concatenates outputs and optionally applies projection head.
    """
    logits_list = []
    for data, ch in zip(datas, channels):
        data = data.to(device)
        ch = ch.to(device)
        logits = model(data, ch)
        logits_list.append(logits.unsqueeze(1))

    logits_concat = torch.cat(logits_list, dim=0)  # [B, 1, D]
    if use_projection:
        logits_concat = model(logits_concat, use_projection=True)
    return logits_concat


def evaluate(model, dataloader, device, criterion):
    model.eval()
    loss_list = []

    with torch.no_grad():
        for datas, arousal, valence, channels in dataloader:
            arousal = torch.cat(arousal, dim=0).to(device)
            valence = torch.cat(valence, dim=0).to(device)

            logits_concat = forward_concat_batch(model, datas, channels, device, use_projection=True)
            loss = criterion(logits_concat, arousal, valence)
            loss_list.append(loss.item())

    return np.mean(loss_list)


def contrastive_train(model, criterion, train_loader, num_epochs, optimizer, scheduler,
                      device, save_ckpt_callback=None, max_norm=3.0):
    record_dict = {"train": [], "val": []}
    global_step = 0
    model.to(device)

    for epoch_id in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch_id, (datas, arousal, valence, channels) in enumerate(train_loader):
            optimizer.zero_grad()

            arousal = torch.cat(arousal, dim=0).to(device)
            valence = torch.cat(valence, dim=0).to(device)

            logits_concat = forward_concat_batch(model, datas, channels, device)
            loss = criterion(logits_concat, arousal, valence)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        record_dict["train"].append({
            "loss": avg_loss,
            "step": epoch_id
        })

        # No validation set provided, so use last loss as "val"
        record_dict["val"].append({
            "loss": avg_loss,
            "step": epoch_id
        })

        if save_ckpt_callback is not None:
            save_ckpt_callback(epoch_id, model.state_dict(), metric=1 / avg_loss)

        print(f"[Epoch {epoch_id}] Loss: {avg_loss:.4f}")

    return record_dict