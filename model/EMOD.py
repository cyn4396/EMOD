from model.AxialTransformer import AxialTransformer
import torch.nn.functional as F
import torch
import torch.nn as nn
import random
from einops.layers.torch import Rearrange




class TokenEmbedding(nn.Module):
    def __init__(self, d_model, eeg_duration=10):
        super().__init__()
        self.time_tokenConv = nn.Sequential(nn.Conv2d(1, d_model, kernel_size=(1, 25), padding='same'),
                                            nn.GroupNorm(4, d_model),
                                            nn.GELU())
        self.freq_tokenConv = nn.Sequential(nn.Conv2d(1, d_model, kernel_size=(1, 25), padding='same'),
                                            nn.GroupNorm(4, d_model),
                                            nn.GELU())

        self.eeg_duration = eeg_duration
        self.l = nn.Sequential(
            Rearrange(' b d C T -> b C T d'),
            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )


    def forward(self, x, eeg_duration=None):
        if eeg_duration is None:
            eeg_duration = self.eeg_duration

        batch_size, channels, eeg_len = x.shape
        x_freq = torch.fft.rfft(x, dim=-1, norm='forward').real
        x = self.time_tokenConv(x.unsqueeze(1))
        gap_time = nn.AdaptiveAvgPool2d((channels, eeg_duration * 5))
        x = gap_time(x)
        x_freq = self.freq_tokenConv(x_freq.unsqueeze(1))
        gap_freq = nn.AdaptiveAvgPool2d((channels, eeg_duration * 5))
        x_freq = gap_freq(x_freq)
        x = torch.cat([x, x_freq], dim=1)
        x = self.l(x)
        return x




class EMOD_pretrain(nn.Module):
    def __init__(self,
                 n_filters: int = 32, depth=2, dropout=0.25):
        super().__init__()
        self.d_model = n_filters
        self.embd = TokenEmbedding(n_filters, eeg_duration=10)
        # self.durations = [1,5,10]
        self.pos_embeddings = nn.Embedding(72, n_filters)
        self.axial_transformer = AxialTransformer(embed_dim=n_filters, num_heads=n_filters // 8,
                                                  dropout=dropout, num_layers=depth)
        self.projection_head = nn.Sequential(
            ChannelAttentionAlongC(n_filters),
            TemporalAttentionAlongT(n_filters),
            nn.Flatten(start_dim=1),
            nn.Linear(n_filters, n_filters * 4, bias=False),
            nn.BatchNorm1d(n_filters * 4),
            nn.GELU(),
            nn.Linear(n_filters * 4, n_filters, bias=True)
        )

    def forward(self, x: torch.Tensor, channel_id=None) -> torch.Tensor:
        batch_size, channels, eeg_len = x.shape
        x = self.embd(x)
        if channel_id is not None:
            x = x + self.pos_embeddings(channel_id).unsqueeze(2)
        x = self.axial_transformer(x)
        x = self.projection_head(x)
        x = F.normalize(x, p=2, dim=-1)
        return x



class TemporalAttentionAlongT(nn.Module):
    def __init__(self, d_model,dropout=0.25):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T, D]
        B, C, T, D = x.shape
        x_ = x.view(B * C, T, D)               # → [B*C, T, D]
        w = self.attn(x_)                      # [B*C, T, 1]
        w = torch.softmax(w, dim=1)            # 时间方向 softmax
        w = self.dropout(w)                    # 应用 dropout
        x_attn = (x_ * w).sum(dim=1)           # [B*C, D]
        x_attn = x_attn.view(B, C, 1, D)          # [B, C, D]
        return x_attn

class ChannelAttentionAlongC(nn.Module):
    def __init__(self, d_model, dropout=0.25):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        # x: [B, C, T, D]
        B, C, T, D = x.shape
        x_ = x.permute(0, 2, 1, 3).contiguous()   # [B, T, C, D]
        x_flat = x_.view(B * T, C, D)             # [B*T, C, D]
        w = self.attn(x_flat)                     # [B*T, C, 1]
        w = torch.softmax(w, dim=1)               # attention weights along C
        w = self.dropout(w)                       # 应用 dropout
        x_attn = (x_flat * w).sum(dim=1)          # [B*T, D]
        x_attn = x_attn.view(B, 1, T, D)             # [B, T, D]
        return x_attn


class MeanOverChannels(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim=self.dim)

class EMOD(nn.Module):
    def __init__(self,
                 n_filters: int = 32, depth=2, dropout=0.25, frequency=250,
                 num_classes=9, eeg_len=250, channels=32, dataset='Faced'):
        super().__init__()
        self.eeg_duration = eeg_len // frequency
        self.embd = TokenEmbedding(n_filters, eeg_duration=eeg_len // frequency)
        self.pos_embeddings = nn.Embedding(72, n_filters)
        self.axial_transformer = AxialTransformer(embed_dim=n_filters, num_heads=n_filters // 8,
                                                  dropout=dropout, num_layers=depth)
        if dataset == 'Faced':
            cls_dim = n_filters * self.eeg_duration * 5
            self.classifier = nn.Sequential(
                ChannelAttentionAlongC(n_filters),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.25),
                nn.Linear(cls_dim, n_filters * 4),
                nn.ELU(), nn.Dropout(0.25),
                nn.Linear(n_filters * 4, n_filters),
                nn.ELU(), nn.Dropout(0.25),
                nn.Linear(n_filters, num_classes))
        elif dataset == 'SEED':
            cls_dim = 5*5*channels
            self.classifier = nn.Sequential(
                MeanOverChannels(3),
                nn.Flatten(start_dim=1),
                nn.Linear(cls_dim, n_filters//2),
                nn.ELU(), nn.Dropout(0.5),
                nn.Linear(n_filters//2, num_classes))
        elif dataset == 'SEEDV':
            cls_dim = channels * n_filters * self.eeg_duration * 5
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(cls_dim, cls_dim//16),
                nn.ELU(), nn.Dropout(0.5),
                nn.Linear(cls_dim//16, n_filters),
                nn.ELU(), nn.Dropout(0.5),
                nn.Linear(n_filters, num_classes))
        else:
            cls_dim = n_filters
            self.classifier = nn.Sequential(
                ChannelAttentionAlongC(n_filters),
                TemporalAttentionAlongT(n_filters),
                nn.Flatten(start_dim=1),
                nn.Dropout(0.25),
                nn.Linear(cls_dim, cls_dim//2),
                nn.ELU(), nn.Dropout(0.25),
                nn.Linear(cls_dim//2, num_classes))

    def forward(self, x: torch.Tensor, channel_id=None) -> torch.Tensor:
        x = self.embd(x)
        if channel_id is not None:
            x = x + self.pos_embeddings(channel_id).unsqueeze(2)
        x = self.axial_transformer(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    n_channels = 32
    frequency = 250
    eeg_duration = 10
    num_classes = 9
    n_filters = 128
    model = EMOD(n_filters=n_filters, depth=3, dropout=0.25, frequency=frequency, num_classes=num_classes,
                 eeg_len=frequency * eeg_duration, channels=n_channels, )
    x = torch.randn(2, n_channels, frequency * eeg_duration)
    y = model(x)

