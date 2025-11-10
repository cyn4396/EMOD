import torch
import numpy as np
import random
import scipy.signal as signal

# 加高斯噪声
def add_gaussian_noise(data, channels, noise_factor=0.1):
    data = reshape_data(data)
    mean = data.mean()
    std = data.std()
    noise = torch.randn_like(data) * std * noise_factor + mean * noise_factor
    return data + noise, channels

# 通道级mask
def channel_mask(data, channels, mask_prob=0.2):
    num_channels = data.size(0)  # 通道数 n_chs
    data = reshape_data(data)
    mask = torch.rand(num_channels) < mask_prob
    data[mask] = 0  # Mask掉通道
    channels[mask] = 0  # Mask掉通道
    return data, channels

# 打乱通道顺序
def shuffle_channels(data, channels, prob=0.2):
    """打乱一定比率的通道顺序"""
    # 获取通道数
    num_channels = data.size(0)
    data = reshape_data(data)
    # 计算要打乱的通道数量
    num_to_shuffle = int(num_channels * prob)

    # 随机选择要打乱的通道索引
    perm_indices = torch.randperm(num_channels)[:num_to_shuffle]

    # 生成打乱的通道索引
    shuffled_indices = torch.randperm(num_to_shuffle)

    # 将选中的通道按照打乱的顺序重新排列
    perm = torch.arange(num_channels)
    perm[perm_indices] = perm[perm_indices][shuffled_indices]

    # 打乱通道
    shuffled_data = data[perm]
    # 更新通道标签
    shuffled_channels = channels[perm]


    return shuffled_data, shuffled_channels

# 时间mask
def time_mask(data, channels, mask_prob=0.2):
    data = reshape_data(data)
    num_timesteps = data.size(1)
    mask_len = int(mask_prob * num_timesteps)
    start = random.randint(0, num_timesteps - mask_len)
    data[:, start:start + mask_len] = 0  # Mask掉时间段
    return data, channels


def time_flip(data, channels):
    flipped_data = data.flip(dims=[1])
    flipped_data = reshape_data(flipped_data)
     # 沿时间轴翻转
    return flipped_data, channels

# EEG 滤波 (低通滤波)
def eeg_filter(data, channels, lowcut=1.0, highcut=40.0):
    if data.ndim != 3:
        raise ValueError("data 应为三维数组 (channels, seconds, sample_rate)")

    n_channels, seconds, sample_rate = data.shape
    fs = sample_rate  # 采样率

    # reshape 为 (channels, seconds * sample_rate)
    data = data.reshape(n_channels, -1)

    # 归一化截止频率
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError(
            f"错误的归一化频率：low={low:.3f}, high={high:.3f}。"
            f"请检查 lowcut/highcut 是否过大，或 fs 是否设置错误（当前 fs={fs}）"
        )

    # 构建滤波器
    b, a = signal.butter(N=4, Wn=[low, high], btype='band')

    # 滤波每个通道
    filtered = np.zeros_like(data)
    for i in range(n_channels):
        filtered[i] = signal.filtfilt(b, a, data[i])

    return torch.tensor(filtered, dtype=torch.float32), channels

# 不做任何增强
def no_aug(data, channels):
    data = reshape_data(data)  # 转换为 [n_chs, seqlen * frequency]
    return data, channels

# 数据维度调整：转换为 [n_chs, seqlen * frequency]
def reshape_data(data):
    # 假设输入是 [n_chs, seqlen, frequency]
    n_chs, seqlen, frequency = data.size()
    return data.view(n_chs, -1)  # 转换为 [n_chs, seqlen * frequency]