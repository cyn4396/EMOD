import torch
import torch.nn as nn


class RelPositionalEncoding1D(nn.Module):
    def __init__(self, head_dim, max_len=50):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.rel_embed = nn.Parameter(torch.randn(2 * max_len - 1, head_dim))  # 可学习
        # nn.init.normal_(self.rel_embed , mean=0.0, std=0.02)
    def forward(self, qlen, klen):
        # 相对距离范围：[-(klen-1), qlen-1]
        range_q = torch.arange(qlen, device=self.rel_embed.device)
        range_k = torch.arange(klen, device=self.rel_embed.device)
        distance = range_q[:, None] - range_k[None, :]  # shape: [qlen, klen]
        distance = distance + self.max_len - 1          # shift to >= 0
        rel_pos = self.rel_embed[distance]              # shape: [qlen, klen, head_dim]
        return rel_pos


class AxialAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, axial_dim, using_rel_pos=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.axial_dim = axial_dim
        self.head_dim = embed_dim // num_heads
        self.use_rel_pos = using_rel_pos

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        if self.axial_dim == "width":
            self.rel_pos_embed = RelPositionalEncoding1D(self.head_dim, 50)


    def forward(self, x):  # x: [B, H, W, C]
        B, H, W, C = x.shape
        if self.axial_dim == "height":
            x = x.permute(0, 2, 1, 3).contiguous()  # [B, W, H, C]
            N, L = B * W, H
        else:
            x = x.permute(0, 1, 3, 2).contiguous()  # [B, H, C, W]
            N, L = B * H, W

        x = x.view(N, L, C)  # [N, L, C]

        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [N, heads, L, head_dim]

        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if self.axial_dim == "width" and self.use_rel_pos:
            rel_pos = self.rel_pos_embed(L, L)  # [L, L, D]
            rel_logits = torch.einsum('bnlh,lmh->bnlm', q, rel_pos)  # [B*N, num_heads, L, L]
            attn = attn + rel_logits

        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)  # [N, heads, L, D]
        out = out.transpose(1, 2).reshape(N, L, C)

        out = self.proj(out)

        # reshape back
        if self.axial_dim == "height":
            out = out.view(B, W, H, C).permute(0, 2, 1, 3)
        else:
            out = out.view(B, H, W, C)

        return out


class AxialTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, using_rel_pos=True):
        super().__init__()
        self.width_attn = AxialAttention(embed_dim, num_heads, "width", using_rel_pos=using_rel_pos)
        self.height_attn = AxialAttention(embed_dim, num_heads, "height", using_rel_pos=False)
        self.norm_width = nn.LayerNorm(embed_dim)
        self.norm_height = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, H, W, C]
        x = x + self.dropout(self.height_attn(self.norm_height(x)))
        x = x + self.dropout(self.width_attn(self.norm_width(x)))
        x = x + self.ffn(self.norm_ffn(x))
        return x


class AxialTransformer(nn.Module):
    """
    Axial Transformer with Relative Positional Encoding

    This implementation is inspired by the following work:

    - Paper: Ho, J., Kalchbrenner, N., Weissenborn, D., & Salimans, T. (2020).
             "Axial Attention in Multidimensional Transformers".
             arXiv preprint arXiv:1912.12180.
    - URL: https://arxiv.org/abs/1912.12180
    - Related Project: https://github.com/lucidrains/axial-attention
    """
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()
        ff_dim = embed_dim*4
        self.layers = nn.ModuleList([
            AxialTransformerBlock(embed_dim, num_heads, ff_dim, dropout, using_rel_pos=True)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 使用示例
if __name__ == "__main__":
    # 输入参数
    batch_size = 2
    height = 32
    width = 50
    embed_dim = 128

    # 创建模型
    model = AxialTransformer(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=6,
        dropout=0.25
    )

    # 测试输入
    x = torch.randn(batch_size, height, width, embed_dim)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")