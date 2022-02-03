import math

import torch
import torch.nn as nn


def generate_encoder(in_channels, max_len):
    pos = torch.arange(max_len).float().unsqueeze(1)

    i = torch.arange(in_channels).float().unsqueeze(0)
    angle_rates = 1 / torch.pow(10000, (2 * (i//2)) / in_channels)

    position_encoder = pos * angle_rates
    position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
    position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

    return position_encoder


class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=2000, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        position_encoder = generate_encoder(in_channels, max_len)
        position_encoder = position_encoder.unsqueeze(0)
        self.register_buffer('position_encoder', position_encoder)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = x + self.position_encoder[:, :x.size(1), :]
        out = self.dropout(out)

        return out



class Adaptive2DPositionEncoder(nn.Module):
    def __init__(self, in_channels, max_h=200, max_w=200, dropout=0.1):
        super(Adaptive2DPositionEncoder, self).__init__()

        h_position_encoder = generate_encoder(in_channels, max_h)
        h_position_encoder = h_position_encoder.transpose(0, 1).view(1, in_channels, max_h, 1)

        w_position_encoder = generate_encoder(in_channels, max_w)
        w_position_encoder = w_position_encoder.transpose(0, 1).view(1, in_channels, 1, max_w)

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(in_channels)
        self.w_scale = self.scale_factor_generate(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)

    def scale_factor_generate(self, in_channels):
        scale_factor = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
                )

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.size()

        avg_pool = self.pool(x)

        h_pos_encoding = self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, start_idx=0):
        # x = x + self.pe[:x.size(0), :]
        x = self.pe[start_idx:start_idx + x.size(0), :]
        return self.dropout(x)


class PositionEmbeddingSine(nn.Module):
    def __init__(
            self, num_pos_feats=64, temperature=10000,
            normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((
            pos_x[:, :, :, 0::2].sin(),
            pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((
            pos_y[:, :, :, 0::2].sin(),
            pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats//2)
        self.col_embed = nn.Embedding(50, num_pos_feats//2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        x = tensor
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        return pos
