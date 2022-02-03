import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, num_class, d_model, pad_id):
        super().__init__()
        self.embed = nn.Embedding(
                num_class, d_model,
                padding_idx=pad_id)

    def forward(self, x):
        return self.embed(x)


class Normalize(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()

        self.size = dim
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True))
        norm = norm / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias

        return norm


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConvFeedForward(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(ConvFeedForward, self).__init__()
        self.conv1 = nn.Conv2d(
                hidden_dim, hidden_dim * 4,
                kernel_size=3, padding=1, bias=True,
                )
        # self.relu = nn.Relu(in_place=True)
        self.conv2 = nn.Conv2d(
                hidden_dim * 4, hidden_dim,
                kernel_size=1, bias=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.dropout2(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, input_dim, dropout=0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(input_dim, input_dim*4, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        # self.relu = nn.Relu(inplace=True)
        self.linear2 = nn.Linear(input_dim*4, input_dim, bias=True)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.dropout2(x)

        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(
            self, in_channels, k_channels, v_channels,
            n_head=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.in_channels = in_channels
        self.k_channels = k_channels
        self.v_channels = v_channels
        self.n_head = n_head

        self.q_linear = nn.Linear(in_channels, n_head * k_channels)
        self.k_linear = nn.Linear(in_channels, n_head * k_channels)
        self.v_linear = nn.Linear(in_channels, n_head * v_channels)
        self.attention = ScaledDotProductAttention(
                temperature=k_channels ** 0.5, dropout=dropout)
        self.out_linear = nn.Linear(n_head * v_channels, in_channels)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.q_linear(q).view(
                b, q_len, self.n_head, self.k_channels
                ).transpose(1, 2)
        k = self.k_linear(k).view(
                b, k_len, self.n_head, self.k_channels
                ).transpose(1, 2)
        v = self.v_linear(v).view(
                b, v_len, self.n_head, self.v_channels
                ).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().view(
                b, q_len, self.n_head * self.v_channels)
        out = self.out_linear(out)
        out = self.dropout(out)

        return out, attn
