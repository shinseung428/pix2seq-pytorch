import copy
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

from models.modules import MultiHeadAttention
from models.modules import ConvFeedForward
from models.modules import FeedForward
from models.modules import Normalize


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Encoder(nn.Module):
    def __init__(
            self, hidden_dim, ff_dim,
            num_layers, num_heads, dropout,
            normalize_before, activation):
        super(Encoder, self).__init__()

        # Encoder Layers
        layer = EncoderLayer(
            hidden_dim, ff_dim,
            n_head=num_heads,
            dropout=dropout,
            normalize_before=normalize_before,
            activation=activation
        )

        self.layers = self.clone_layers(layer, num_layers)

        self.norm = nn.LayerNorm(hidden_dim)
        self.normalize_before = normalize_before

    def clone_layers(self, layer, num):
        return nn.ModuleList([copy.deepcopy(layer) for _ in range(num)])

    def forward(
            self, src,
            src_mask=None,
            pos=None
    ):
        x = src

        if self.normalize_before:
            x = self.norm(x)

        for layer_num, layer in enumerate(self.layers):
            x = layer(x, pos=pos)

        if not self.normalize_before:
            x = self.norm(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(
            self, d_model, dim_feedforward, n_head,
            dropout=0.1, normalize_before=False,
            activation="relu"):
        super(EncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
