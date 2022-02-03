import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

from models.pos_encoder import PositionalEncoder
from models.modules import MultiHeadAttention
from models.modules import FeedForward
from models.modules import Normalize
from models.modules import Embedder
from models.modules import MLP


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Decoder(nn.Module):
    def __init__(
            self, num_class, pad_id, max_w, max_h,
            hidden_dim, ff_dim, num_layers,
            num_heads, dec_max_len, dropout,
            normalize_before, activation):
        super(Decoder, self).__init__()

        self.pad_id = pad_id # na token
        self.num_heads = num_heads

        self.embed = Embedder(
            num_class, hidden_dim, self.pad_id
        )

        self.tgt_pos_enc = PositionalEncoder(hidden_dim, max_len=dec_max_len+1)

        layer = DecoderLayer(
            hidden_dim, ff_dim,
            n_head=num_heads,
            dropout=dropout,
            normalize_before=normalize_before,
            activation=activation
        )
        self.layers = self.clone_layers(layer, num_layers)

        self.generator = MLP(hidden_dim, hidden_dim, num_class, 3)

        self.norm = nn.LayerNorm(hidden_dim)
        self.normalize_before = normalize_before


    def clone_layers(self, layer, num):
        return nn.ModuleList([copy.deepcopy(layer) for _ in range(num)])

    def _pad_mask(self, label):
        pad_mask = (label == self.pad_id)
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)
        return pad_mask.cuda()

    def _order_mask(self, label):
        t = label.size(1)
        order_mask = torch.triu(torch.ones(t, t), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0)
        return order_mask.cuda()

    def _create_mask(self, labels):
        tgt_mask = (self._pad_mask(labels) | self._order_mask(labels))
        tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)
        return tgt_mask

    def forward(
        self, src, labels,
        src_pos=None,
        input_mask=None,
        src_mask=None
    ):
        tgt = self.embed(labels)
        tgt_mask = self._create_mask(labels)

        # (b, n, c) -> (n, b, c)
        tgt = tgt.permute(1, 0, 2)

        tgt_pos = self.tgt_pos_enc(tgt)

        if self.normalize_before:
            tgt = self.norm(tgt)

        for layer_num, layer in enumerate(self.layers):
            tgt = layer(
                tgt, src,
                query_pos=tgt_pos,
                pos=src_pos,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=input_mask
            )

        if not self.normalize_before:
            tgt = self.norm(tgt)

        tgt = self.generator(tgt)

        return tgt


class DecoderLayer(nn.Module):
    def __init__(
            self, d_model, dim_feedforward, n_head,
            dropout=0.1, activation="relu",
            normalize_before=False):
        super(DecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

