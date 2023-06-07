# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor
from cuda.shift import Shift

# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output, attention


# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x



def  bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)


class Shift_tcn_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(Shift_tcn_layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        bn_init(self.bn2, 1)
        # self.bn = nn.LayerNorm(in_channels, eps=1e-6)
        # self.bn2 = nn.LayerNorm(in_channels, eps=1e-6)
        self.relu = nn.ReLU(inplace=True)
        self.shift_in = Shift(channel=in_channels, stride=1, init_scale=1)
        self.shift_out = Shift(channel=out_channels, stride=1, init_scale=1)

        self.temporal_linear = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.kaiming_normal(self.temporal_linear.weight, mode='fan_out')

    def forward(self, x):
        # y = x[:, :, int(512/2):]   # partial_shift
        # x = x[:, :, :int(512/2)]
        x = x.unsqueeze(-1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = self.bn(x)
        # shift1
        x = self.shift_in(x)
        x = self.temporal_linear(x)
        x = self.relu(x)
        # shift2
        x = self.shift_out(x)
        x = self.bn2(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.squeeze(-1)
        # x =  torch.cat([x,y], dim=-1)
        return x

class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds the position-encodings to the
    input for as many time steps as necessary.

    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, size: int = 0, max_len: int = 5000):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, : emb.size(1)] # emb: [batch_size, seq_len, dim] 前面标注的有问题


class PositionalEncoding_Hirerachical(nn.Module):


    def __init__(self, size: int = 0, max_len: int = 5000):

        if size % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(size)
            )
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10000.0) / size))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding_Hirerachical, self).__init__()
        self.register_buffer("pe", pe)
        self.dim = size

        self.conv1d = nn.Conv1d(42, 42, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb):

        # Divided input into 4 parts
        b, t, c = emb.size()
        div = t // 8

        pe = self.pe[:,: emb.size(1)]
        pe_1 = pe[:,:int(div*8), :int(c-42)]
        pe_2 = pe[:,int(div*8):, :]
        pe_3 = pe[:,:int(div*8), int(c-42):] 
        pe_3a = pe_3[:,:int(div*8//2),:32]
        pe_3b = pe_3[:,:int(div*8//4),32:40]
        pe_3c = pe_3[:,:int(div*8//8),40:]

        hpe_a = torch.zeros((1, int(div*8) , 32))
        hpe_b = torch.zeros((1, int(div*8) , 8))
        hpe_b1 =torch.zeros((1, int(div*8//2) , 8))
        hpe_c = torch.zeros((1, int(div*8) , 2))
        hpe_c1 =torch.zeros((1, int(div*8//2) , 2))
        hpe_c2 =torch.zeros((1, int(div*8//4) , 2))

        hpe_a[:,0::2,:] = pe_3a
        hpe_a[:,1::2,:] = pe_3a

        hpe_b1[:,0::2,:] = pe_3b
        hpe_b1[:,1::2,:] = pe_3b
        hpe_b[:,0::2,:] = hpe_b1
        hpe_b[:,1::2,:] = hpe_b1

        hpe_c2[:,0::2,:] = pe_3c
        hpe_c2[:,1::2,:] = pe_3c
        hpe_c1[:,0::2,:] = hpe_c2
        hpe_c1[:,1::2,:] = hpe_c2
        hpe_c[:,0::2,:] = hpe_c1
        hpe_c[:,1::2,:] = hpe_c1

        hpe = torch.cat([hpe_a, hpe_b, hpe_c], dim=-1)

        hpe = hpe.cuda()
        # hpe = hpe + pe_3
        hpe = hpe.permute(0, 2, 1).contiguous()
        hpe = self.relu(self.conv1d(hpe))
        hpe = hpe.permute(0, 2, 1).contiguous()


        hpe = hpe.cuda()
        pe_all = torch.cat([pe_1, hpe], dim = -1)
        pe_all = torch.cat([pe_all, pe_2], dim=1)

        out = emb + pe_all.cuda()


        return out 


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()


        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size
    

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """

        x_norm = self.layer_norm(x)

        cls_mask = mask[:,:,0]
        cls_mask = cls_mask.unsqueeze(1)
        mask = torch.cat([cls_mask, mask], dim = -1)

        h, _a = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o



class TransformerEncoderLayer_Local(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer_Local, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)  # 入维度：size=512
        self.src_src_att = MultiHeadedAttention(num_heads, int(size/2), dropout=dropout)  # num_heads=8,size=512
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

        self.shift_pe = Shift_tcn_layer(in_channels=int(size/2), out_channels=int(size/2), kernel_size=1, stride=1)
    

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        cls_mask = mask[:,:,0]
        cls_mask = cls_mask.unsqueeze(1)
        mask = torch.cat([cls_mask, mask], dim = -1)

        x_tcn = x_norm[:,:, int(512/2):]    
        x_norm = x_norm[:,:, :int(512/2)]

        h, _a = self.src_src_att(x_norm, x_norm, x_norm, mask)
        s = self.shift_pe(x_tcn)
        h_s = torch.cat([h,s], dim=-1) # origion
        h = self.dropout(h_s) + x
        o = self.feed_forward(h)

        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.

    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(
            self, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1
    ):
        """
        Represents a single Transformer decoder layer.

        It attends to the source representation and the previous decoder states.

        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.src_trg_att = MultiHeadedAttention(num_heads, size, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)
     

    # pylint: disable=arguments-differ
    def forward(
            self,
            x: Tensor = None,
            memory: Tensor = None,
            src_mask: Tensor = None,
            trg_mask: Tensor = None,
    ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.

        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :return: output tensor
        """
        # decoder/target self-attention

        x_norm = self.x_layer_norm(x)
        h1, _a1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2, _a2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask)

        # final position-wise feed-forward layer
        o = self.feed_forward(self.dropout(h2) + h1)

        return o


class BERTIdentity(nn.Module):
    """Identity layer for use in BertModel, which takes 8 inputs in forward instead of 2."""

    def __init__(self):
        super(BERTIdentity, self).__init__()

    def forward(self,
                hidden_states,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=False, ):
        return [hidden_states]


class mBARTIdentity(nn.Module):
    """Identity layer for use in mBartModel, which takes 8 inputs in forward instead of 2."""

    def __init__(self):
        super(mBARTIdentity, self).__init__()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                layer_head_mask: torch.Tensor,
                output_attentions: bool = False):
        return [hidden_states]


class mBARTIdentityDecoder(nn.Module):
    """Identity layer for use in mBartModel, which takes 8 inputs in forward instead of 2."""

    def __init__(self):
        super(mBARTIdentityDecoder, self).__init__()

    def forward(self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                layer_head_mask=None,
                cross_attn_layer_head_mask=None,
                past_key_value=None,
                output_attentions=None,
                use_cache=None):
        return [hidden_states]
