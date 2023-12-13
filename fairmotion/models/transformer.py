# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from fairmotion.models import decoders


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerLSTMModel(nn.Module):
    def __init__(
        self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5
    ):
        super(TransformerLSTMModel, self).__init__()
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(
            ninp, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers
        )
        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = decoders.LSTMDecoder(
            input_dim=ntoken, hidden_dim=hidden_dim, output_dim=ntoken,
        )
        self.num_layers = num_layers

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for name, param in self.decoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask=None)
        final_encoder_state = output[:, -1].unsqueeze(0).contiguous()
        output = self.decoder(
            tgt,
            hidden=final_encoder_state,
            cell=final_encoder_state,
            max_len=max_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return output


class TransformerModel(nn.Module):
    def __init__(
        self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5
    ):
        super(TransformerModel, self).__init__()
        self.model_type = "Transformer"
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layer = TransformerEncoderLayer(
            ninp, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(ninp),
        )
        decoder_layer = TransformerDecoderLayer(
            ninp, num_heads, hidden_dim, dropout
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(ninp),
        )

        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(ntoken, ninp)
        self.project = nn.Linear(ninp, ntoken)
        self.ninp = ninp

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # src and tgt are now (T, B, E)
        if max_len is None:
            max_len = tgt.shape[0]

        projected_src = self.encoder(src) * np.sqrt(self.ninp)
        pos_encoded_src = self.pos_encoder(projected_src)
        encoder_output = self.transformer_encoder(pos_encoded_src)

        if self.training:

            # Create mask for training
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
                device=tgt.device,
            )

            # Use last source pose as first input to decoder
            tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
            pos_encoder_tgt = self.pos_encoder(
                self.encoder(tgt) * np.sqrt(self.ninp)
            )
            output = self.transformer_decoder(
                pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
            )
            output = self.project(output)
        else:
            # greedy decoding
            decoder_input = torch.zeros(
                max_len, src.shape[1], src.shape[-1],
            ).type_as(src.data)
            next_pose = tgt[0].clone()

            # Create mask for greedy encoding across the decoded output
            tgt_mask = self._generate_square_subsequent_mask(max_len).to(
                device=tgt.device
            )
            
            for i in range(max_len):
                decoder_input[i] = next_pose
                pos_encoded_input = self.pos_encoder(
                    self.encoder(decoder_input) * np.sqrt(self.ninp)
                )
                decoder_outputs = self.transformer_decoder(
                    pos_encoded_input, encoder_output, tgt_mask=tgt_mask,
                )
                output = self.project(decoder_outputs)
                next_pose = output[i].clone()
                del output
            output = decoder_input
        return output.transpose(0, 1)


class TransformerModelV1(TransformerModel):
    def __init__(self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5, k=3):
        super(TransformerModelV1, self).__init__(ntoken, ninp, num_heads, hidden_dim, num_layers, dropout)
        self.k = k
        self.temporal_convolution = nn.Conv1d(in_channels=72, out_channels=72, kernel_size=k, padding=0)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        # we get (n, len, dim) (64, 120, 72) as src

        # pad calculated amount of zeros to beginning of src. padding depends on kernel size.
        # padding should change dimensions to (64, 120 + 2P, 72)
        p = (self.k - 1) // 2
        src = F.pad(input=src, pad=(0, 0, 2 * p, 0), mode='constant', value=0)

        # (n, len, dim) => (n, dim, len)
        src = src.permute((0, 2, 1))

        # pass this padded src to cnn with padding 0. output should still be n x dim x len
        src = self.temporal_convolution(src)  # (n, dim, len) (64, 72, 120)

        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.permute((2, 0, 1)) # (len, n, dim) (120, 64, 72)

        #src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # src and tgt are now (T, B, E)
        if max_len is None:
            max_len = tgt.shape[0]

        projected_src = self.encoder(src) * np.sqrt(self.ninp)
        pos_encoded_src = self.pos_encoder(projected_src)
        encoder_output = self.transformer_encoder(pos_encoded_src)

        if self.training:

            # Create mask for training
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
                device=tgt.device,
            )

            # Use last source pose as first input to decoder
            tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
            pos_encoder_tgt = self.pos_encoder(
                self.encoder(tgt) * np.sqrt(self.ninp)
            )
            output = self.transformer_decoder(
                pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
            )
            output = self.project(output)
        else:
            # greedy decoding
            decoder_input = torch.zeros(
                max_len, src.shape[1], src.shape[-1],
            ).type_as(src.data)
            next_pose = tgt[0].clone()

            # Create mask for greedy encoding across the decoded output
            tgt_mask = self._generate_square_subsequent_mask(max_len).to(
                device=tgt.device
            )

            for i in range(max_len):
                decoder_input[i] = next_pose
                pos_encoded_input = self.pos_encoder(
                    self.encoder(decoder_input) * np.sqrt(self.ninp)
                )
                decoder_outputs = self.transformer_decoder(
                    pos_encoded_input, encoder_output, tgt_mask=tgt_mask,
                )
                output = self.project(decoder_outputs)
                next_pose = output[i].clone()
                del output
            output = decoder_input
        return output.transpose(0, 1)


class ConvolutionalTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", k=3):
        super(ConvolutionalTransformerEncoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)
        self.k = k
        self.temporal_convolution = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k, padding=0)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        #######################
        # we get (len, n, dim) (120, 64, 1024) as src
        # pad calculated amount of zeros to beginning of src. padding depends on kernel size.
        # padding should change dimensions to (120 + 2P, 64, 1024)
        p = (self.k - 1) // 2
        src = F.pad(input=src, pad=(0, 0, 0, 0,  2 * p, 0), mode='constant', value=0)

        # (len, n, dim) => (n, dim, len)
        src = src.permute((1, 2, 0))

        # pass this padded src to cnn with padding 0. output should still be n x dim x len
        src = self.temporal_convolution(src)  # (n, dim, len) (64, 1024, 120)

        # (n, dim, len) => (len, n, dim)
        src = src.permute((2, 0, 1))

        src = src + self.dropout3(src)
        src = self.norm3(src)

        #######################
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerModelV2(TransformerModel):
    def __init__(self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5, k=3):
        super(TransformerModelV2, self).__init__(ntoken, ninp, num_heads, hidden_dim, num_layers, dropout)
        encoder_layer = ConvolutionalTransformerEncoderLayer(
            ninp, num_heads, hidden_dim, dropout, k=k
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(ninp),
        )