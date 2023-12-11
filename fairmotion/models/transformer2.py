# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm, Conv1d, ConvTranspose1d
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

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

class TransformerModel2(nn.Module):
    def __init__(
        self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5
    ):
        super(TransformerModel2, self).__init__()
        self.model_type = "Transformer2"
        self.src_mask = None

        # hard coded number of joints for the model we are using.
        # representation size is the number of values used to represent the joint angle
        # compression layer is a simple 1d convolution that
        self.num_joints = 24
        self.rep_size = int(ntoken / self.num_joints)
        self.compression_layer = Conv1d(
            in_channels=self.num_joints,
            out_channels=self.num_joints,
            kernel_size=self.rep_size,
            stride=self.rep_size,
            # groups=self.num_joints
        )
        self.decompression_layer = ConvTranspose1d(
            in_channels=self.num_joints,
            out_channels=self.num_joints,
            kernel_size=self.rep_size,
            stride=self.rep_size,
            # groups=self.num_joints
        )

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
        self.encoder = nn.Linear(self.num_joints, ninp)
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
        # Transformer expects src and tgt in format (seq_len, batch_size, dim)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # src and tgt are now (seq_len, batch_size, dim)
        if max_len is None:
            max_len = tgt.shape[0]

        # apply input compression
        reshaped_pre_compression = src.reshape(
            src.shape[1],
            self.num_joints,
            self.rep_size * src.shape[0]
        )
        compressed_src = self.compression_layer(reshaped_pre_compression)
        reshaped_post_compression = compressed_src.view(
            src.shape[0],
            src.shape[1],
            self.num_joints
        )

        projected_src = self.encoder(reshaped_post_compression) * np.sqrt(self.ninp)
        pos_encoded_src = self.pos_encoder(projected_src)
        encoder_output = self.transformer_encoder(pos_encoded_src)

        if self.training:

            # Create mask for training
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
                device=tgt.device,
            )

            # Use last source pose as first input to decoder
            tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
            expanded_tgt = tgt.reshape(
                tgt.shape[1],
                self.num_joints,
                self.rep_size * tgt.shape[0]
            )
            compressed_tgt = self.compression_layer(expanded_tgt)
            reshaped_post_compression = compressed_tgt.view(
                tgt.shape[0],
                tgt.shape[1],
                self.num_joints
            ).squeeze(0)
            
            pos_encoder_tgt = self.pos_encoder(
                self.encoder(reshaped_post_compression) * np.sqrt(self.ninp)
            )
            output = self.transformer_decoder(
                pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
            )

            output = self.project(output).transpose(0,1)
        else:
            # greedy decoding
            decoder_input = torch.zeros(
                max_len, src.shape[1], self.num_joints,
            ).type_as(src.data)
            
            next_pose = tgt[0].clone()

            # Create mask for greedy encoding across the decoded output
            tgt_mask = self._generate_square_subsequent_mask(max_len).to(
                device=tgt.device
            )

            for i in range(max_len):
                expanded_next_pose = next_pose.reshape(
                    next_pose.shape[0],
                    self.num_joints,
                    self.rep_size
                )
                compressed_next_pose = self.compression_layer(expanded_next_pose)
                reshaped_post_compression = compressed_next_pose.view(
                    1,
                    expanded_next_pose.shape[0],
                    self.num_joints
                )
            
                decoder_input[i] = reshaped_post_compression.squeeze(0)
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
            output = self.decompression_layer(output.transpose(0,1))
        return output
