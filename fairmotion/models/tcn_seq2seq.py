# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
from fairmotion.models import tcn_decoders, tcn_encoders

# Reference: Sequence Modeling Benchmarks and Temporal Convolutional Networks(TCN) https://github.com/locuslab/TCN
class TCNSeq2Seq(nn.Module):
    """TCN for sequence generation. The interface takes predefined
    encoder and decoder as input.

    Attributes:
        encoder: Pre-built encoder (tcn_encoder)
        decoder: Pre-built decoder (tcn_dncoder)
    """

    def __init__(self, encoder, decoder):
        super(TCNSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src: Source sequence provided as input to the encoder.
                Expected shape: (batch_size, seq_len, input_dim)
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (batch_size, seq_len, input_dim)
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        hidden = self.encoder(src)
        outputs = self.decoder(
            tgt, max_len, teacher_forcing_ratio,
        )
        return outputs

