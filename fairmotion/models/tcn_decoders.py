# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Reference: Sequence Modeling Benchmarks and Temporal Convolutional Networks(TCN) https://github.com/locuslab/TCN
class DecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, lstm=None):
        super(DecoderStep, self).__init__()
        # input_dim: 72, hidden_dim: 1024, output_dim: 72
        self.out1 = nn.Linear(input_dim, hidden_dim)
        self.out2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, encoder_outputs=None):
        # print("input size: ", input.size())
        input = input.squeeze(0)
        input = self.out1(input)
        output = self.out2(input)
        return output


class TCNDecoder(nn.Module):
    """Decoder of TCN

    Attributes:
        input_dim: Size of input vector
        output_dim: Size of output to be generated at each time step
        hidden_dim: Size of hidden state vector
        device: Optional; Device to be used "cuda" or "cpu"
        lstm: Optional; If provided, the lstm cell will be used in the decoder.
            This is useful for sharing lstm parameters from encoder.
    """

    def __init__(
        self, input_dim, output_dim, hidden_dim, device="cuda", lstm=None
    ):
        super(TCNDecoder, self).__init__()
        self.input_dim = input_dim
        self.decoder_step = DecoderStep(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lstm=None,
        )
        self.device = device


    def forward(
        self,
        tgt,
        max_len=None,
        teacher_forcing_ratio=0.5,
    ):
        """
        Inputs:
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (seq_len, batch_size, input_dim)
            hidden, cell: Hidden state and cell state to be used in LSTM cell
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        tgt = tgt.transpose(0, 1)
        max_len = max_len if max_len is not None else tgt.shape[0]
        batch_size = tgt.shape[1]

        input = tgt[0, :]
        outputs = torch.zeros(max_len, batch_size, self.input_dim,).to(
            self.device
        )
        for t in range(max_len):
            input = input.unsqueeze(0)
            output = self.decoder_step(input)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = tgt[t] if teacher_force else output

        outputs = outputs.transpose(0, 1)
        return outputs


