# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.nn.utils import weight_norm


class DecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, lstm=None):
        super(DecoderStep, self).__init__()
        self.lstm = (
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
            if not lstm
            else lstm
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None, cell=None, encoder_outputs=None):
        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(input)
        else:
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoder(nn.Module):
    """Decoder to generate sequences using LSTM cells. Decoding is done in a
    greedy manner without attention mechanism.

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
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.decoder_step = DecoderStep(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lstm=lstm,
        )
        self.device = device

        # Added for convolutional seq2seq
        # todo if changing k here, make sure to change k in forward as well
        k = 3
        # when there is dilation, padding = (dilation * (cnn_kernel_size - 1) // 2
        d = 1
        p = (d * (k - 1)) // 2
        dropout = 0.2
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim * 2, kernel_size=k, dilation=d, padding=p)
        self.relu1 = nn.ReLU()
        # self.glu1 = F.glu()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=input_dim, out_channels=input_dim * 2, kernel_size=k, dilation=d, padding=p)
        self.relu2 = nn.ReLU()
        # self.glu2 = F.glu()
        self.dropout2 = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(
        self,
        tgt,
        hidden=None,
        cell=None,
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

        # hidden size: [1, batch_size, hidden_dim], cell size: [1, batch_size, hidden_dim]
        # tgt size: [batch_size, 24, input_dim]
        tgt = tgt.transpose(0, 1)
        # tgt size: [24, batch_size, input_dim]
        max_len = max_len if max_len is not None else tgt.shape[0]
        batch_size = tgt.shape[1]
        """
        # Added for convolutional seq2seq
        input = tgt
        # input size: [24, batch_size, input_dim] (24, 64, 72)
        init_input = input.permute((1, 2, 0))
        # input size: [batch_size, input_dim, 24] [64, 72, 24]

        # Residual connection for every convolution
        input = self.conv1(init_input)  # (n, dim, 24) (64, 72, 24)
        # input = self.relu1(input)
        input = F.glu(input, dim=1)
        input = self.dropout1(input)
        input = self.relu(init_input + input)

        init_input2 = input
        input = self.conv2(input)  # (n, dim, len)
        # input = self.relu2(input)
        input = F.glu(input, dim=1)
        input = self.dropout2(input)
        input = self.relu(init_input2 + input)

        # Seq2Seq expects src and tgt in format (len, batch_size, dim)
        input = input.permute((2, 0, 1))  # (len, n, dim) (24, 64, 72)
        # Added for convolutional seq2seq
        """
        input = tgt[0, :]
        # input size: [batch_size, input_dim]

        outputs = torch.zeros(max_len, batch_size, self.input_dim,).to(
            self.device
        )
        # output size: [24, batch_size, input_dim]

        for t in range(max_len):
            input = input.unsqueeze(0)
            output, hidden, cell = self.decoder_step(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = tgt[t] if teacher_force else output

        outputs = outputs.transpose(0, 1)
        return outputs


class DecoderStepWithAttention(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, source_length, device="cuda",
    ):
        super(DecoderStepWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.source_length = source_length
        self.device = device

        self.attn = nn.Linear(
            self.hidden_dim + self.input_dim, self.source_length,
        )
        self.attn_combine = nn.Linear(
            self.hidden_dim + self.input_dim, self.input_dim,
        )
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden), 2)), dim=2,
        )
        attn_applied = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs)

        output = torch.cat((input, attn_applied.transpose(0, 1)), 2)
        output = self.attn_combine(output)
        output = F.relu(output)

        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(output)
        else:
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoderWithAttention(LSTMDecoder):
    def __init__(
        self,
        input_dim,
        output_dim,
        max_source_length,
        hidden_dim=128,
        device="cuda",
    ):
        """Extension of LSTMDecoder that uses attention mechanism to generate
        sequences.

        Attributes:
            input_dim: Size of input vector
            output_dim: Size of output to be generated at each time step
            max_source_length: Length of source sequence
            hidden_dim: Size of hidden state vector
            device: Optional; Device to be used "cuda" or "cpu"
        """
        super(LSTMDecoderWithAttention, self).__init__(
            input_dim, output_dim, hidden_dim, device
        )
        self.decoder_step = DecoderStepWithAttention(
            input_dim, output_dim, hidden_dim, max_source_length
        )
        self.device = device
