# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1, lstm=None,
    ):
        """LSTMEncoder encodes input vector using LSTM cells.

        Attributes:
            input_dim: Size of input vector
            hidden_dim: Size of hidden state vector
            num_layers: Number of layers of LSTM units
            lstm: Optional; If provided, the lstm cell will be used in the
                encoder. This is useful for sharing lstm parameters with
                decoder.
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = lstm
        if not lstm:
            assert input_dim is not None
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
            )

            # todo dont hardcode k and feature size
            # todo if changing k here, make sure to change k in forward as well
            k = 3
            # when there is dilation, padding = (dilation * (cnn_kernel_size - 1) // 2
            d = 2
            p = (d * (k - 1)) // 2
            dropout = 0.2
            self.conv1 = weight_norm(nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=k, dilation = d, padding=p))
            self.relu1 = nn.ReLU()
            #self.glu1 = F.glu()
            self.dropout1 = nn.Dropout(dropout)

            self.conv2 = weight_norm(nn.Conv1d(in_channels=input_dim, out_channels=input_dim*2, kernel_size=k, dilation = d, padding=p))
            self.relu2 = nn.ReLU()
            #self.glu2 = F.glu()
            self.dropout2 = nn.Dropout(dropout)

            self.relu = nn.ReLU()

    def forward(self, input):
        """
        Input:
            input: Input vector to be encoded.
                Expected shape is (batch_size, seq_len, input_dim)
        """
        # we get (n, len, dim) (64, 120, 72) as src
        # (n, len, dim) => (n, dim, len)
        init_input = input.permute((0, 2, 1))

        # Residual connection for every convolution
        input = self.conv1(init_input)  # (n, dim, len) (64, 72, 120)
        #input = self.relu1(input)
        input = F.glu(input, dim=1)
        input = self.dropout1(input)
        input = self.relu(init_input + input)

        init_input2 = input
        input = self.conv2(input)  # (n, dim, len)
        #input = self.relu2(input)
        input = F.glu(input, dim=1)
        input = self.dropout2(input)
        input = self.relu(init_input2 + input)

        # Seq2Seq expects src and tgt in format (len, batch_size, dim)
        input = input.permute((2, 0, 1))  # (len, n, dim) (120, 64, 72)

        outputs, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell, outputs.transpose(0, 1)
