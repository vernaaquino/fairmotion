# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# Reference: Sequence Modeling Benchmarks and Temporal Convolutional Networks(TCN) https://github.com/locuslab/TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, input):
        # (n, len, dim) => (n, dim, len)
        out = self.net(input)
        res = input if self.downsample is None else self.downsample(input)
        outputs = self.relu(out + res)
        return outputs


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """Todo: Clarify num_inputs and num_channels in our dataset
        Attributes:
            num_inputs: ...
            num_channels: ...
            kernel_size:
            dropout:
        """
        super(TemporalConvNet, self).__init__()

        # Todo: If changing k here, make sure to change k in forward as well
        kernel_size = 3
        dilation_size = 2
        # padding = (kernel_size - 1) * dilation_size
        dropout = 0.2

        layers = []
        # num_levels is the number of residual blocks
        # let's put num_levels 3 according to the calculation of
        # https://unit8.com/resources/temporal-convolutional-networks-and-forecasting/
        # num_levels = len(num_channels)
        num_levels = 3
        for i in range(num_levels):
            dilation_size = 2 ** i
            # in_channels = num_inputs if i == 0 else num_channels[i-1]
            # out_channels = num_channels[i]
            layers += [TemporalBlock(n_inputs=72, n_outputs=72, kernel_size=kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, input):
        """
                Input:
                    input: Input vector to be encoded.
                        Expected shape is (batch_size, seq_len, input_dim)
        """

        return self.network(input)

class TCNEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1, lstm=None,
    ):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=72, num_channels=3, kernel_size=3, dropout=0.2)
        # Todo: confirm num_inputs = input_dim in our dataset
        # put num_channels as the number of conv block

    def forward(self, input):
        # we get (n, len, dim) (64, 120, 72) as src
        # (n, len, dim) => (n, dim, len)
        input = input.permute((0, 2, 1))
        outputs = self.tcn(input)
        return outputs.transpose(0, 1)
