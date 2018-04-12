import math

import torch
from torch.nn import Parameter
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import functional as F


class mLSTM(RNNBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=input_size, hidden_size=input_size,
                 num_layers=1, bias=bias, batch_first=True,
                 dropout=0, bidirectional=False)

        w_mx = torch.Tensor(hidden_size, input_size)
        w_mh = torch.Tensor(hidden_size, hidden_size)
        self.w_mx = Parameter(w_mx)
        self.w_mh = Parameter(w_mh)

        self.lstm_cell = LSTMCell(input_size, hidden_size, bias)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx=None):
        n_batch, n_seq, n_feat = input.size()

        assert hx is not None
        # Fix two things. First input and first hidden state should not be zero!

        hx, cx = hx
        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq):
            # ToDo: Think about adding biases here
            mx = F.linear(input[:, seq, :], self.w_mx) * F.linear(hx, self.w_mh)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        return torch.cat(steps, dim=1)
