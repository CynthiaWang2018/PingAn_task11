import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self, input_sizes=[10, 3], hidden_size=[64, 128], num_classes=1, num_layer=3):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layer = num_layer
        self.para_size = input_sizes[0]
        self.ques_size = input_sizes[1]
        self.conv1 = nn.Conv1d(1, self.hidden_size[0], kernel_size=3) #in_channels, out_channels, kernel_size
        self.embeding = nn.Linear(self.ques_size, self.hidden_size[1], bias=True)
        self.rnn = nn.LSTM( # if use nn.RNN(), it hardly learns
            input_size=self.hidden_size[0],
            hidden_size=self.hidden_size[1], # rnn hidden unit
            num_layers=self.num_layer, # number of rnn layer
            batch_first=True, # input & output will has batch size as 1s dimension. e.g.(batch, time_step, input_size)
        )
        self.out = nn.Linear(self.hidden_size[1], self.num_classes)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        # x: (n, 10)
        # h: (n, 3)
        x = x.unsqueeze(1)
        h = self.embeding(h)
        h = h.unsqueeze(0)
        h = h.expand(self.num_layer, -1, -1).contiguous()
        # print(h.size())
        # x: (n, 1, 10)
        # h: (n, hidden_size)
        x = self.conv1(x)
        # (n, 3, 7)
        x = x.permute(0, 2, 1)
        # (n, 7, 3)
        # r_out shape (batch, time_step, output_size/hidden_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_szie)
        r_out, (h_n, h_c) = self.rnn(x, (h, h))  # None represents zero initial hidden state
        # print(h_n.size(), h_c.size())
        # h_n, h_c = h_n.permute(1, 0, 2), h_c.permute(1, 0, 2)
        out = self.out(r_out[:, -1, :])
        out = self.Sigmoid(out)
        return out