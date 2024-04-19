import torch
import torch.nn as nn


# Define ConvLSTM cell
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_channels,
                              out_channels=4 * self.hidden_channels,
                              kernel_size=self.kernel_size,
                              padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


# Define ConvLSTM model
class ConvLSTMModel(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(ConvLSTMModel, self).__init__()
        self.hidden_channels = [hidden_channels] * num_layers
        self.num_layers = num_layers
        self.conv_lstm = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = input_channels if i == 0 else hidden_channels
            self.conv_lstm.append(ConvLSTMCell(input_dim, self.hidden_channels[i], kernel_size))
        self.conv_out = nn.Conv2d(hidden_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Initialize hidden state and cell state
        batch_size, seq_len, _, height, width = x.size()
        h, c = [torch.zeros(batch_size, hidden, height, width).to(x.device) for hidden in self.hidden_channels], \
            [torch.zeros(batch_size, hidden, height, width).to(x.device) for hidden in self.hidden_channels]

        # Forward propagate ConvLSTM
        for t in range(seq_len):
            input_tensor = x[:, t, :, :, :]
            for i in range(self.num_layers):
                h[i], c[i] = self.conv_lstm[i](input_tensor, (h[i], c[i]))
                input_tensor = h[i]

        # Only use the last output for classification
        out = self.conv_out(h[-1])
        return nn.Sigmoid()(out)
        # return out

