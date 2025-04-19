import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from torch.nn import Parameter


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x+self.pe[:, :x.size(1)]
        return x


class TimeSeriesEmbedding(nn.Module):
    def __init__(self, window=10, cycle_length=16, d_model=256, d_LLM=768, device="cuda:0"):
        super(TimeSeriesEmbedding, self).__init__()
        self.d_LLM = d_LLM
        self.device = device
        self.window = window
        self.d_model = d_model
        self.cycle_length = cycle_length

        # Trend Component
        self.trend = Parameter(torch.zeros(1, 1, 1, window))

        # Seasonal Component
        self.season = Parameter(torch.zeros(
            d_model // cycle_length, d_model // cycle_length, 1, 1))

        self.linear1 = nn.Linear(d_model, d_LLM // 3)

        # Long-Term Component
        self.long_term_out_dim = d_LLM // 3
        self.long_term_linear = nn.Linear(1, self.long_term_out_dim)

    def forward(self, x):
        b, n, l = x.shape
        x = x.unsqueeze(1)  # b,1,n,l
        b, c, n, l = x.shape

        # Trend
        k_trend = torch.softmax(self.trend, dim=-1)
        x_pad = F.pad(x, (self.window - 1, 0), "constant", 0)
        x_trend = F.conv2d(x_pad.reshape(b * c, 1, n, -1),
                           k_trend).reshape(b, c, n, l)
        x_trend = x_trend.squeeze(1)  # [batch, n, 256]
        x_trend_ori = x_trend
        x_trend = self.linear1(x_trend.reshape(
            b * n, l)).reshape(b, n, self.d_LLM // 3)

        # Seasonal
        x_season = x - x_trend_ori.unsqueeze(1)
        x_season = x_season.reshape(b*c, n, -1, self.cycle_length)
        x_season = F.conv2d(x_season.permute(0, 2, 3, 1),
                            self.season).permute(0, 3, 1, 2)
        x_season = x_season.reshape(b, c, n, l)
        x_season = x_season.squeeze(1)
        x_season = self.linear1(x_season.reshape(
            b * n, l)).reshape(b, n, self.d_LLM // 3)

        # Long-Term
        x_mean = torch.mean(x.squeeze(1), dim=-1, keepdim=True)
        x_long = x_mean.repeat(1, 1, self.d_LLM - self.d_LLM // 3 * 2)

        # x_trend = x_trend.reshape(b * n, -1)
        # x_season = x_season.reshape(b * n, -1)
        # x_long = x_long.reshape(b * n, -1)
        E_all = torch.cat([x_trend, x_season, x_long], dim=2)

        return E_all
