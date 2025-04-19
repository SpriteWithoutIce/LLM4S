import torch
import torch.nn as nn

class CNN_TS(nn.Module):
    def __init__(self, args):
        super(CNN_TS, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=args.input_dim, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，减少参数量

        self.classifier = nn.Linear(256, args.num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = self.conv_layers(x)  # 经过 CNN 提取特征
        x = self.global_avg_pool(x)  # (batch_size, 256, 1)
        x = x.squeeze(-1)  # (batch_size, 256)
        x = self.classifier(x)  # 分类
        return x
