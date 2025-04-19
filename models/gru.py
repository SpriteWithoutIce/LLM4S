import torch.nn as nn

class GRU4TS(nn.Module):
    def __init__(self, args, gru_layers=2, hidden_size=256):
        super(GRU4TS, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=args.input_dim, hidden_size=hidden_size, num_layers=gru_layers, bidirectional=True, batch_first=True)

        # Classifier
        self.classifier = nn.Linear(hidden_size * 2, args.num_classes)

    def forward(self, x):
        gru_output, _ = self.gru(x)  # (batch_size, seq_len, hidden_size * 2)
        cls_embedding = gru_output[:, 0, :]  # (batch_size, hidden_size * 2)
        logits = self.classifier(cls_embedding)  # (batch_size, num_classes)
        return logits