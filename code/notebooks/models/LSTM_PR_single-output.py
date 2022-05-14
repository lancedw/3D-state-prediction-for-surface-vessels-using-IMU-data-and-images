import torch.nn as nn
import torch.nn.functional as F

class PitchPredictor(nn.Module):
    def __init__(self, n_features, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.2
        )

        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        # print(f"input size: {x.shape}")
        self.lstm.flatten_parameters()
        # print(f"flatten size: {x.shape}")
        _, (hidden, _) = self.lstm(x)
        # print(f"hidden size: {hidden.shape}")
        # grab hidden vector from last LSTM layer (hidden size = (1*num_layers, batch size, hidden size))
        out = hidden[-1]
        # print(f"out size: {out.shape}")
        return self.fc(out)