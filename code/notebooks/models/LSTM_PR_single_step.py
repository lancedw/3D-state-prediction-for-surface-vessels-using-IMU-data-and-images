import torch.nn as nn
import torch.nn.functional as F

class SingleStepPredictor(nn.Module):

    # INPUT LAYER (2 features) --> LSTM 128 hidden --> LSTM 128 hidden --> Linear --> OUTPUT (2 features)
    def __init__(self, in_features, out_features, n_hidden=128, n_layers=2):
        super().__init__()

        self.n_hidden = n_hidden

        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.2
        )

        self.regressor = nn.Linear(in_features=n_hidden, out_features=out_features)

    def forward(self, x):
        # call flatten_parameters function to aggregate all the weight 
        # tensors into continuous space of GPU memory
        self.lstm.flatten_parameters()
    
        output, (hidden_state, cell_state) = self.lstm(x)
        out = hidden_state[-1]

        return self.regressor(out)