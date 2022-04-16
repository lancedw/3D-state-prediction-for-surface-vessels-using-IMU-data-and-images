import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_stack_FC(nn.Module):
    def __init__(self, num_channel = 3,  cnn_fc_size = 1024, num_output=20, h_dim=2688, z_dim=1024):
        super(CNN_stack_FC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.fc1 = nn.Linear(5376  , cnn_fc_size) 
        self.fc2 = nn.Linear(cnn_fc_size, 512) 
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_output)

        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.4)
        self.dropout3 = nn.Dropout(p=0.4)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.mu(h), F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def forward(self, x, p_and_roll, num_images):

        x = self.encode(x).view(x.size(0), 1, -1)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.tanh(self.fc4(x)).view(x.size(0), -1, 2)

        return x
