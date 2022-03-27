import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, cuda = True, num_channel=3, h_dim=2688, z_dim=1024):
        super(AutoEncoder, self).__init__()
        self.cuda_p = cuda
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),

            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),
        )

        # self.fc0 = nn.Linear(h_dim, int(h_dim/2))
        # self.dropout0 = nn.Dropout(p=0.1)
        # self.fc00 = nn.Linear(int(h_dim/2), int(h_dim/2))
        # self.dropout00 = nn.Dropout(p=0.05)

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)


        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding = 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding = (0,1)),
            # nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding= 1, output_padding = 0),
            # nn.BatchNorm2d(3),
            nn.Tanh()
        )


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.relu(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 32, 7, 12)
        z = self.decoder(z)
        return z


    def forward(self, x):
        features = self.encode(x)
        z = self.decode(features)
        return features, z
