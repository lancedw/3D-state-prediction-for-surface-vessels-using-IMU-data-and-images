import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_LSTM_encoder_decoder_images(nn.Module):
    def __init__(self,cuda = True, h_dim=2688, z_dim=1024, encoder_input_size = 4096, encoder_hidden_size = 1024, decoder_hidden_size = 1024,  output_size = 20):
        super(CNN_LSTM_encoder_decoder_images, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

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
        )

        self.mu = nn.Linear(h_dim, z_dim)
        self.std = nn.Linear(h_dim, z_dim)

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()

        esp = torch.randn(*mu.size())
        if self.cuda_p:
            esp = esp.cuda()
        z = mu + std * esp
        return z


    def bottleneck(self, h):
        mu  = self.mu(h)
        logvar = F.relu(self.std(h))
        z = self.reparameterize(mu, logvar)
        return z


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.bottleneck(h)
        return z


    def LSTM_encoder(self, inputs, hiddens):
        outputs, hiddens = self.encoder_lstm(inputs, hiddens)
        return outputs, hiddens


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.decoder_hidden_size)


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden
