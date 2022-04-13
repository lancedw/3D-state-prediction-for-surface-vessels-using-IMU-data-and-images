import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_image_encoder_PR_encoder_decoder(nn.Module):
    def __init__(self, cuda = True, h_dim=2688, z_dim=1024, im_encoder_input_size = 4096, pr_encoder_input_size = 20 , im_encoder_hidden_size = 128, pr_encoder_hidden_size = 128, decoder_hidden_size = 256,  output_size = 20):
        super(CNN_LSTM_image_encoder_PR_encoder_decoder, self).__init__()
        self.cuda_p = cuda
        self.im_encoder_hidden_size = im_encoder_hidden_size
        self.pr_encoder_hidden_size = pr_encoder_hidden_size
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

        self.im_encoder_lstm = nn.LSTM(im_encoder_input_size, im_encoder_hidden_size, batch_first=True)
        self.pr_encoder_lstm = nn.LSTM(pr_encoder_input_size, pr_encoder_hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(decoder_hidden_size, int(decoder_hidden_size/2), batch_first=True)

        self.decoder_fc_1 = nn.Linear(int(decoder_hidden_size/2), int(decoder_hidden_size/4))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/4), output_size)


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


    def LSTM_decoder(self, inputs, hiddens):
        outputs = F.relu(inputs)
        outputs, hiddens = self.decoder_lstm(outputs, hiddens)
        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))
        return outputs, hiddens


    def initHiddenEncoderIm(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.im_encoder_hidden_size)


    def initHiddenEncoderPR(self, n_batch):
        return  torch.zeros(1, n_batch  ,self.pr_encoder_hidden_size)


    def initHiddenDecoder(self, n_batch):
        return  torch.zeros(1, n_batch  ,int(self.decoder_hidden_size/2))


    def forward(self, image_s, pr_s, use_n_im, predict_n_pr, im_encoder_hidden, pr_encoder_hidden, decoder_hidden):

        features = [self.encode( image_s[i] ) for i in range(use_n_im)]
        PR  = [pr_s[i] for i in range(use_n_im)]

        lstm_input_features = torch.cat(features, 1).view(image_s[0].size(0), 1, -1)
        lstm_input_PR = torch.cat(PR, 1).view(image_s[0].size(0), 1, -1)

        encoder_output_images, im_encoder_hidden = self.im_encoder_lstm(lstm_input_features,  im_encoder_hidden )
        encoder_output_PR, pr_encoder_hidden = self.pr_encoder_lstm(lstm_input_PR,  pr_encoder_hidden )

        lstm_input_decoder = torch.cat((encoder_output_images, encoder_output_PR), 2).view(image_s[0].size(0), 1, -1)
        decoder_output, decoder_hidden = self.LSTM_decoder(lstm_input_decoder, decoder_hidden)

        decoder_output = decoder_output.view(image_s[0].size(0), predict_n_pr, -1)

        return decoder_output, im_encoder_hidden, pr_encoder_hidden, decoder_hidden


