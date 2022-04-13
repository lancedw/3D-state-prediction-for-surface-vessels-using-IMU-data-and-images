import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM_encoder_decoder_PR(nn.Module):
    def __init__(self, cuda = True, encoder_input_size = 10, encoder_hidden_size = 300, decoder_hidden_size = 300,  output_size = 20):
        super(LSTM_encoder_decoder_PR, self).__init__()
        self.cuda_p = cuda
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.encoder_lstm = nn.LSTM(encoder_input_size, encoder_hidden_size, batch_first=True)
        self.decoder_lstm = nn.LSTM(decoder_hidden_size, decoder_hidden_size, batch_first=True)

        self.decoder_fc_1 = nn.Linear(decoder_hidden_size, int(decoder_hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(decoder_hidden_size/2), output_size)


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

        PR  = [pr_s[i] for i in range(use_n_im)]
        lstm_input_features = torch.cat(PR, 1).view(pr_s[0].size(0), 1, -1)

        encoder_output, encoder_hidden = self.LSTM_encoder(lstm_input_features,  encoder_hidden)
        decoder_output, decoder_hidden = self.LSTM_decoder(encoder_output, decoder_hidden)

        decoder_output = decoder_output.view(pr_s[0].size(0), predict_n_pr, -1)

        return decoder_output, encoder_hidden, decoder_hidden