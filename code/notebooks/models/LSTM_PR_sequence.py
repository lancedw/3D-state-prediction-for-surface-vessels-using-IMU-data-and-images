import torch
import torch.nn as nn
import torch.nn.functional as F

# ENCODER
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=300, num_layers=1):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        self.encoder_lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, input_seq):
        ouputs, hidden = self.encoder_lstm(input_seq)
        
        return ouputs, hidden

# DECODER
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size=300, num_layers = 1):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.decoder_lstm = nn.LSTM(
            input_size = hidden_size, 
            hidden_size = hidden_size,
            num_layers = num_layers, 
            batch_first = True
        )

        self.decoder_fc_1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.decoder_fc_2 = nn.Linear(int(hidden_size/2), output_size)

        #self.linear = nn.Linear(hidden_size, output_size)           

    def forward(self, x_input, hidden):
        outputs = F.relu(x_input)

        outputs, hidden = self.decoder_lstm(outputs, hidden)

        outputs = F.relu(self.decoder_fc_1(outputs))
        outputs = torch.tanh(self.decoder_fc_2(outputs))

        return outputs, hidden

    # Wrapper class for encoder and decoder
class LSTM_seq2seq(nn.Module):    
    def __init__(self, input_size, output_size, hidden_size = 300):
        super(LSTM_seq2seq, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.encoder = Encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = Decoder(output_size = output_size, hidden_size = hidden_size)

    def forward(self, input_tensor):
        encoder_outputs, encoder_hidden  = self.encoder.forward(input_tensor)

        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_outputs, encoder_hidden)

        return decoder_outputs