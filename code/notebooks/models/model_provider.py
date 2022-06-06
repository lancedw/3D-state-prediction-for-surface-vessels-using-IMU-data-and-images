from notebooks.models.LSTM_single_step import SingleStepPredictor
from notebooks.models.LSTM_enc_dec import LSTM_seq2seq
from notebooks.models.CNN_sequential import CNN_linear
from notebooks.models.CNN_LSTM_img import CNN_LSTM_seq2seq as CNN_single
from notebooks.models.CNN_LSTM_img_PR import CNN_LSTM_seq2seq as CNN_dual

class ModelProvider():
    @staticmethod
    def single_output_lstm(in_features, out_features, n_hidden=128, n_layers=2):
        return SingleStepPredictor(in_features, out_features, n_hidden, n_layers)
    
    @staticmethod
    def sequence_lstm(input_size, output_size, hidden_size=300):
        return LSTM_seq2seq(input_size, output_size, hidden_size)
    
    @staticmethod
    def cnn_linear(linear_input_size, output_size, channels=3):
        return CNN_linear(linear_input_size, output_size, channels)

    @staticmethod
    def cnn_lstm_single(encoder_input_size, output_size, channels=3, hidden_size=1024):
        return CNN_single(encoder_input_size, output_size, channels, hidden_size)

    @staticmethod
    def cnn_lstm_dual(encoder_input_size, output_size, channels=3, hidden_size=1024):
        return CNN_dual(encoder_input_size, output_size, channels, hidden_size)
        