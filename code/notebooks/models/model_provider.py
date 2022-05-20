from LSTM_PR_single_step import SingleStepPredictor
from LSTM_PR_sequence import LSTM_seq2seq
from CNN_linear_img_to_PR import CNN_linear
from CNN_LSTM_img_to_PR import CNN_LSTM_seq2seq

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
    def cnn_lstm(encoder_input_size, output_size, hidden_size = 1024):
        return CNN_LSTM_seq2seq(encoder_input_size, output_size, hidden_size)
        