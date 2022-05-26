from models.LSTM_PR_single_step import SingleStepPredictor
from models.LSTM_PR_sequence import LSTM_seq2seq
from models.CNN_linear_img_to_PR import CNN_linear
from models.CNN_LSTM_img_to_PR import CNN_LSTM_seq2seq as cnn_single
from models.CNN_LSTM_img_PR_to_PR import CNN_LSTM_seq2seq as cnn_dual

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
    def cnn_lstm_single(encoder_input_size, output_size, hidden_size = 1024):
        return cnn_single(encoder_input_size, output_size, hidden_size)

    @staticmethod
    def cnn_lstm_dual(encoder_input_size, output_size, hidden_size = 1024):
        return cnn_dual(encoder_input_size, output_size, hidden_size)
        