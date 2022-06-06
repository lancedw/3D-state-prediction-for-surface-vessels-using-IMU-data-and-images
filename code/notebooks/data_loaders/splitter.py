import random
import numpy as np

class DataSplitter():

    @staticmethod
    def split_sequences(all_sequences:list, train_size=0.8):
        # Grab random subsets from all sequences for training and test data (without overlapping data)
        n_sequences = len(all_sequences)
        data_indices = list(np.arange(0, n_sequences, 1))

        # train indices are random sample from all data indices
        random.seed(42)
        train_size = int(train_size * n_sequences)
        train_indices = random.sample(data_indices, train_size)

        # test indices are the difference of all data indices and train indices
        test_indices = list(set(data_indices) - set(train_indices))

        random.shuffle(train_indices)
        random.shuffle(test_indices)

        print("Training size:", len(train_indices),"| Test size:", len(test_indices))
        print(train_indices[:10])
        print(test_indices[:10])

        train_sequences = []
        test_sequences = []

        for idx in train_indices:
            input = all_sequences[idx][0]
            output = all_sequences[idx][1]
            train_sequences.append((input, output))

        for idx in test_indices:
            input = all_sequences[idx][0]
            output = all_sequences[idx][1]
            test_sequences.append((input, output))

        return train_sequences, test_sequences