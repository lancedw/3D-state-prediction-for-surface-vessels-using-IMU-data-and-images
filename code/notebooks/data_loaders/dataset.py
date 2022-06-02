import torch
from torch.utils.data import Dataset

class SinglePRDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        input_sequence, output_sequence = self.sequences[index]

        # single step format with squeeze
        return dict(
            input = torch.Tensor(input_sequence.to_numpy()),
            output = torch.Tensor(output_sequence.to_numpy().squeeze()),
        )
        
class SequencePRDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        input_sequence, output_sequence = self.sequences[index]

        # sequence PR only format
        return dict(
            input = torch.Tensor(input_sequence.to_numpy()),
            output = torch.Tensor(output_sequence.to_numpy())
        )

class ImgToPRDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        input_sequence, output_sequence = self.sequences[index]
        
        # image only format
        return dict(
            input = input_sequence,
            output = torch.Tensor(output_sequence.to_numpy())
        )



class ImgPRToPRDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        (img_sequence, pr_sequence), labels = self.sequences[index]

        return dict(
            input = (img_sequence, torch.Tensor(pr_sequence.to_numpy())),
            output = torch.Tensor(labels.to_numpy())
        )