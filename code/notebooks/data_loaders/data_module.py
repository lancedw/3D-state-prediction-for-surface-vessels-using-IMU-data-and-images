from msilib import sequence
from re import S
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from notebooks.data_loaders.dataset import SequencePRDataset, SinglePRDataset, ImgToPRDataset, ImgPRToPRDataset

class DataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batchsize, n_workers, type):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batchsize = batchsize
        self.n_workers = n_workers
        self.type = type

    def setup(self):
        if(self.type == 1):
            self.train_dataset = SinglePRDataset(self.train_sequences)
            self.test_dataset = SinglePRDataset(self.test_sequences)
        if(self.type == 2):
            self.train_dataset = SequencePRDataset(self.train_sequences)
            self.test_dataset = SequencePRDataset(self.test_sequences)
        if(self.type == 3):
            self.train_dataset = ImgToPRDataset(self.train_sequences)
            self.test_dataset = ImgToPRDataset(self.test_sequences)
        if(self.type == 4):
            self.train_dataset = ImgPRToPRDataset(self.train_sequences)
            self.test_dataset = ImgPRToPRDataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batchsize,
            shuffle = False,
            num_workers=self.n_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batchsize,
            shuffle = False,
            num_workers=self.n_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers=self.n_workers,
        )