import cv2
import json
import numpy as np
import pandas as pd
from notebooks.data_loaders.utils import Utilities
from notebooks.data_loaders.sequence_generator import SequenceGenerator

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader

class CustomDataLoader():
    def __init__(self, folder, n_episodes, load_images=False):
        self.folder = folder
        self.pr_data = []
        self.image_tensor = torch.empty(0)

        self.load_data(n_episodes, load_images)

    def load_data(self, n_episodes, load_images):
        # load all data
        episodes_data = []
        img_tensors_array = []

        for ep in range(1, n_episodes+1):
            print(f"loading ep {ep}...", end='\r')
            folder = self.folder + str(ep) + "/"
            filename = folder + "labels_0.json"
            labels = json.load(open(filename))
            
            for i in labels:
                # pitch and roll is read with labels[i] as [pitch, roll]
                episodes_data.append(labels[i])

            if(load_images):
                for i in labels:
                    # load image, normalize and convert to tensor
                    img = cv2.imread(folder + str(i) + ".png")
                    img = Utilities.norm_pixel(img)
                    img_tensors_array.append(torch.Tensor(img))
        
        self.pr_data = pd.DataFrame(episodes_data)
        self.pr_data.rename(columns = {0:'pitch', 1:'roll'}, inplace=True)
        self.pr_data = self.pr_data.astype({"pitch": float, "roll": float})
        self.pr_data['pitch'] = self.pr_data['pitch'].apply(lambda x : Utilities.norm_pr(x))
        self.pr_data['roll'] = self.pr_data['roll'].apply(lambda x : Utilities.norm_pr(x))
        self.pr_data.head()

        if(load_images):
            self.image_tensor = torch.stack(img_tensors_array)
            self.image_tensor = self.image_tensor.permute(0,3,1,2)

    

