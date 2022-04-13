import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_stack_FC_first(nn.Module):
    def __init__(self, cuda = True, num_channel = 3,  cnn_fc_size = 1024, num_output=20):
        super(CNN_stack_FC_first, self).__init__()
        self.cuda_p = cuda
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channel, 8, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        )
        self.fc1 = nn.Linear(5376  , cnn_fc_size) #5376 / 20736
        self.fc2 = nn.Linear(cnn_fc_size, 128)
        self.fc3 = nn.Linear(128, num_output)
        self.dropout0 = nn.Dropout(p=0.3)
        self.dropout1 = nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p= 0.4)


    def forward(self, x, p_and_roll, num_images):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.dropout0(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x =  torch.tanh(self.fc3(x)).view(x.size(0), -1, 2)

        return x