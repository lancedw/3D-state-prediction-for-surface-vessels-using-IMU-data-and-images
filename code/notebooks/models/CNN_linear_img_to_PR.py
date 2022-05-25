import torch
import torch.nn as nn

class CNN_encoder(nn.Module):

    def __init__(self, channels=3):
        super(CNN_encoder, self).__init__()

        # outputs a tensor of shape (batch, out_channels, h/2/2/2, w/2/2/2)
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
    
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        # )
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        # )
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        batch_size = x.size(0)

        # x is of shape (batch, seq_lenth, channels, height, width)
        # iterate over the sequence and encode a feature vector for each image
        img_features = [self.cnn_encoder(x[:,i,:,:,:].squeeze(1)) for i in range(x.size(1))]
        
        # flatten feature vectors in img_features
        img_features = [img.reshape(batch_size, -1) for img in img_features]

        # convert list of vectors into tensor of shape: (batch, len(img_features), features)
        img_features = torch.stack(img_features, 1)

        # reshape into (batch, 1, n*features)
        img_features = img_features.reshape(batch_size, -1).unsqueeze(1)

        return img_features
    
    def test(self, x):
        x = self.conv1(x)
        print("conv 1", x.shape)
        x = self.pool1(x)
        print("pool 1", x.shape)

        x = self.conv2(x)
        print("conv 2", x.shape)
        x = self.pool2(x)
        print("pool 2", x.shape)

        x = self.conv3(x)
        print("conv 3", x.shape)
        x = self.pool3(x)
        print("pool 3", x.shape)

class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, int(input_size/2))
        self.fc2 = nn.Linear(int(input_size/2), int(input_size/4))
        self.fc3 = nn.Linear(int(input_size/4), output_size*2)

    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

class CNN_linear(nn.Module):
    def __init__(self, linear_input_size, output_size, channels=3):
        super(CNN_linear, self).__init__()

        self.output_size = output_size

        self.cnn_encoder = CNN_encoder(channels)
        self.linear = Linear(linear_input_size, output_size)

    def forward(self, x:torch.Tensor):
        batch_size = x.size(0)

        x = self.cnn_encoder(x)
        x = self.linear(x)

        output = x.reshape(batch_size, 2, self.output_size)

        return output