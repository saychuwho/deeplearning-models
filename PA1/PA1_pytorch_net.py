import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PytorchThreeLayerNet(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2, n_output):
        super().__init__()

        self.linear1 = nn.Linear(n_input, n_hidden1)
        self.linear2 = nn.Linear(n_hidden1, n_hidden2)
        self.linear3 = nn.Linear(n_hidden2, n_output)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu(x1)

        x3 = self.linear2(x2)
        x4 = self.relu(x3)

        y = self.linear3(x4)

        return y
    

class PytorchCNN(nn.Module):
    def __init__(self, n_output=10):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)     # input size (batch_size, 1, 28, 28)
        self.conv2 = nn.Conv2d(32, 32, 3)               # input size (batch_size, 1, 14, 14)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2,2))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1152, n_output)         # flatten의 결과를 보고 직접 구함

        self.CNNlayers = nn.Sequential(
            self.conv1,
            self.relu,
            self.maxpool,
            self.conv2,
            self.relu,
            self.maxpool
        )


    def forward(self, x):
        x = self.CNNlayers(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x
