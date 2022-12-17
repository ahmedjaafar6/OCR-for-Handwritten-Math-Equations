import torch.nn as nn
from utils import get_key_length

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Covolutional layer
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels=1,
                                  out_channels=16,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=64,
                                  out_channels=120,
                                  kernel_size=5,
                                  stride=1,
                                  padding=2
                                  ),
                        nn.BatchNorm2d(120),
                        nn.ReLU(),
                        
        )
        #Fully connected layer
        
        self.out = nn.Linear(1080, get_key_length())

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
