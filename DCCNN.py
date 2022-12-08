import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.fork1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=64,
                      kernel_size=5,
                      stride=1,
                      padding='same'
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.fork2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=1,
                      padding='same'
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.fork3 = nn.Sequential(
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      stride=1,
                      padding='same'
                      ),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding='same')

        self.dropout = nn.Dropout(p=0.5)

        # Fully connected layer
        self.out = nn.Sequential(
                nn.Linear(1080, 10),  # fix dims later
                nn.Softmax()
        )


    def forward(self, batch):
        x1=self.fork1(batch)
        x2=self.fork1(batch)
        layer1=torch.cat((x1, x2), dim=1)
        layer1=self.maxpool(layer1)
        x1=self.fork2(layer1)
        x2=self.fork2(layer1)
        layer2=torch.cat((x1, x2), dim=1)
        layer2=self.maxpool(layer2)
        x1=self.fork3(layer2)
        x2=self.fork3(layer2)
        x3=self.fork3(layer2)
        x4=self.fork3(layer2)
        x5=self.fork3(layer2)
        x6=self.fork3(layer2)
        layer3=torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        layer3=self.maxpool(layer3)

        layer3=layer3.view(layer3.size(0), -1)  # maybe not needed
        dropout=self.dropout(layer3)
        output=self.out(dropout)
        return output