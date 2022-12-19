import torch.nn as nn
import torch
import torchvision.transforms as T
from utils import get_key_length


class another(nn.Module):
    def __init__(self):
        super(another, self).__init__()

        conv1 = nn.Conv2d(1, 16, 9, padding='same')
        conv2 = nn.Conv2d(16, 32, 7, padding='same')
        conv3 = nn.Conv2d(32, 64, 7, padding='same')
        conv4 = nn.Conv2d(64, 128, 7, padding='same')
        conv5 = nn.Conv2d(128, 256, 7, padding='same')
        conv6 = nn.Conv2d(256, 512, 7, padding='same')
     
        

        self.c1 = nn.Sequential(conv1, nn.ReLU())
        self.c2 = nn.Sequential(conv2, nn.ReLU())
        self.c3 = nn.Sequential(conv3, nn.ReLU())
        self.c4 = nn.Sequential(conv4, nn.ReLU())
        self.c5 = nn.Sequential(conv5, nn.ReLU())
        self.c6 = nn.Sequential(conv6, nn.ReLU())


        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_6 = nn.MaxPool2d(kernel_size=2, padding=1)


        self.batchnorm1 = nn.BatchNorm2d(16)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.batchnorm5 = nn.BatchNorm2d(256)
        self.batchnorm6 = nn.BatchNorm2d(512)


        self.dense = nn.Sequential(nn.Linear(512, get_key_length())) #1536

    def forward(self, batch):
        # rotated_batch = T.RandomRotation(degrees=(0,180))(batch)
        l1 = self.c1(batch)
        l1 = self.batchnorm1(l1)
        l1 = self.max_pool_1(l1)

        # rotated_l1 = T.RandomRotation(degrees=(0,180))(l1)
        l2 = self.c2(l1)
        l2 = self.batchnorm2(l2)
        l2 = self.max_pool_2(l2)

        l3 = self.c3(l2)
        l3 = self.batchnorm3(l3)
        l3 = self.max_pool_3(l3)

        l4 = self.c4(l3)
        l4 = self.batchnorm4(l4)
        l4 = self.max_pool_4(l4)

        l5 = self.c5(l4)
        l5 = self.batchnorm5(l5)
        # l5 = self.max_pool_5(l5)

        l6 = self.c6(l5)
        l6 = self.batchnorm6(l6)
        # l6 = self.max_pool_6(l6)
        
    
        l6 = l6.view(l6.size(0), -1)
        output = self.dense(l6)
        return output
