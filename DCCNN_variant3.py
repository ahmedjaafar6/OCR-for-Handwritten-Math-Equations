import torch.nn as nn
import torch
import torchvision.transforms as T
from utils import get_key_length


class DCCNN(nn.Module):
    def __init__(self):
        super(DCCNN, self).__init__()

        conv11 = nn.Conv2d(1, 64, 9, padding='same')
        conv12 = nn.Conv2d(1, 64, 9, padding='same')
        
        conv21 = nn.Conv2d(128, 128, 7, padding='same') 
        conv22 = nn.Conv2d(128, 128, 7, padding='same')
        conv23 = nn.Conv2d(128, 128, 7, padding='same')
        conv24 = nn.Conv2d(128, 128, 7, padding='same')

        conv31 = nn.Conv2d(512, 512, 5, padding='same')
        conv32 = nn.Conv2d(512, 512, 5, padding='same')
        conv33 = nn.Conv2d(512, 512, 5, padding='same')
        conv34 = nn.Conv2d(512, 512, 5, padding='same')
        conv35 = nn.Conv2d(512, 512, 5, padding='same')
        conv36 = nn.Conv2d(512, 512, 5, padding='same')
        

        self.fork_1_1 = nn.Sequential(conv11, nn.ReLU())
        self.fork_1_2 = nn.Sequential(conv12, nn.ReLU())
        
        self.fork_2_1 = nn.Sequential(conv21, nn.ReLU())
        self.fork_2_2 = nn.Sequential(conv22, nn.ReLU())
        self.fork_2_3 = nn.Sequential(conv23, nn.ReLU())
        self.fork_2_4 = nn.Sequential(conv24, nn.ReLU())

        self.fork_3_1 = nn.Sequential(conv31, nn.ReLU())
        self.fork_3_2 = nn.Sequential(conv32, nn.ReLU())
        self.fork_3_3 = nn.Sequential(conv33, nn.ReLU())
        self.fork_3_4 = nn.Sequential(conv34, nn.ReLU())
        self.fork_3_5 = nn.Sequential(conv35, nn.ReLU())
        self.fork_3_6 = nn.Sequential(conv36, nn.ReLU())
   
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2) 
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, padding=1)


        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(512)
        self.batchnorm3 = nn.BatchNorm2d(3072)

        self.dense = nn.Sequential(nn.Linear(49152, get_key_length())) #1536

    def forward(self, batch):
        x1 = self.fork_1_1(batch)
        x2 = self.fork_1_2(batch)
        l1 = torch.cat((x1, x2), dim=1)
        l1 = self.batchnorm1(l1)
        l1 = self.max_pool_1(l1)

        x1 = self.fork_2_1(l1)
        x2 = self.fork_2_2(l1)
        x3 = self.fork_2_3(l1)
        x4 = self.fork_2_4(l1)
        l2 = torch.cat((x1, x2, x3, x4), dim=1)
        l2 = self.batchnorm2(l2)
        l2 = self.max_pool_2(l2)

        x1 = self.fork_3_1(l2)
        x2 = self.fork_3_2(l2)
        x3 = self.fork_3_3(l2)
        x4 = self.fork_3_4(l2)
        x5 = self.fork_3_5(l2)
        x6 = self.fork_3_6(l2)
        l3 = torch.cat((x1, x2, x3, x4, x5, x6), dim=1)
        l3 = self.batchnorm3(l3)
        l3 = self.max_pool_3(l3)

    
        l3 = l3.view(l3.size(0), -1)
        output = self.dense(l3)
        return output
