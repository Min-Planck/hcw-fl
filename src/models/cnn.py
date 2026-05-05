import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate(size, kernel, stride, padding):
  return int(((size+(2*padding)-kernel)/stride) + 1)

class CNN(nn.Module):
    def __init__(self, in_channels=1, im_size=[28, 28], num_classes=10):
        super(CNN, self).__init__()
        outsize = im_size[0]
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        outsize = calculate(outsize, kernel=5, stride=1, padding=2)
        outsize = calculate(outsize, kernel=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        outsize = calculate(outsize, kernel=5, stride=1, padding=2)
        outsize = calculate(outsize, kernel=2, stride=2, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * outsize * outsize, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1) 
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
