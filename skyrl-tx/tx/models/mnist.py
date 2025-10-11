import torch
import torch.nn as nn
import torch.nn.functional as F


class Mnist(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(p=0.025)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.linear1 = nn.Linear(3136, 256)
        self.dropout2 = nn.Dropout(p=0.025)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.avg_pool2d(F.relu(self.batch_norm1(self.dropout1(self.conv1(x)))), kernel_size=2, stride=2)
        x = F.avg_pool2d(F.relu(self.batch_norm2(self.conv2(x))), kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = F.relu(self.dropout2(self.linear1(x)))
        x = self.linear2(x)
        return x
