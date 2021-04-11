import torch.nn as nn
import torch
import torch.nn.functional as F


class DensityNet(nn.Module):
    def __init__(self,net_width=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(1792, 1024, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=1024)
        self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=512)
        self.fc1 = nn.Linear(512 * 4 * 5, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1_batchnorm(self.conv1(x))), 2)
        out = torch.tanh(self.conv2_batchnorm(self.conv2(out)))
        out = out.view(-1, 512 * 4 * 5)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out

