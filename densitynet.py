import torch.nn as nn
import torch
import torch.nn.functional as F


class DensityNet(nn.Module):
    def __init__(self,net_width=1024,nclasses=1):
        super().__init__()
        self.nclasses = nclasses
        self.net_width = net_width

        self.conv1 = nn.Conv2d(1792, net_width, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=net_width)
        self.conv2 = nn.Conv2d(net_width, net_width//2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=net_width//2)
        self.fc1 = nn.Linear(net_width//2 * 4 * 5, net_width//2)
        self.fc2 = nn.Linear(net_width//2, 32)
        self.fc3 = nn.Linear(32, nclasses)

    def forward(self, x):
        out = F.max_pool2d(torch.tanh(self.conv1_batchnorm(self.conv1(x))), 2)
        out = torch.tanh(self.conv2_batchnorm(self.conv2(out)))
        out = out.view(-1, 512 * 4 * 5)
        out = torch.tanh(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        return out



