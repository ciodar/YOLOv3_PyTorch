import torch.nn as nn

import torch.nn.functional as F


class DensityNet(nn.Module):
    def __init__(self, net_width=1024):
        super().__init__()
        self.net_width = net_width
        self.conv1 = nn.Conv2d(in_channels=1792, out_channels=net_width, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=net_width)
        self.conv2 = nn.Conv2d(in_channels=net_width, out_channels=net_width, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=net_width)
        self.fc1 = nn.Linear(net_width * 5 * 4, net_width)
        self.fc2 = nn.Linear(net_width, net_width)
        self.fc3 = nn.Linear(net_width, 1)

    def forward(self, x):
        out = self.conv1_batchnorm(self.conv1(x))
        out = F.max_pool2d(F.relu(out), 2)
        out = self.conv2_batchnorm(self.conv2(out))
        out = F.relu(out)
        out = out.view(-1, self.net_width * 4 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        return out

